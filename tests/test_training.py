"""Fitting an activity model, and the splitting protocol around it."""

import unittest

import numpy as np
from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import Effector
from cas13guide.models.features import FeatureSpec
from cas13guide.services.training import (
    TrainingExample,
    compare_splitting,
    cross_validate,
    fit_model,
    fit_ridge_logistic,
    group_kfold,
    three_way_group_split,
    train_and_calibrate,
)
from cas13guide.simulate import simulate_screen
from cas13guide.util.duplex import profile_duplex


class TestRidgeLogistic(unittest.TestCase):
    def test_recovers_a_known_linear_signal(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(600, 5))
        true_w = np.array([2.0, -1.5, 0.0, 0.8, 0.0])
        p = 1.0 / (1.0 + np.exp(-(X @ true_w + 0.3)))
        w, b = fit_ridge_logistic(X, p, alpha=1e-4)
        np.testing.assert_allclose(w, true_w, atol=0.15)
        self.assertAlmostEqual(b, 0.3, delta=0.15)

    def test_ridge_shrinks_coefficients(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 5))
        y = 1.0 / (1.0 + np.exp(-(X @ np.array([3.0, 0, 0, 0, 0]))))
        weak, _ = fit_ridge_logistic(X, y, alpha=1e-4)
        strong, _ = fit_ridge_logistic(X, y, alpha=50.0)
        self.assertLess(np.abs(strong).sum(), np.abs(weak).sum())

    def test_matches_sklearn_when_available(self):
        """Independent oracle for the IRLS implementation."""
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:  # pragma: no cover - optional dev dependency
            self.skipTest("sklearn not installed")

        rng = np.random.default_rng(4)
        X = rng.normal(size=(400, 6))
        y = (1.0 / (1.0 + np.exp(-(X @ np.array([1.5, -1.0, 0.5, 0, 0.2, -0.4]))))
             > rng.random(400)).astype(float)

        alpha = 1.0
        mine_w, mine_b = fit_ridge_logistic(X, y, alpha=alpha)
        # sklearn parameterizes as C = 1 / (2 * alpha) for our loss scaling.
        ref = LogisticRegression(C=1.0 / (2.0 * alpha), max_iter=5000, tol=1e-10)
        ref.fit(X, y)
        np.testing.assert_allclose(mine_w, ref.coef_[0], atol=0.05)
        self.assertAlmostEqual(mine_b, float(ref.intercept_[0]), delta=0.05)

    def test_intercept_is_never_penalized(self):
        """Shrinking the intercept biases every prediction toward 0.5.

        Under a heavy penalty the weights collapse and the model becomes
        intercept-only, which must reproduce the base rate exactly. A
        penalized intercept pulls it toward zero instead -- a small, uniform,
        easy-to-miss bias on every activity prediction the system makes.
        """
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 4))
        y = 1.0 / (1.0 + np.exp(-(X @ np.array([1.5, -1.0, 0.0, 0.5]) + 2.0)))
        base_rate_logit = float(np.log(y.mean() / (1.0 - y.mean())))

        w, b = fit_ridge_logistic(X, y, alpha=1000.0)
        self.assertLess(np.abs(w).sum(), 0.2, "weights should be crushed")
        self.assertAlmostEqual(b, base_rate_logit, delta=0.01)

    def test_is_deterministic(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(100, 4))
        y = rng.random(100)
        a = fit_ridge_logistic(X, y, alpha=1.0)
        b = fit_ridge_logistic(X, y, alpha=1.0)
        np.testing.assert_array_equal(a[0], b[0])


class TestGroupedSplitting(unittest.TestCase):
    def test_a_group_never_straddles_a_fold(self):
        groups = [f"T{i // 10}" for i in range(120)]
        for train_idx, test_idx in group_kfold(groups, 4, seed=1):
            train_groups = {groups[i] for i in train_idx}
            test_groups = {groups[i] for i in test_idx}
            self.assertEqual(train_groups & test_groups, set())

    def test_every_row_is_tested_exactly_once(self):
        groups = [f"T{i // 10}" for i in range(120)]
        seen = []
        for _, test_idx in group_kfold(groups, 4, seed=1):
            seen.extend(test_idx)
        self.assertEqual(sorted(seen), list(range(120)))

    def test_is_deterministic_for_a_seed(self):
        groups = [f"T{i // 5}" for i in range(60)]
        self.assertEqual(group_kfold(groups, 3, seed=7), group_kfold(groups, 3, seed=7))

    def test_rejects_more_folds_than_groups(self):
        with self.assertRaises(ValueError):
            group_kfold(["A", "A", "B"], 5)

    def test_three_way_split_is_disjoint_by_group(self):
        groups = [f"T{i // 8}" for i in range(160)]
        train, cal, test = three_way_group_split(groups, seed=2)
        gt, gc, gs = ({groups[i] for i in part} for part in (train, cal, test))
        self.assertEqual(gt & gc, set())
        self.assertEqual(gt & gs, set())
        self.assertEqual(gc & gs, set())
        self.assertEqual(len(train) + len(cal) + len(test), 160)
        for part in (train, cal, test):
            self.assertTrue(part, "no split may be empty")


class TestSplittingBiasIsMeasuredNotAssumed(unittest.TestCase):
    """The ungrouped-split bias is regime-dependent; pin both regimes."""

    def test_leakage_appears_when_the_model_has_capacity_to_spare(self):
        examples, _ = simulate_screen(
            seed=3, step=1, signal_scale=10, transcript_effect_sd=1.0,
            noise_sd=0.2, n_transcripts=4, transcript_length=60,
        )
        # ~156 rows against 156 features: the overfitting regime.
        self.assertLess(len(examples), 250)
        comparison = compare_splitting(examples, k=4, seed=1)
        self.assertGreater(comparison.gap, 0.10)
        self.assertIn("inflated", comparison.interpretation)

    def test_the_bias_reverses_when_data_is_plentiful(self):
        examples, _ = simulate_screen(
            seed=3, step=1, signal_scale=10, transcript_effect_sd=1.0,
            noise_sd=0.2, n_transcripts=12, transcript_length=240,
        )
        self.assertGreater(len(examples), 2000)
        comparison = compare_splitting(examples, k=4, seed=1)
        self.assertLess(comparison.gap, 0.0)
        self.assertIn("composition", comparison.interpretation)

    def test_removing_the_transcript_effect_removes_the_gap(self):
        """Pins the mechanism: neither effect comes from sequence overlap alone."""
        examples, _ = simulate_screen(
            seed=7, step=1, signal_scale=10, transcript_effect_sd=0.0,
            noise_sd=0.2, n_transcripts=12, transcript_length=240,
        )
        self.assertAlmostEqual(compare_splitting(examples, k=5, seed=1).gap, 0.0, delta=0.02)


class TestFittedModel(unittest.TestCase):
    def setUp(self):
        self.examples, _ = simulate_screen(
            seed=11, step=2, signal_scale=10, transcript_effect_sd=0.8,
            noise_sd=0.2, n_transcripts=10, transcript_length=200,
        )

    def test_learns_real_signal_on_unseen_transcripts(self):
        report = cross_validate(self.examples, alpha=1.0, k=5, seed=1, grouped=True)
        self.assertGreater(report.spearman_mean, 0.5)

    def test_domain_is_read_off_the_training_data(self):
        model = fit_model(self.examples)
        self.assertIn(Effector.RFX_CAS13D, model.domain.effectors)
        self.assertEqual(model.domain.spacer_lengths, frozenset({22}))
        observed_max = max(e.duplex().n_mismatch for e in self.examples)
        self.assertEqual(model.domain.max_mismatches, observed_max)
        # Nothing wider than what it saw.
        self.assertFalse(model.domain.supports(Effector.PSP_CAS13B, 22, 0))
        self.assertFalse(model.domain.supports(Effector.RFX_CAS13D, 30, 0))

    def test_predictions_are_probabilities(self):
        model = fit_model(self.examples)
        for e in self.examples[:50]:
            p = model.predict(e.spacer, e.duplex(), e.accessibility)
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_rejects_mixed_spacer_lengths(self):
        mixed = list(self.examples[:5])
        odd = mixed[0]
        mixed.append(
            TrainingExample(
                spacer=odd.spacer[:-1],
                target_site=odd.target_site[:-1],
                observed_activity=0.5,
                group="X",
                effector=odd.effector,
                assay=odd.assay,
            )
        )
        with self.assertRaises(ValueError):
            fit_model(mixed)

    def test_top_features_are_named(self):
        model = fit_model(self.examples)
        top = model.top_features(5)
        self.assertEqual(len(top), 5)
        for name, weight in top:
            self.assertIn(name, model.spec.names)
            self.assertIsInstance(weight, float)


class TestTrainAndCalibrate(unittest.TestCase):
    def setUp(self):
        self.examples, _ = simulate_screen(
            seed=21, step=2, signal_scale=10, transcript_effect_sd=0.8,
            noise_sd=0.2, n_transcripts=12, transcript_length=220,
        )
        self.result = train_and_calibrate(self.examples, alpha=1.0, seed=5)

    def test_splits_are_grouped_and_disjoint(self):
        groups = self.result.split_groups
        self.assertEqual(set(groups["train"]) & set(groups["calibrate"]), set())
        self.assertEqual(set(groups["train"]) & set(groups["test"]), set())
        self.assertEqual(set(groups["calibrate"]) & set(groups["test"]), set())

    def test_conformal_coverage_holds_on_held_out_transcripts(self):
        """The end-to-end claim: calibrated bounds cover on unseen transcripts."""
        self.assertTrue(self.result.benchmark.coverage_ok, self.result.benchmark.summary())

    def test_model_has_skill_on_held_out_transcripts(self):
        self.assertGreater(self.result.benchmark.spearman, 0.4)

    def test_result_summarizes_the_whole_protocol(self):
        text = self.result.summary()
        for token in ("grouped", "spearman", "coverage", "train"):
            self.assertIn(token, text)


class TestFeatureSpec(unittest.TestCase):
    def test_block_offsets_match_the_layout(self):
        spec = FeatureSpec.for_length(22)
        blocks = spec.blocks
        self.assertEqual(spec.n_features, 156)
        self.assertEqual(blocks["onehot"], (0, 88))
        self.assertEqual(blocks["global"][1], spec.n_features)

    def test_one_hot_encodes_exactly_one_base_per_position(self):
        spec = FeatureSpec.for_length(22)
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        x = spec.transform(spacer, profile_duplex(spacer, spacer[::-1]), None)
        lo, hi = spec.blocks["onehot"]
        self.assertEqual(x[lo:hi].sum(), 22.0)

    def test_mismatch_block_marks_the_right_spacer_position(self):
        spec = FeatureSpec.for_length(22)
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        from cas13guide.util.seq import revcomp

        site = list(revcomp(spacer))
        site[0] = "C" if site[0] != "C" else "G"  # spacer position 22
        x = spec.transform(spacer, profile_duplex(spacer, "".join(site)), None)
        lo, _ = spec.blocks["mismatch"]
        self.assertEqual(x[lo + 21], 1.0)
        self.assertEqual(x[lo : lo + 21].sum(), 0.0)

    def test_rejects_a_spacer_of_the_wrong_length(self):
        spec = FeatureSpec.for_length(22)
        with self.assertRaises(ValueError):
            spec.transform("AUGC", profile_duplex("AUGC", "GCAU"), None)


class TestAnUnmeasuredCrossValidationSaysSo(unittest.TestCase):
    """Nought is not a score, and CVReport.summary knows the difference.

    A fold is skipped when its train or test side comes out empty, which
    happens for real: a three-group screen leaves the training split with a
    single group and every fold is skipped. `measured` is then False and the
    numbers below it mean nothing.

    The guard that stops them being printed was unpinned. Disabling it --
    `if self.measured is None`, which a bool never is -- left the whole suite
    green while `cas13guide train` reported "grouped 0-fold: spearman 0.000
    +/- 0.000" for a screen it could not evaluate at all. The comment above
    the branch says exactly what that costs: it reads as "no skill" rather
    than "not measured", and those are opposite conclusions about a model.
    """

    def _report(self, **kw):
        from cas13guide.services.training import CVReport

        base = dict(grouped=True, folds=0, spearman_mean=0.0, spearman_std=0.0,
                    mae_mean=0.0, requested_folds=5)
        base.update(kw)
        return CVReport(**base)

    def test_it_says_not_measured_and_shows_no_number(self):
        summary = self._report().summary()
        self.assertIn("NOT MEASURED", summary)
        self.assertNotIn("spearman", summary)

    def test_it_names_the_folds_asked_for_not_the_none_that_ran(self):
        """"0-fold" would be true and useless; 5 is what the user requested."""
        self.assertIn("5-fold", self._report().summary())

    def test_a_measured_report_still_reports_its_numbers(self):
        """Or the tests above would pass with the summary always refusing."""
        summary = self._report(folds=5, spearman_mean=0.42).summary()
        self.assertIn("spearman", summary)
        self.assertNotIn("NOT MEASURED", summary)


if __name__ == "__main__":
    unittest.main()
