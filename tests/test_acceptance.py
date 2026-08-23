"""The gate a real screen has to clear before its model is trusted."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.services.acceptance import evaluate_dataset
from cas13guide.simulate import simulate_screen


def screen(**kw):
    # 40 transcripts so the calibration split gets >= 8 independent groups;
    # with fewer, conformal coverage is unreliable and the gate says so.
    #
    # step=4 rather than 8: at step=8 the calibration split leaves the mm3+
    # regime with 19 rows, one short of MIN_STRATUM_N, so that stratum resolves
    # to nothing and its test pairs are scored against the [0, 1] fallback --
    # which "covers" all of them and scored a vacuous 1.000. A good screen has
    # to be good in every regime it will be evaluated in, so the fixture is
    # dense enough to calibrate all three (mm0/mm1-2/mm3+ get 265/93/42 rows).
    base = dict(
        seed=21, step=4, signal_scale=10, transcript_effect_sd=0.8,
        noise_sd=0.2, n_transcripts=40, transcript_length=220,
    )
    base.update(kw)
    return simulate_screen(**base)[0]


class TestAcceptsAGoodScreen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The only test that pays for the split-bias probe.
        cls.report = evaluate_dataset(screen(), seed=5)

    def test_accepted(self):
        self.assertTrue(self.report.accepted, self.report.summary())

    def test_every_criterion_is_reported_with_its_measurement(self):
        names = {c.name for c in self.report.criteria}
        self.assertEqual(
            names,
            {
                "rows",
                "transcript groups",
                "calibration transcripts",
                "grouped CV spearman",
                "held-out spearman",
                "conformal coverage",
            },
        )
        for criterion in self.report.criteria:
            self.assertIsInstance(criterion.measured, float)
            self.assertIn(criterion.name, criterion.line())

    def test_carries_the_trained_result_through(self):
        self.assertIsNotNone(self.report.result)
        self.assertTrue(self.report.result.model.version)

    def test_reports_the_split_bias_regime(self):
        self.assertTrue(self.report.split_bias)


class TestRejectsBadScreens(unittest.TestCase):
    def test_rejects_too_few_rows(self):
        report = evaluate_dataset(
            screen(n_transcripts=8, transcript_length=60, step=6), probe_split_bias=False
        )
        self.assertFalse(report.accepted)
        self.assertFalse(next(c for c in report.criteria if c.name == "rows").passed)

    def test_rejects_too_few_groups_without_training(self):
        examples = screen()
        keep = sorted({e.group for e in examples})[:3]
        small = [e for e in examples if e.group in keep]
        report = evaluate_dataset(small, min_rows=1, probe_split_bias=False)
        self.assertFalse(report.accepted)
        self.assertFalse(
            next(c for c in report.criteria if c.name == "transcript groups").passed
        )
        # Cheap checks run first: no model is fitted for a dataset this thin.
        self.assertIsNone(report.result)

    def test_rejects_a_screen_with_no_learnable_signal(self):
        """Pure noise must fail the skill criteria, not squeak through."""
        import random

        rng = random.Random(3)
        noise = [
            type(e)(
                spacer=e.spacer, target_site=e.target_site,
                observed_activity=rng.random(), group=e.group,
                effector=e.effector, assay=e.assay,
            )
            for e in screen(n_transcripts=20, step=10)
        ]
        report = evaluate_dataset(noise, seed=5, probe_split_bias=False)
        self.assertFalse(report.accepted)
        skill = next(c for c in report.criteria if c.name == "grouped CV spearman")
        self.assertFalse(skill.passed)

    def test_summary_names_the_failing_criterion(self):
        report = evaluate_dataset(
            screen(n_transcripts=8, transcript_length=60, step=6), probe_split_bias=False
        )
        text = report.summary()
        self.assertIn("REJECTED", text)
        self.assertIn("[FAIL]", text)


class TestCoverageIsMeasuredNotAssumed(unittest.TestCase):
    """Grouped calibration breaks conformal's exchangeability assumption.

    Calibration rows come from one set of transcripts and test rows from
    another, so the marginal guarantee is approximate: empirical coverage runs
    slightly under nominal on average, and is *highly variable* when few
    independent groups back it. Measuring it on held-out groups -- and failing
    the gate when it breaks -- is the protection that actually holds.
    """

    def test_calibration_group_count_is_a_reported_criterion(self):
        report = evaluate_dataset(screen(), seed=5, probe_split_bias=False)
        criterion = next(
            c for c in report.criteria if c.name == "calibration transcripts"
        )
        self.assertGreaterEqual(criterion.measured, 8.0)
        self.assertIn("unreliable", criterion.note)

    def test_too_few_calibration_transcripts_is_flagged(self):
        report = evaluate_dataset(
            screen(n_transcripts=12, step=6), seed=5, min_rows=1, probe_split_bias=False
        )
        criterion = next(
            c for c in report.criteria if c.name == "calibration transcripts"
        )
        self.assertLess(criterion.measured, 8.0)
        self.assertFalse(criterion.passed)
        self.assertFalse(report.accepted)

    def test_a_thin_calibration_stratum_is_flagged_rather_than_certified(self):
        """The bound is still returned -- it is just not presented as certified."""
        from cas13guide.services.calibration import CalibrationService

        cs = CalibrationService(min_stratum_n=10, min_stratum_groups=8)
        key = CalibrationService.stratum_key("RfxCas13d", "a", "mm0")
        for i in range(200):
            cs.add_observation(key, 0.5, 0.5 + (i % 7) * 0.01, group=f"T{i % 3}")
        interval = cs.interval(0.5, key, alpha=0.10)
        self.assertTrue(interval.ood)
        self.assertIn("only-3-groups", interval.stratum)
        self.assertLess(interval.width, 1.0)  # usable, not maximal ignorance


if __name__ == "__main__":
    unittest.main()
