"""FR6.3 -- Bayesian panel confidence.

Two claims are under test, and they fail in opposite directions, so both
directions are pinned:

* the *floor* must not be inflated by adding members. A panel is worth its
  best member unless independence is shown, and the previous number (noisy-OR)
  rose with every guide added regardless of whether it added anything;
* the *point estimate* must not exist without a measured dependence
  parameter. A confident-looking number whose confidence is a guess is the
  failure this whole package is arranged against.
"""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import ReasonCode, TargetClass, Verdict
from cas13guide.contract.spec import TargetSpec
from cas13guide.metrics.bounds import Interval
from cas13guide.models.evidence import (
    Candidate,
    GateResult,
    GuideAssessment,
    MetricVector,
    PairEvidence,
)
from cas13guide.pipeline.optimizer import Panel
from cas13guide.services.calibration import CalibrationService
from cas13guide.services.panel_confidence import (
    BayesianPanelConfidence,
    PanelDependence,
    UnmeasuredDependenceError,
    beta_quantile,
    regularized_incomplete_beta,
)

SPACER = "GCAUUGCUGUGGAGCUGAAGUC"


def guide(guide_id, coverage: dict, start: int = 0):
    evidence = []
    for target_id, lcb in coverage.items():
        hit = dataclasses.replace(
            make_hit(guide_id, f"L_{target_id}", target_id, TargetClass.REQUIRED),
            start=start,
        )
        evidence.append(
            PairEvidence(
                guide_id=guide_id,
                locus_id=f"L_{target_id}",
                context="reference",
                hit=hit,
                activity=Interval(lcb, lcb, min(1.0, lcb + 0.05)),
                expression=Interval(0.5, 0.5, 0.5),
                hazard_weight=0.0,
            )
        )
    worst = min(coverage.values())
    return GuideAssessment(
        guide_id=guide_id,
        candidate=Candidate(guide_id=guide_id, spacer=SPACER, source_target_id="t1", start=start),
        qc=clean_qc(guide_id),
        metrics=MetricVector(worst, 0.0, 0.05, 0.05, worst, 1.0),
        gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
        evidence=tuple(evidence),
    )


def panel_of(*assessments):
    return Panel(guides=tuple(assessments), coverage={}, objective=0.0, guarantee="test")


def calibration_for(spec, *, rows=400, success_rate=0.8):
    """A stratum where a stated fraction of predictions cleared threshold.

    Deterministic rather than random: the posterior mean is then a number the
    test can state in full, which is the point of testing an inference.
    """
    service = CalibrationService()
    key = CalibrationService.stratum_key(
        spec.profile.effector.value, spec.profile.assay.value, "mm0"
    )
    for i in range(rows):
        predicted = 0.90
        # Observed clears the 0.50 gate threshold for exactly `success_rate` of
        # the rows; which rows is irrelevant and fixed by the counter.
        observed = 0.95 if i < round(rows * success_rate) else 0.10
        service.add_observation(key, predicted, observed, group=f"T{i % 12:02d}")
    return service


ONE_TARGET = (
    TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
)


class TestBetaFunctions(unittest.TestCase):
    """The posterior is only as good as the special functions under it."""

    def test_cdf_matches_closed_form_for_integer_parameters(self):
        # I_x(2, 3) = 1 - (1-x)^3 (1 + 3x) -- integrate the density directly.
        for x in (0.05, 0.25, 0.5, 0.75, 0.99):
            with self.subTest(x=x):
                expected = 1.0 - (1 - x) ** 3 * (1 + 3 * x)
                self.assertAlmostEqual(
                    regularized_incomplete_beta(2, 3, x), expected, places=10
                )

    def test_cdf_matches_the_uniform_case(self):
        for x in (0.0, 0.1, 0.5, 1.0):
            self.assertAlmostEqual(regularized_incomplete_beta(1, 1, x), x, places=12)

    def test_quantile_inverts_the_cdf(self):
        for a, b in ((0.5, 0.5), (3.0, 7.0), (120.5, 30.5)):
            for p in (0.05, 0.5, 0.95):
                with self.subTest(a=a, b=b, p=p):
                    x = beta_quantile(a, b, p)
                    self.assertAlmostEqual(
                        regularized_incomplete_beta(a, b, x), p, places=8
                    )

    def test_the_posterior_concentrates_as_data_arrives(self):
        """A credible interval that does not narrow is not doing inference."""
        widths = []
        for n in (10, 100, 1000):
            a, b = 0.5 + 0.8 * n, 0.5 + 0.2 * n
            widths.append(beta_quantile(a, b, 0.95) - beta_quantile(a, b, 0.05))
        self.assertTrue(widths[0] > widths[1] > widths[2], widths)


class TestGuideSuccessPosterior(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec(targets=ONE_TARGET)
        self.model = BayesianPanelConfidence(
            calibration_for(self.spec, rows=400, success_rate=0.75), self.spec
        )

    def test_it_counts_calibration_rows_that_cleared_the_threshold(self):
        success = self.model.guide_success("g1", "t1", 0.90)
        self.assertEqual(success.trials, 400)
        self.assertEqual(success.successes, 300)
        # Jeffreys prior: (0.5 + 300) / (1 + 400).
        self.assertAlmostEqual(success.mean, 300.5 / 401.0, places=12)

    def test_the_posterior_is_not_the_activity_bound(self):
        """The whole point: a 0.90 bound does not mean a 0.90 success rate.

        Reading the bound as a probability is the mistake FR6.3 exists to
        correct, so a model that merely echoes it back has done nothing.
        """
        success = self.model.guide_success("g1", "t1", 0.90)
        self.assertLess(success.mean, 0.85)

    def test_a_credible_interval_is_reported_and_brackets_the_mean(self):
        success = self.model.guide_success("g1", "t1", 0.90)
        self.assertLess(success.credible_lower, success.mean)
        self.assertGreater(success.credible_upper, success.mean)

    def test_no_calibration_leaves_the_prior_and_says_so(self):
        model = BayesianPanelConfidence(CalibrationService(), self.spec)
        success = model.guide_success("g1", "t1", 0.90)
        self.assertEqual(success.trials, 0)
        self.assertAlmostEqual(success.mean, 0.5)
        self.assertTrue(success.thin)
        self.assertTrue(success.stratum.endswith("none"))

    def test_a_prediction_outside_every_window_falls_back_unconditionally(self):
        """Weaker evidence, not absent evidence -- and it must be labelled."""
        success = self.model.guide_success("g1", "t1", 0.10)
        self.assertEqual(success.trials, 400)
        self.assertTrue(success.thin)
        self.assertIn("unconditional", success.stratum)


class TestPanelInterval(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec(targets=ONE_TARGET)
        self.calibration = calibration_for(self.spec, rows=400, success_rate=0.75)
        self.model = BayesianPanelConfidence(self.calibration, self.spec)

    def _confidence(self, *assessments):
        result = self.model.for_panel(panel_of(*assessments))
        return result.by_target()["t1"]

    def test_a_single_guide_has_floor_equal_to_ceiling(self):
        """Nothing is being combined, so there is no dependence question."""
        target = self._confidence(guide("gA", {"t1": 0.90}))
        self.assertAlmostEqual(target.lower, target.upper, places=12)

    def test_the_floor_is_the_best_member_not_the_noisy_or(self):
        target = self._confidence(
            guide("gA", {"t1": 0.90}, start=0),
            guide("gB", {"t1": 0.90}, start=100),
            guide("gC", {"t1": 0.90}, start=200),
        )
        best = max(m.mean for m in target.members)
        self.assertAlmostEqual(target.lower, best, places=12)
        self.assertGreater(target.upper, target.lower)

    def test_adding_members_cannot_raise_the_floor_on_its_own(self):
        """The defect in the number this replaces.

        Noisy-OR credits a third correlated copy of the same evidence as
        though it were a third independent experiment. The floor must not.
        """
        one = self._confidence(guide("gA", {"t1": 0.90}, start=0))
        three = self._confidence(
            guide("gA", {"t1": 0.90}, start=0),
            guide("gB", {"t1": 0.90}, start=100),
            guide("gC", {"t1": 0.90}, start=200),
        )
        self.assertAlmostEqual(one.lower, three.lower, places=12)
        self.assertGreater(three.upper, one.upper)

    def test_the_ceiling_is_exactly_the_noisy_or_of_the_posteriors(self):
        target = self._confidence(
            guide("gA", {"t1": 0.90}, start=0),
            guide("gB", {"t1": 0.90}, start=100),
        )
        miss = 1.0
        for member in target.members:
            miss *= 1.0 - member.mean
        self.assertAlmostEqual(target.upper, 1.0 - miss, places=12)

    def test_the_credible_envelope_contains_the_posterior_mean_interval(self):
        target = self._confidence(
            guide("gA", {"t1": 0.90}, start=0),
            guide("gB", {"t1": 0.90}, start=100),
        )
        self.assertLessEqual(target.credible_lower, target.lower)
        self.assertGreaterEqual(target.credible_upper, target.upper)

    def test_a_target_no_member_reaches_gets_no_entry(self):
        """An absence, reported as an absence rather than as a computed 0.0."""
        spec = make_spec(
            targets=ONE_TARGET
            + (
                TargetSpec(
                    target_id="t2", sequence=SEQ_B, target_class=TargetClass.REQUIRED
                ),
            )
        )
        model = BayesianPanelConfidence(calibration_for(spec), spec)
        result = model.for_panel(panel_of(guide("gA", {"t1": 0.90})))
        self.assertEqual([t.target_id for t in result.targets], ["t1"])

    def test_it_states_its_assumptions_where_the_number_is_read(self):
        result = self.model.for_panel(panel_of(guide("gA", {"t1": 0.90})))
        text = " ".join(result.notes).lower()
        self.assertIn("interval", text)
        self.assertIn("independently", text)


class TestDependenceIsRefusedUntilMeasured(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec(targets=ONE_TARGET)
        self.calibration = calibration_for(self.spec)
        self.panel = panel_of(
            guide("gA", {"t1": 0.90}, start=0),
            guide("gB", {"t1": 0.90}, start=100),
        )

    def test_no_point_estimate_without_a_dependence_parameter(self):
        result = BayesianPanelConfidence(self.calibration, self.spec).for_panel(self.panel)
        self.assertFalse(result.has_point_estimates)
        self.assertIsNone(result.dependence_version)

    def test_absolute_confidence_refuses_when_none_was_supplied(self):
        model = BayesianPanelConfidence(self.calibration, self.spec)
        with self.assertRaises(UnmeasuredDependenceError) as caught:
            model.absolute_confidence(self.panel)
        self.assertIn("--panel-dependence", str(caught.exception))

    def test_absolute_confidence_refuses_a_provisional_value(self):
        """A guessed beta is refused exactly as an absent one is.

        Otherwise the flag is decoration: anyone wanting the number writes
        provisional=true and gets it anyway.
        """
        model = BayesianPanelConfidence(
            self.calibration, self.spec,
            dependence=PanelDependence(0.2, source="guess", provisional=True),
        )
        with self.assertRaises(UnmeasuredDependenceError):
            model.absolute_confidence(self.panel)
        self.assertFalse(model.for_panel(self.panel).has_point_estimates)

    def test_a_measured_value_unlocks_the_point_estimate(self):
        model = BayesianPanelConfidence(
            self.calibration, self.spec,
            dependence=PanelDependence(0.2, source="replicate panels, 2026-08", provisional=False),
        )
        values = model.absolute_confidence(self.panel)
        self.assertIn("t1", values)
        target = model.for_panel(self.panel).by_target()["t1"]
        self.assertIsNotNone(target.point)
        self.assertGreaterEqual(target.point, target.lower)
        self.assertLessEqual(target.point, target.upper)

    def test_beta_zero_reproduces_the_independent_ceiling(self):
        """The dependence model has to contain the old number as a special case,
        or it is a different model wearing its name."""
        model = BayesianPanelConfidence(
            self.calibration, self.spec,
            dependence=PanelDependence(0.0, source="measured", provisional=False),
        )
        target = model.for_panel(self.panel).by_target()["t1"]
        self.assertAlmostEqual(target.point, target.upper, places=9)

    def test_beta_one_collapses_onto_the_floor(self):
        model = BayesianPanelConfidence(
            self.calibration, self.spec,
            dependence=PanelDependence(1.0, source="measured", provisional=False),
        )
        target = model.for_panel(self.panel).by_target()["t1"]
        self.assertAlmostEqual(target.point, target.lower, places=9)

    def test_more_shared_failure_never_makes_a_panel_look_better(self):
        points = []
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            model = BayesianPanelConfidence(
                self.calibration, self.spec,
                dependence=PanelDependence(beta, source="measured", provisional=False),
            )
            points.append(model.for_panel(self.panel).by_target()["t1"].point)
        self.assertEqual(points, sorted(points, reverse=True), points)


class TestPanelDependenceIO(unittest.TestCase):
    def test_round_trip_through_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beta.json"
            original = PanelDependence(0.3, source="replicate panels", provisional=False)
            original.to_json(path)
            self.assertEqual(PanelDependence.from_json(path), original)

    def test_a_fraction_outside_zero_to_one_is_refused(self):
        for value in (-0.1, 1.4):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PanelDependence(value)

    def test_an_unknown_key_is_named_rather_than_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beta.json"
            path.write_text(json.dumps({"common_cause_beta": 0.2, "bta": 1}), "utf-8")
            with self.assertRaises(ValueError) as caught:
                PanelDependence.from_json(path)
            self.assertIn("bta", str(caught.exception))

    def test_the_version_says_whether_the_value_was_measured(self):
        self.assertIn("provisional", PanelDependence(0.2).version)
        self.assertIn(
            "measured",
            PanelDependence(0.2, source="s", provisional=False).version,
        )


class TestRunCarriesTheConfidence(unittest.TestCase):
    """It has to reach the places a decision is made from, not just exist."""

    def test_limitations_say_the_dependence_was_never_measured(self):
        from cas13guide.services.limitations import limitations_for_run

        spec = make_spec(targets=ONE_TARGET)
        confidence = BayesianPanelConfidence(
            calibration_for(spec), spec
        ).for_panel(panel_of(guide("gA", {"t1": 0.9}), guide("gB", {"t1": 0.9}, start=100)))
        ids = {
            item.id
            for item in limitations_for_run(spec, panel_size=2, panel_confidence=confidence)
        }
        self.assertIn("panel-dependence-unmeasured", ids)

    def test_that_warning_is_absent_once_a_value_is_measured(self):
        """A warning that fires unconditionally is indistinguishable from a
        broken one."""
        from cas13guide.services.limitations import limitations_for_run

        spec = make_spec(targets=ONE_TARGET)
        confidence = BayesianPanelConfidence(
            calibration_for(spec), spec,
            dependence=PanelDependence(0.2, source="measured", provisional=False),
        ).for_panel(panel_of(guide("gA", {"t1": 0.9}), guide("gB", {"t1": 0.9}, start=100)))
        ids = {
            item.id
            for item in limitations_for_run(spec, panel_size=2, panel_confidence=confidence)
        }
        self.assertNotIn("panel-dependence-unmeasured", ids)

    def test_an_empty_panel_raises_neither_warning(self):
        from cas13guide.services.limitations import limitations_for_run

        spec = make_spec(targets=ONE_TARGET)
        confidence = BayesianPanelConfidence(calibration_for(spec), spec).for_panel(panel_of())
        ids = {
            item.id
            for item in limitations_for_run(spec, panel_size=0, panel_confidence=confidence)
        }
        self.assertNotIn("panel-dependence-unmeasured", ids)
        self.assertNotIn("panel-confidence-thin-calibration", ids)

    def test_the_report_dictionary_carries_it_under_the_panel(self):
        from cas13guide.pipeline.g_output import DesignReport
        from cas13guide.pipeline.g_output import MetricSummary

        spec = make_spec(targets=ONE_TARGET)
        panel = panel_of(guide("gA", {"t1": 0.9}))
        confidence = BayesianPanelConfidence(calibration_for(spec), spec).for_panel(panel)
        report = DesignReport(
            spec_fingerprint="x",
            verdict_counts={},
            review_rate=0.0,
            ranked=(),
            distinct_sites=frozenset(),
            panel=panel,
            controls=(),
            coverage_map={},
            metric_summary=MetricSummary(
                n_assessed=0, coverage_min=0.0, coverage_median=0.0,
                coverage_max=0.0, distinct_coverage_values=0,
                worst_off_target_max=0.0,
            ),
            off_target_rows=(),
            provenance={},
            validation_plan=(),
            panel_confidence=confidence,
        )
        block = report.to_dict()["panel"]["confidence"]
        self.assertEqual(block["targets"][0]["target_id"], "t1")
        self.assertIsNone(block["dependence"])
        self.assertTrue(block["notes"])

    def test_a_report_without_the_model_still_serialises(self):
        """Optional means optional -- an older caller must not break."""
        from cas13guide.pipeline.g_output import DesignReport
        from cas13guide.pipeline.g_output import MetricSummary

        report = DesignReport(
            spec_fingerprint="x", verdict_counts={}, review_rate=0.0, ranked=(),
            distinct_sites=frozenset(), panel=panel_of(), controls=(),
            coverage_map={},
            metric_summary=MetricSummary(
                n_assessed=0, coverage_min=0.0, coverage_median=0.0,
                coverage_max=0.0, distinct_coverage_values=0,
                worst_off_target_max=0.0,
            ),
            off_target_rows=(), provenance={}, validation_plan=(),
        )
        self.assertIsNone(report.to_dict()["panel"]["confidence"])


if __name__ == "__main__":
    unittest.main()
