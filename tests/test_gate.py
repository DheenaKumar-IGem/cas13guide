"""The gate: every FAIL condition must actually bite, and REVIEW must grade."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import (
    QCFlag,
    ReasonCode,
    ReviewTier,
    TargetClass,
    Task,
    Universe,
    Verdict,
)
from cas13guide.contract.spec import TargetSpec
from cas13guide.metrics.aggregate import build_metric_vector
from cas13guide.models.evidence import QCResult
from cas13guide.pipeline.f_decide import gate

GOOD_EVIDENCE = None


def strong_required(**kw):
    return make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED, **kw)


def run_gate(spec, evidence, qc=None, bound=None, gaps=frozenset()):
    qc = qc or clean_qc()
    bound = bound or exhaustive_bound()
    metrics = build_metric_vector({"reference": evidence}, qc, spec)
    return metrics, gate(metrics, qc, evidence, bound, gaps, spec)


class TestPass(unittest.TestCase):
    def test_clean_design_passes(self):
        spec = make_spec()
        _, result = run_gate(spec, (strong_required(),))
        self.assertIs(result.verdict, Verdict.PASS)
        self.assertIn(ReasonCode.CONSTRAINTS_SATISFIED, result.reasons)
        self.assertIsNone(result.review_tier)


class TestFailConditions(unittest.TestCase):
    def test_qc_failure_is_deterministic_incompatibility(self):
        spec = make_spec()
        qc = QCResult(guide_id="g1", passed=False, flags=(QCFlag.POLY_U_TERMINATOR,))
        _, result = run_gate(spec, (strong_required(),), qc=qc)
        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIn(ReasonCode.DETERMINISTIC_INCOMPATIBLE, result.reasons)

    def test_required_target_missed(self):
        spec = make_spec()
        weak = make_evidence(0.10, 0.20, target_id="t1", target_class=TargetClass.REQUIRED)
        _, result = run_gate(spec, (weak,))
        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIn(ReasonCode.REQUIRED_TARGET_MISSED, result.reasons)

    def test_credible_forbidden_hit(self):
        spec = make_spec(
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="tf", sequence=SEQ_B, target_class=TargetClass.FORBIDDEN),
            )
        )
        evidence = (
            strong_required(),
            make_evidence(
                0.3, 0.7, locus_id="LF", target_id="tf",
                target_class=TargetClass.FORBIDDEN, expression=(0.0, 0.0),
            ),
        )
        _, result = run_gate(spec, evidence)
        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIn(ReasonCode.CREDIBLE_FORBIDDEN_HIT, result.reasons)

    def test_universe_inadequacy_fails_even_with_a_clean_report(self):
        """Fix #3: an unsearched universe yields a *clean* off-target report.

        The guide below looks perfect -- strong on-target, zero off-targets --
        precisely because rRNA and mito were never searched. That must FAIL,
        not pass quietly.
        """
        spec = make_spec(universes=frozenset({Universe.CODING}))
        gaps = spec.context.missing_universes(spec.profile.assay)
        self.assertTrue(gaps)
        metrics, result = run_gate(spec, (strong_required(),), gaps=gaps)
        self.assertEqual(metrics.worst_off_target, 0.0)  # looks spotless
        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIn(ReasonCode.UNIVERSE_INADEQUATE, result.reasons)

    def test_discrimination_margin_gate(self):
        spec = make_spec(
            task=Task.ALLELE,
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="wt", sequence=SEQ_B, target_class=TargetClass.FORBIDDEN),
            ),
            forbidden_activity_max=0.9,  # isolate the margin from the forbidden gate
        )
        evidence = (
            strong_required(),
            make_evidence(
                0.5, 0.8, locus_id="LW", target_id="wt",
                target_class=TargetClass.FORBIDDEN, expression=(0.0, 0.0),
            ),
        )
        metrics, result = run_gate(spec, evidence)
        self.assertLess(metrics.discrimination_margin, spec.thresholds.discrimination_margin_min)
        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIn(ReasonCode.DISCRIMINATION_MARGIN_INSUFFICIENT, result.reasons)

    def test_hazard_budget_exceeded(self):
        spec = make_spec()
        evidence = (
            strong_required(),
            make_evidence(0.9, 0.99, locus_id="LX", expression=(0.9, 0.99), hazard_weight=3.0),
        )
        _, result = run_gate(spec, evidence)
        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIn(ReasonCode.HAZARD_BUDGET_EXCEEDED, result.reasons)


class TestReviewGrading(unittest.TestCase):
    def test_r3_model_out_of_domain(self):
        spec = make_spec()
        evidence = (
            strong_required(),
            make_evidence(0.0, 0.05, locus_id="LO", ood=True, domain_ok=False),
        )
        _, result = run_gate(spec, evidence)
        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIs(result.review_tier, ReviewTier.R3)
        self.assertIn(ReasonCode.MODEL_OUT_OF_DOMAIN, result.reasons)
        self.assertNotIn(ReasonCode.CALIBRATION_UNSUPPORTED, result.reasons)

    def test_r3_calibration_gap_is_reported_distinctly(self):
        """A missing calibration stratum is not a model-domain violation.

        Both widen the interval, but they call for different remedies -- more
        calibration data versus a model that covers the case at all -- so the
        gate must not report one as the other.
        """
        spec = make_spec()
        evidence = (
            strong_required(),
            make_evidence(0.0, 0.05, locus_id="LO", ood=True, domain_ok=True),
        )
        _, result = run_gate(spec, evidence)
        self.assertIs(result.review_tier, ReviewTier.R3)
        self.assertIn(ReasonCode.CALIBRATION_UNSUPPORTED, result.reasons)
        self.assertNotIn(ReasonCode.MODEL_OUT_OF_DOMAIN, result.reasons)

    def test_r2_search_not_exhaustive(self):
        spec = make_spec()
        _, result = run_gate(
            spec, (strong_required(),), bound=exhaustive_bound(exhaustive=False)
        )
        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIs(result.review_tier, ReviewTier.R2)
        self.assertIn(ReasonCode.SEARCH_NOT_EXHAUSTIVE, result.reasons)

    def test_r2_search_envelope_too_shallow(self):
        spec = make_spec()
        _, result = run_gate(
            spec, (strong_required(),), bound=exhaustive_bound(max_mismatches=2)
        )
        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIn(ReasonCode.SEARCH_NOT_EXHAUSTIVE, result.reasons)

    def test_r2_unknown_expression(self):
        spec = make_spec()
        evidence = (
            strong_required(),
            make_evidence(
                0.0, 0.05, locus_id="LU", expression=(0.0, 0.9), expression_imputed=True
            ),
        )
        _, result = run_gate(spec, evidence)
        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIs(result.review_tier, ReviewTier.R2)
        self.assertIn(ReasonCode.EXPRESSION_UNKNOWN, result.reasons)

    def test_r1_borderline_coverage(self):
        spec = make_spec()
        borderline = make_evidence(
            0.55, 0.65, target_id="t1", target_class=TargetClass.REQUIRED
        )
        _, result = run_gate(spec, (borderline,))
        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIs(result.review_tier, ReviewTier.R1)
        self.assertIn(ReasonCode.BORDERLINE_COVERAGE, result.reasons)

    def test_worst_tier_wins_when_several_apply(self):
        spec = make_spec()
        evidence = (
            make_evidence(0.55, 0.65, target_id="t1", target_class=TargetClass.REQUIRED),  # R1
            make_evidence(0.0, 0.05, locus_id="LO", ood=True, domain_ok=False),  # R3
        )
        _, result = run_gate(spec, evidence)
        self.assertIs(result.review_tier, ReviewTier.R3)
        self.assertIn(ReasonCode.BORDERLINE_COVERAGE, result.reasons)
        self.assertIn(ReasonCode.MODEL_OUT_OF_DOMAIN, result.reasons)

    def test_fail_outranks_every_review_signal(self):
        """A FAIL condition is never downgraded by co-occurring REVIEW flags."""
        spec = make_spec(universes=frozenset({Universe.CODING}))
        gaps = spec.context.missing_universes(spec.profile.assay)
        evidence = (
            strong_required(),
            make_evidence(0.0, 0.05, locus_id="LO", ood=True, domain_ok=False),
        )
        _, result = run_gate(spec, evidence, gaps=gaps)
        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIsNone(result.review_tier)


if __name__ == "__main__":
    unittest.main()
