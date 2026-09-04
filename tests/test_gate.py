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
from cas13guide.contract.enums import AssayType
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

    def test_the_targets_own_unknown_expression_does_not_downgrade(self):
        """The complement of the test above, and the one that was missing.

        `EXPRESSION_UNKNOWN` used to count every locus with no abundance
        evidence, the run's own target included. An intended locus's
        expression reaches no metric, no threshold and no verdict -- its two
        readers, `pair_hazard` and the off-target explorer rows, both sit
        behind `not e.is_intended` -- so the downgrade fired for a gap that
        changes no number the run produces, on every guide of every run that
        did not pass an undocumented `--expression`.

        Measured on the README's own command shape: PASS 0 / REVIEW 272 with
        this as the sole reason, no panel, no controls, no FR6.3 interval, and
        a 100% review rate -- the degeneracy this module's docstring says the
        R1/R2/R3 grading exists to prevent. `--expression 0`, asserting the
        target is not expressed at all, cleared it just as well as 120.
        """
        spec = make_spec()
        _, result = run_gate(spec, (strong_required(expression_imputed=True),))
        self.assertNotIn(ReasonCode.EXPRESSION_UNKNOWN, result.reasons)
        self.assertIs(result.verdict, Verdict.PASS)

    def test_an_off_target_still_downgrades_beside_an_unknown_target(self):
        """And the reason still fires when it means something.

        Both loci lack abundance evidence here; only the off-target one is
        supposed to count.
        """
        spec = make_spec()
        evidence = (
            strong_required(expression_imputed=True),
            make_evidence(0.0, 0.05, locus_id="LU", expression=(0.0, 0.9),
                          expression_imputed=True),
        )
        _, result = run_gate(spec, evidence)
        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIn(ReasonCode.EXPRESSION_UNKNOWN, result.reasons)
        self.assertIn("1 locus/loci", " ".join(result.detail))

    def test_a_guide_that_loses_a_supplied_variant_is_not_certified(self):
        """FR7.2's flag has to reach the verdict, not just the metric vector.

        README and services/conservation.py both say FR7.2 -- "flagging a
        guide that loses activity against one variant" -- *is* the robustness
        metric. It was a number and nothing else: the gate never read it, so a
        guide with robustness 0.000, reaching no site at all in a variant the
        user supplied, was certified PASS with "all constraints satisfied with
        adequate evidence", sat on the Pareto frontier and was selected into
        the panel. A PASS is the opposite of a flag.

        R1, the least severe REVIEW: the guide meets every requirement on the
        primary target, and what needs a human is whether that variant matters
        to them.
        """
        spec = make_spec()
        reference = (strong_required(),)
        variant = (make_evidence(0.0, 0.05, target_id="t1",
                                 target_class=TargetClass.REQUIRED),)
        metrics = build_metric_vector(
            {"reference": reference, "variant": variant}, clean_qc(), spec)
        self.assertGreaterEqual(metrics.required_coverage,
                                spec.thresholds.required_coverage_min)
        self.assertLess(metrics.robustness, spec.thresholds.required_coverage_min)

        result = gate(metrics, clean_qc(), reference, exhaustive_bound(),
                      frozenset(), spec)
        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIn(ReasonCode.VARIANT_COVERAGE_LOST, result.reasons)

    def test_a_guide_that_holds_across_contexts_is_still_certified(self):
        """Or the rule above would just refuse everything with --variants."""
        spec = make_spec()
        reference = (strong_required(),)
        metrics = build_metric_vector(
            {"reference": reference, "variant": reference}, clean_qc(), spec)
        result = gate(metrics, clean_qc(), reference, exhaustive_bound(),
                      frozenset(), spec)
        self.assertNotIn(ReasonCode.VARIANT_COVERAGE_LOST, result.reasons)
        self.assertIs(result.verdict, Verdict.PASS)

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



class TestUniverseAdequacyReadsTheSearchNotTheDeclaration(unittest.TestCase):
    """A gap is what the index failed to cover, not what the spec forgot to ask for.

    The check used to be ``ASSAY_REQUIRED - declared_universes``, which never
    looks at the search at all. An index built over a narrower universe set than
    the spec declares therefore produced a perfectly clean off-target report and
    the gate saw no gap: in a reproduction the identical spec and transcriptome
    went from 42 FAIL to 42 PASS with zero off-target rows purely by narrowing
    the index, with rRNA sites at 5000 TPM simply ceasing to exist.
    """

    def _bound_over(self, universes):
        from cas13guide.services.indexes import ExhaustivenessBound

        return ExhaustivenessBound(
            max_mismatches=4,
            wobble_aware=True,
            universes=frozenset(universes),
            exhaustive=True,
            method="test",
        )

    def test_no_gap_when_the_index_covered_everything_declared(self):
        from cas13guide.pipeline.c_search import universe_gaps

        spec = make_spec(universes=FULL_UNIVERSES)
        self.assertEqual(universe_gaps(spec, self._bound_over(FULL_UNIVERSES)), frozenset())

    def test_a_narrower_index_is_a_gap_even_though_the_spec_declared_more(self):
        from cas13guide.pipeline.c_search import universe_gaps

        spec = make_spec(universes=FULL_UNIVERSES)
        gaps = universe_gaps(spec, self._bound_over({Universe.CODING}))
        self.assertIn(Universe.RRNA, gaps)
        self.assertIn(Universe.MITO, gaps)
        self.assertNotIn(Universe.CODING, gaps)

    def test_that_gap_fails_the_gate(self):
        spec = make_spec(universes=FULL_UNIVERSES)
        metrics = build_metric_vector({"reference": ()}, clean_qc(), spec)
        gaps = None
        from cas13guide.pipeline.c_search import universe_gaps

        gaps = universe_gaps(spec, self._bound_over({Universe.CODING}))
        result = gate(metrics, clean_qc(), (), exhaustive_bound(4), gaps, spec)
        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIn(ReasonCode.UNIVERSE_INADEQUATE, result.reasons)

    def test_declaring_a_universe_and_not_searching_it_is_also_a_gap(self):
        """Under-searching what you promised is the same hole as under-declaring."""
        from cas13guide.pipeline.c_search import universe_gaps

        spec = make_spec(
            universes=FULL_UNIVERSES | {Universe.REPEAT},
            assay=AssayType.CELL_CULTURE_KNOCKDOWN,
        )
        # REPEAT is not required by this assay, but it *was* declared.
        gaps = universe_gaps(spec, self._bound_over(FULL_UNIVERSES))
        self.assertEqual(gaps, frozenset({Universe.REPEAT}))


class TestBothHazardBudgetsAreReported(unittest.TestCase):
    """One reason code, but the detail must name every budget that blew.

    The two hazard checks were joined by an ``elif``, so a guide over both
    budgets reported only the worst-single-hit overrun. An operator who fixes
    that hit re-runs straight into a second, previously invisible FAIL -- and
    the detail tuple is what the module docstring calls the machine-readable
    reason surface.
    """

    def test_detail_names_both_when_both_are_exceeded(self):
        from cas13guide.models.evidence import MetricVector

        spec = make_spec()
        metrics = MetricVector(
            required_coverage=0.9,
            desired_utility=1.0,
            worst_off_target=1.5,      # cap 1.0
            cumulative_burden=40.0,    # cap 5.0
            robustness=0.9,
            manufacturability=0.9,
        )
        result = gate(metrics, clean_qc(), (), exhaustive_bound(4), frozenset(), spec)
        self.assertIs(result.verdict, Verdict.FAIL)
        joined = " | ".join(result.detail)
        self.assertIn("worst_off_target", joined)
        self.assertIn("cumulative_burden", joined)
        # Still exactly one reason code -- the verdict did not change.
        self.assertEqual(
            [r for r in result.reasons if r is ReasonCode.HAZARD_BUDGET_EXCEEDED],
            [ReasonCode.HAZARD_BUDGET_EXCEEDED],
        )

    def test_each_budget_alone_still_reports_only_itself(self):
        from cas13guide.models.evidence import MetricVector

        spec = make_spec()
        only_worst = gate(
            MetricVector(0.9, 1.0, 1.5, 0.0, 0.9, 0.9),
            clean_qc(), (), exhaustive_bound(4), frozenset(), spec,
        )
        only_burden = gate(
            MetricVector(0.9, 1.0, 0.0, 40.0, 0.9, 0.9),
            clean_qc(), (), exhaustive_bound(4), frozenset(), spec,
        )
        self.assertNotIn("cumulative_burden", " ".join(only_worst.detail))
        self.assertNotIn("worst_off_target", " ".join(only_burden.detail))




class TestCoverageFailureNamesItsCause(unittest.TestCase):
    """Missing evidence and demonstrated absence both zero the LCB.

    They produce the same verdict -- correctly, since neither can be certified
    -- but they are opposite findings. Reporting an unvalidated geometry as
    "required target missed" tells the user their guide is bad when in fact
    nothing has been measured, and points them at redesigning instead of at
    validating.
    """

    def _metrics_with_zero_coverage(self, evidence):
        spec = make_spec()
        return build_metric_vector({"reference": evidence}, clean_qc(), spec), spec

    def test_out_of_domain_evidence_is_named_as_such(self):
        evidence = (
            make_evidence(0.0, 1.0, target_id="t1", target_class=TargetClass.REQUIRED,
                          domain_ok=False, ood=True),
        )
        metrics, spec = self._metrics_with_zero_coverage(evidence)
        result = gate(metrics, clean_qc(), evidence, exhaustive_bound(4), frozenset(), spec)

        self.assertIs(result.verdict, Verdict.FAIL)     # still refuses to certify
        self.assertIn(ReasonCode.REQUIRED_TARGET_MISSED, result.reasons)
        self.assertIn(ReasonCode.MODEL_OUT_OF_DOMAIN, result.reasons)
        self.assertIn("missing validation", " ".join(result.detail))

    def test_uncalibrated_evidence_is_named_as_such(self):
        evidence = (
            make_evidence(0.0, 1.0, target_id="t1", target_class=TargetClass.REQUIRED,
                          domain_ok=True, ood=True),
        )
        metrics, spec = self._metrics_with_zero_coverage(evidence)
        result = gate(metrics, clean_qc(), evidence, exhaustive_bound(4), frozenset(), spec)

        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIn(ReasonCode.CALIBRATION_UNSUPPORTED, result.reasons)
        self.assertIn("calibration", " ".join(result.detail))

    def test_a_genuine_miss_is_not_relabelled_as_a_domain_problem(self):
        """In-domain, calibrated, and simply inactive: that IS a missed target."""
        evidence = (
            make_evidence(0.02, 0.08, target_id="t1", target_class=TargetClass.REQUIRED,
                          domain_ok=True, ood=False),
        )
        metrics, spec = self._metrics_with_zero_coverage(evidence)
        result = gate(metrics, clean_qc(), evidence, exhaustive_bound(4), frozenset(), spec)

        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertIn(ReasonCode.REQUIRED_TARGET_MISSED, result.reasons)
        self.assertNotIn(ReasonCode.MODEL_OUT_OF_DOMAIN, result.reasons)
        self.assertNotIn(ReasonCode.CALIBRATION_UNSUPPORTED, result.reasons)


if __name__ == "__main__":
    unittest.main()
