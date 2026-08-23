"""Metric semantics: the aggregation rules that stop misses from hiding."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass, Task
from cas13guide.contract.spec import TargetSpec
from cas13guide.metrics.aggregate import (
    build_metric_vector,
    cumulative_burden,
    desired_utility,
    discrimination_margin,
    required_coverage,
    worst_off_target,
)


class TestRequiredCoverage(unittest.TestCase):
    def test_min_not_mean_so_a_miss_cannot_hide(self):
        """The whole point of min(): one strong target must not mask a miss."""
        spec = make_spec(
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="t2", sequence=SEQ_B, target_class=TargetClass.REQUIRED),
            )
        )
        evidence = (
            make_evidence(0.95, 0.99, target_id="t1", target_class=TargetClass.REQUIRED),
            make_evidence(0.05, 0.10, target_id="t2", target_class=TargetClass.REQUIRED),
        )
        mean_would_be = (0.95 + 0.05) / 2
        self.assertAlmostEqual(required_coverage(evidence, spec), 0.05)
        self.assertLess(required_coverage(evidence, spec), mean_would_be)

    def test_absent_required_target_scores_zero_not_missing(self):
        spec = make_spec(
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="t2", sequence=SEQ_B, target_class=TargetClass.REQUIRED),
            )
        )
        evidence = (make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED),)
        self.assertEqual(required_coverage(evidence, spec), 0.0)

    def test_best_site_wins_within_one_target(self):
        spec = make_spec()
        evidence = (
            make_evidence(0.2, 0.3, locus_id="L1", target_id="t1", target_class=TargetClass.REQUIRED),
            make_evidence(0.8, 0.9, locus_id="L2", target_id="t1", target_class=TargetClass.REQUIRED),
        )
        self.assertAlmostEqual(required_coverage(evidence, spec), 0.8)


class TestHazard(unittest.TestCase):
    def test_tolerated_earns_no_utility(self):
        spec = make_spec(
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="t3", sequence=SEQ_B, target_class=TargetClass.TOLERATED),
            )
        )
        evidence = (
            make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED),
            make_evidence(0.9, 0.95, target_id="t3", target_class=TargetClass.TOLERATED),
        )
        self.assertEqual(desired_utility(evidence, spec), 0.0)

    def test_cumulative_burden_catches_what_max_misses(self):
        """Twelve weak hits are invisible to max() and obvious to sum()."""
        weak = tuple(
            make_evidence(0.05, 0.12, locus_id=f"L{i}", expression=(0.4, 0.5), hazard_weight=0.6)
            for i in range(12)
        )
        self.assertLess(worst_off_target(weak), 0.05)
        self.assertGreater(cumulative_burden(weak), 0.35)

    def test_hazard_uses_upper_bounds_on_both_axes(self):
        e = make_evidence(0.1, 0.8, expression=(0.1, 0.9), hazard_weight=2.0)
        # 0.8 (UCB act) * 0.9 (UCB expr) * 2.0 -- not the point estimates.
        self.assertAlmostEqual(worst_off_target((e,)), 0.8 * 0.9 * 2.0)

    def test_intended_targets_are_not_off_targets(self):
        e = make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED)
        self.assertEqual(cumulative_burden((e,)), 0.0)


class TestDiscriminationMargin(unittest.TestCase):
    def test_none_for_non_allele_tasks(self):
        spec = make_spec(task=Task.KNOCKDOWN)
        self.assertIsNone(discrimination_margin((), spec))

    def test_ratio_succeeds_where_independent_thresholds_cannot(self):
        """The case fix #2 exists for.

        On-allele LCB 0.45 sits *below* the 0.50 coverage floor and the
        off-allele UCB 0.15 sits *below* the 0.20 forbidden ceiling, so two
        independent thresholds call this guide a failure. The ratio -- 3.0
        against tau 2.0 -- says the guide discriminates the alleles fine.
        """
        spec = make_spec(
            task=Task.ALLELE,
            targets=(
                TargetSpec(target_id="alt", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="wt", sequence=SEQ_B, target_class=TargetClass.FORBIDDEN),
            ),
        )
        evidence = (
            make_evidence(0.45, 0.60, target_id="alt", target_class=TargetClass.REQUIRED),
            make_evidence(0.02, 0.15, target_id="wt", target_class=TargetClass.FORBIDDEN),
        )
        margin = discrimination_margin(evidence, spec)
        self.assertAlmostEqual(margin, 3.0)
        self.assertGreaterEqual(margin, spec.thresholds.discrimination_margin_min)
        # And the independent threshold on coverage would indeed have failed it.
        self.assertLess(required_coverage(evidence, spec), spec.thresholds.required_coverage_min)

    def test_infinite_when_off_allele_has_no_activity(self):
        spec = make_spec(
            task=Task.ALLELE,
            targets=(
                TargetSpec(target_id="alt", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="wt", sequence=SEQ_B, target_class=TargetClass.FORBIDDEN),
            ),
        )
        evidence = (
            make_evidence(0.8, 0.9, target_id="alt", target_class=TargetClass.REQUIRED),
            make_evidence(0.0, 0.0, target_id="wt", target_class=TargetClass.FORBIDDEN),
        )
        self.assertEqual(discrimination_margin(evidence, spec), float("inf"))


class TestRobustness(unittest.TestCase):
    def test_hazard_across_contexts_is_max_not_mean(self):
        """'Hazard is read at its worst across contexts too -- never averaged.'"""
        spec = make_spec()
        required = make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED)
        quiet = (required, make_evidence(0.0, 0.0, locus_id="LX", expression=(0.0, 0.0)))
        loud = (
            required,
            make_evidence(0.8, 0.9, locus_id="LX", expression=(0.5, 0.6), hazard_weight=1.0),
        )
        mv = build_metric_vector({"reference": quiet, "hap2": loud}, clean_qc(), spec)
        loud_hazard = 0.9 * 0.6 * 1.0
        self.assertAlmostEqual(mv.worst_off_target, loud_hazard)
        self.assertAlmostEqual(mv.cumulative_burden, loud_hazard)
        self.assertGreater(mv.worst_off_target, loud_hazard / 2)  # a mean would halve it

    def test_discrimination_margin_across_contexts_is_min(self):
        spec = make_spec(
            task=Task.ALLELE,
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="wt", sequence=SEQ_B, target_class=TargetClass.FORBIDDEN),
            ),
        )
        good = (
            make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED),
            make_evidence(0.0, 0.1, target_id="wt", target_class=TargetClass.FORBIDDEN),
        )
        poor = (
            make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED),
            make_evidence(0.0, 0.9, target_id="wt", target_class=TargetClass.FORBIDDEN),
        )
        mv = build_metric_vector({"reference": good, "hap2": poor}, clean_qc(), spec)
        self.assertAlmostEqual(mv.discrimination_margin, 0.9 / 0.9)  # the worse context

    def test_robustness_is_worst_case_across_contexts(self):
        spec = make_spec()
        good = (make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED),)
        bad = (make_evidence(0.3, 0.4, target_id="t1", target_class=TargetClass.REQUIRED),)
        mv = build_metric_vector({"reference": good, "hap2": bad}, clean_qc(), spec)
        self.assertAlmostEqual(mv.required_coverage, 0.9)  # reference headline
        self.assertAlmostEqual(mv.robustness, 0.3)  # worst context
        self.assertLess(mv.robustness, mv.required_coverage)


if __name__ == "__main__":
    unittest.main()
