"""Gate -> Pareto -> preferences. The ordering is the design."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass, Verdict
from cas13guide.metrics.aggregate import build_metric_vector
from cas13guide.models.evidence import GuideAssessment, MetricVector
from cas13guide.pipeline.f_decide import (
    DEFAULT_WEIGHTS,
    dominates,
    gate,
    pareto_front,
    rank_by_preference,
    review_rate,
)


def assessment(guide_id, evidence, spec, qc=None):
    qc = qc or clean_qc(guide_id)
    metrics = build_metric_vector({"reference": evidence}, qc, spec)
    result = gate(metrics, qc, evidence, exhaustive_bound(), frozenset(), spec)
    return GuideAssessment(
        guide_id=guide_id,
        candidate=make_candidate(guide_id),
        qc=qc,
        metrics=metrics,
        gate=result,
        evidence=evidence,
    )


class TestPareto(unittest.TestCase):
    def test_dominated_guide_is_excluded(self):
        spec = make_spec()
        strong = assessment(
            "g_strong",
            (make_evidence(0.95, 0.98, target_id="t1", target_class=TargetClass.REQUIRED),),
            spec,
        )
        weak = assessment(
            "g_weak",
            (
                make_evidence(0.60, 0.70, target_id="t1", target_class=TargetClass.REQUIRED),
                make_evidence(0.4, 0.5, locus_id="LX", expression=(0.3, 0.4), hazard_weight=0.6),
            ),
            spec,
        )
        front = pareto_front([strong, weak])
        self.assertIn(strong, front)
        self.assertNotIn(weak, front)

    def test_trade_offs_are_preserved_not_averaged(self):
        """Two guides, each better on a different axis -- both survive."""
        a = MetricVector(0.9, 1.0, 0.8, 2.0, 0.9, 0.9)  # better coverage, worse hazard
        b = MetricVector(0.6, 1.0, 0.1, 0.2, 0.6, 0.9)  # worse coverage, better hazard
        self.assertFalse(dominates(a, b))
        self.assertFalse(dominates(b, a))


class TestPreferencesCannotRescueAFail(unittest.TestCase):
    def test_ranking_only_sees_gate_survivors(self):
        spec = make_spec()
        failing = assessment(
            "g_fail",
            (make_evidence(0.05, 0.10, target_id="t1", target_class=TargetClass.REQUIRED),),
            spec,
        )
        self.assertIs(failing.gate.verdict, Verdict.FAIL)

        passing = assessment(
            "g_pass",
            (make_evidence(0.95, 0.98, target_id="t1", target_class=TargetClass.REQUIRED),),
            spec,
        )
        # Even a weighting that maximally favours the failing guide's profile
        # cannot surface it: the gate runs first and it never reaches ranking.
        survivors = [a for a in (failing, passing) if a.gate.verdict is Verdict.PASS]
        ranked = rank_by_preference(survivors, {"required_coverage": 0.0, "worst_off_target": 99.0})
        self.assertEqual([r.assessment.guide_id for r in ranked], ["g_pass"])

    def test_ranking_reports_rationale_and_sensitivity(self):
        spec = make_spec()
        guides = [
            assessment(
                f"g{i}",
                (
                    make_evidence(
                        0.7 + i * 0.05, 0.8 + i * 0.05,
                        target_id="t1", target_class=TargetClass.REQUIRED,
                    ),
                ),
                spec,
            )
            for i in range(3)
        ]
        ranked = rank_by_preference(guides)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0].rank, 1)
        self.assertGreaterEqual(ranked[0].score, ranked[-1].score)
        for r in ranked:
            self.assertTrue(r.rationale)
            self.assertEqual(set(r.sensitivity) & set(DEFAULT_WEIGHTS), set(DEFAULT_WEIGHTS))

    def test_user_weights_change_the_order(self):
        spec = make_spec()
        coverage_led = assessment(
            "g_cov",
            (
                make_evidence(0.95, 0.98, target_id="t1", target_class=TargetClass.REQUIRED),
                make_evidence(0.3, 0.4, locus_id="LX", expression=(0.3, 0.4), hazard_weight=0.5),
            ),
            spec,
        )
        clean = assessment(
            "g_clean",
            (make_evidence(0.75, 0.85, target_id="t1", target_class=TargetClass.REQUIRED),),
            spec,
        )
        pool = [coverage_led, clean]
        by_coverage = rank_by_preference(pool, {"required_coverage": 10.0, "worst_off_target": 0.0})
        by_safety = rank_by_preference(pool, {"required_coverage": 0.0, "worst_off_target": 10.0})
        self.assertEqual(by_coverage[0].assessment.guide_id, "g_cov")
        self.assertEqual(by_safety[0].assessment.guide_id, "g_clean")


class TestInfiniteObjectives(unittest.TestCase):
    """An ideal guide has an infinite discrimination margin. It must rank first."""

    def _allele_pool(self):
        from cas13guide.contract.enums import Task
        from cas13guide.contract.spec import TargetSpec

        spec = make_spec(
            task=Task.ALLELE,
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="wt", sequence=SEQ_B, target_class=TargetClass.FORBIDDEN),
            ),
            forbidden_activity_max=0.95,
        )
        best = assessment(  # no credible off-allele hit at all -> margin = inf
            "BEST",
            (make_evidence(0.99, 1.0, target_id="t1", target_class=TargetClass.REQUIRED),),
            spec,
        )
        mid = assessment(
            "MID",
            (
                make_evidence(0.80, 0.85, target_id="t1", target_class=TargetClass.REQUIRED),
                make_evidence(
                    0.0, 0.10, locus_id="LW", target_id="wt",
                    target_class=TargetClass.FORBIDDEN, expression=(0.0, 0.0),
                ),
            ),
            spec,
        )
        low = assessment(
            "LOW",
            (
                make_evidence(0.60, 0.70, target_id="t1", target_class=TargetClass.REQUIRED),
                make_evidence(
                    0.0, 0.19, locus_id="LW", target_id="wt",
                    target_class=TargetClass.FORBIDDEN, expression=(0.0, 0.0),
                ),
            ),
            spec,
        )
        return [best, mid, low]

    def test_infinite_margin_ranks_first_not_last(self):
        pool = self._allele_pool()
        self.assertEqual(pool[0].metrics.discrimination_margin, float("inf"))
        ranked = rank_by_preference(pool)
        self.assertEqual(ranked[0].assessment.guide_id, "BEST")

    def test_ranking_is_independent_of_input_order(self):
        """A NaN score makes the sort order depend on how the list arrived."""
        import itertools

        orders = []
        for perm in itertools.permutations(self._allele_pool()):
            ranked = rank_by_preference(list(perm))
            orders.append(tuple(r.assessment.guide_id for r in ranked))
        self.assertEqual(len(set(orders)), 1, f"unstable ranking: {set(orders)}")

    def test_no_score_is_nan(self):
        import math

        for r in rank_by_preference(self._allele_pool()):
            self.assertFalse(math.isnan(r.score))
            for value in r.sensitivity.values():
                self.assertFalse(math.isnan(value))


class TestGateIsEnforcedByTheRanker(unittest.TestCase):
    def test_ranker_drops_failed_guides_even_if_the_caller_does_not(self):
        """Defence in depth for the README's headline invariant.

        Filtering in the caller is not enough -- the ranker itself must refuse
        a FAIL, or one careless call site puts an unsafe guide on the list.
        """
        spec = make_spec()
        failing = assessment(
            "g_fail",
            (make_evidence(0.05, 0.10, target_id="t1", target_class=TargetClass.REQUIRED),),
            spec,
        )
        passing = assessment(
            "g_pass",
            (make_evidence(0.95, 0.98, target_id="t1", target_class=TargetClass.REQUIRED),),
            spec,
        )
        self.assertIs(failing.gate.verdict, Verdict.FAIL)
        ranked = rank_by_preference([failing, passing])  # unfiltered on purpose
        self.assertEqual([r.assessment.guide_id for r in ranked], ["g_pass"])

    def test_ranking_an_all_failed_pool_returns_nothing(self):
        spec = make_spec()
        failing = assessment(
            "g_fail",
            (make_evidence(0.05, 0.10, target_id="t1", target_class=TargetClass.REQUIRED),),
            spec,
        )
        self.assertEqual(rank_by_preference([failing]), [])


class TestReviewRate(unittest.TestCase):
    def test_review_rate_is_measurable(self):
        spec = make_spec()
        passing = assessment(
            "p", (make_evidence(0.95, 0.98, target_id="t1", target_class=TargetClass.REQUIRED),), spec
        )
        reviewing = assessment(
            "r",
            (
                make_evidence(0.95, 0.98, target_id="t1", target_class=TargetClass.REQUIRED),
                make_evidence(0.0, 0.05, locus_id="LO", ood=True, domain_ok=False),
            ),
            spec,
        )
        self.assertIs(reviewing.gate.verdict, Verdict.REVIEW)
        self.assertAlmostEqual(review_rate([passing, reviewing]), 0.5)
        self.assertEqual(review_rate([]), 0.0)


if __name__ == "__main__":
    unittest.main()
