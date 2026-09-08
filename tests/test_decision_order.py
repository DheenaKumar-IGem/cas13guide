"""Gate -> Pareto -> preferences. The ordering is the design."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass, Verdict
from cas13guide.metrics.aggregate import build_metric_vector
from cas13guide.models.evidence import GuideAssessment, MetricVector
from cas13guide.pipeline.f_decide import (
    DEFAULT_WEIGHTS,
    RankedGuide,
    distinct_site_ids,
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
            # Every weighted objective this guide actually *has* must appear in
            # the sensitivity report -- scoped to the guide's own objective set,
            # because discrimination_margin is emitted only for allele tasks and
            # this is a knockdown one. (Asserting against all of DEFAULT_WEIGHTS
            # would forbid any task-conditional objective from ever having a
            # default weight.)
            present = set(r.assessment.metrics.objectives())
            self.assertEqual(set(r.sensitivity), present & set(DEFAULT_WEIGHTS))
            self.assertNotIn("discrimination_margin", present)

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



class TestAlleleTasksAreRankedByTheirGoverningMetric(unittest.TestCase):
    """``discrimination_margin`` must actually influence the order.

    ``objectives()`` emits it for every allele task, but ``DEFAULT_WEIGHTS``
    omitted it and ``weights.get(k, 0.0)`` silently gave it weight zero. Two
    guides identical except for a thousand-fold difference in on/off-allele
    ratio scored the same, and the order fell back to input order via the
    stable sort -- while the rationale line still printed the objective, which
    reads as though it had been considered.
    """

    def _pair(self):
        from cas13guide.contract.enums import Task
        from cas13guide.contract.spec import TargetSpec
        from cas13guide.models.evidence import (
            Candidate, GateResult, GuideAssessment, MetricVector,
        )
        from cas13guide.contract.enums import ReasonCode

        def one(guide_id, margin):
            return GuideAssessment(
                guide_id=guide_id,
                candidate=Candidate(
                    guide_id=guide_id, spacer="GCAUUGCUGUGGAGCUGAAGUC",
                    source_target_id="t1", start=0,
                ),
                qc=clean_qc(guide_id),
                metrics=MetricVector(
                    required_coverage=0.9, desired_utility=0.0,
                    worst_off_target=0.0, cumulative_burden=0.0,
                    robustness=0.9, manufacturability=1.0,
                    discrimination_margin=margin,
                ),
                gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
            )

        # identical on every other objective; only the margin differs
        return [one("barely", 2.01), one("excellent", 1_000_000.0)]

    def test_the_better_margin_ranks_first(self):
        ranked = rank_by_preference(self._pair())
        self.assertEqual(ranked[0].assessment.guide_id, "excellent")

    def test_it_carries_a_non_zero_weight_in_the_sensitivity_report(self):
        ranked = rank_by_preference(self._pair())
        best = next(r for r in ranked if r.assessment.guide_id == "excellent")
        self.assertIn("discrimination_margin", best.sensitivity)
        self.assertGreater(best.sensitivity["discrimination_margin"], 0.0)

    def test_reversing_the_input_does_not_change_the_winner(self):
        forward = rank_by_preference(self._pair())
        backward = rank_by_preference(list(reversed(self._pair())))
        self.assertEqual(
            forward[0].assessment.guide_id, backward[0].assessment.guide_id
        )

    def test_a_knockdown_task_is_unaffected(self):
        """The weight is inert for every task that does not emit the objective."""
        spec = make_spec()
        guides = [
            assessment(
                f"g{i}",
                (make_evidence(0.7 + i * 0.05, 0.8 + i * 0.05,
                               target_id="t1", target_class=TargetClass.REQUIRED),),
                spec,
            )
            for i in range(3)
        ]
        for r in rank_by_preference(guides):
            self.assertNotIn("discrimination_margin", r.sensitivity)



class TestDistinctSites(unittest.TestCase):
    """A ranked list is not a panel.

    Activity varies smoothly along a transcript, so the top entries are usually
    the same window shifted by a base or two. Measured on a 400 nt target with a
    28 nt spacer, the top ten ranked guides covered **two** distinct sites and
    ranks 1-3 differed by one nucleotide each. Ordering that buys one experiment
    ten times while looking like ten independent candidates.
    """

    SPACER = "GCAUUGCUGUGGAGCUGAAGUC"

    def _ranked(self, starts, target_ids=None):
        """A ranked list with the given start offsets, in rank order."""
        from cas13guide.contract.enums import ReasonCode
        from cas13guide.models.evidence import (
            Candidate, GateResult, GuideAssessment, MetricVector,
        )

        target_ids = target_ids or ["t1"] * len(starts)
        out = []
        for index, (start, target_id) in enumerate(zip(starts, target_ids)):
            guide_id = f"g{index}"
            assessment = GuideAssessment(
                guide_id=guide_id,
                candidate=Candidate(
                    guide_id=guide_id, spacer=self.SPACER,
                    source_target_id=target_id, start=start,
                ),
                qc=clean_qc(guide_id),
                metrics=MetricVector(0.9, 0.0, 0.0, 0.0, 0.9, 1.0),
                gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
            )
            out.append(
                RankedGuide(assessment=assessment, rank=index + 1,
                            score=10.0 - index, rationale=(), sensitivity={})
            )
        return out

    def test_shifted_copies_of_one_site_collapse_to_the_first(self):
        ranked = self._ranked([100, 101, 102, 103])
        distinct = distinct_site_ids(ranked, min_spacing=22)
        self.assertEqual(distinct, frozenset({"g0"}))

    def test_genuinely_separate_sites_all_survive(self):
        ranked = self._ranked([0, 50, 100, 150])
        self.assertEqual(len(distinct_site_ids(ranked, min_spacing=22)), 4)

    def test_the_best_ranked_representative_wins(self):
        """Rank order decides which of an overlapping cluster is kept."""
        ranked = self._ranked([300, 301, 10])
        distinct = distinct_site_ids(ranked, min_spacing=22)
        self.assertIn("g0", distinct)     # rank 1, start 300
        self.assertNotIn("g1", distinct)  # rank 2, start 301 -- overlaps g0
        self.assertIn("g2", distinct)     # rank 3, start 10 -- separate site

    def test_exactly_spacer_length_apart_is_distinct(self):
        """Adjacent but not overlapping: no shared target base."""
        ranked = self._ranked([0, 22])
        self.assertEqual(len(distinct_site_ids(ranked, min_spacing=22)), 2)
        # one closer and they overlap
        ranked = self._ranked([0, 21])
        self.assertEqual(len(distinct_site_ids(ranked, min_spacing=22)), 1)

    def test_positions_on_different_targets_are_not_compared(self):
        """Start 100 on one transcript says nothing about start 100 on another."""
        ranked = self._ranked([100, 100], target_ids=["t1", "t2"])
        self.assertEqual(len(distinct_site_ids(ranked, min_spacing=22)), 2)

    def test_an_empty_ranking_is_handled(self):
        self.assertEqual(distinct_site_ids([], min_spacing=22), frozenset())

    def test_it_marks_rather_than_filters(self):
        """Nothing is dropped and no rank changes -- the caller decides.

        Renumbering would hide that guides were collapsed; a reader seeing
        #1, #2, #15 can tell the list was condensed rather than sparse.
        """
        ranked = self._ranked([100, 101, 200])
        distinct = distinct_site_ids(ranked, min_spacing=22)
        self.assertEqual(len(ranked), 3)                       # nothing removed
        self.assertEqual([r.rank for r in ranked], [1, 2, 3])  # nothing renumbered
        self.assertEqual(len(distinct), 2)


class TestDistinctSitesReachTheReport(unittest.TestCase):
    """The flag is only worth having if it arrives where results are read."""

    def _report(self):
        import random

        from cas13guide.contract.enums import Universe
        from cas13guide.pipeline.orchestrator import DesignContext, run_design
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.indexes import ExhaustiveScanIndex
        from cas13guide.services.registry import (
            HeuristicCas13Model, ModelRegistry, NullAccessibilityModel,
        )
        from cas13guide.services.transcriptome import Locus, Transcriptome

        spec = make_spec()
        rng = random.Random(5)
        calibration = CalibrationService()
        for regime in ("mm0", "mm1-2", "mm3+"):
            key = CalibrationService.stratum_key(
                spec.profile.effector.value, spec.profile.assay.value, regime
            )
            for i in range(150):
                truth = rng.random()
                calibration.add_observation(
                    key, min(1.0, max(0.0, truth + rng.gauss(0, 0.1))), truth,
                    group=f"T{i % 12:02d}",
                )
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("L_T1", SEQ_A, Universe.CODING, gene="TGT", expression=100.0))
        context = DesignContext(
            "reference", world,
            ExhaustiveScanIndex(world, spec.context.declared_universes),
            {"L_T1": "t1"},
        )
        return run_design(
            spec, [context],
            ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel()),
            calibration, panel_size=2,
        )

    def test_every_ranked_row_carries_the_flag(self):
        data = self._report().to_dict()
        self.assertTrue(data["ranked"])
        for row in data["ranked"]:
            self.assertIn("distinct_site", row)
            self.assertIsInstance(row["distinct_site"], bool)

    def test_the_summary_count_matches_the_flagged_rows(self):
        data = self._report().to_dict()
        flagged = sum(1 for row in data["ranked"] if row["distinct_site"])
        self.assertEqual(data["distinct_site_count"], flagged)

    def test_a_densely_tiled_target_really_does_collapse(self):
        """The condition that motivated this: step-1 tiling on one transcript."""
        data = self._report().to_dict()
        self.assertGreater(len(data["ranked"]), 10)
        self.assertLess(data["distinct_site_count"], len(data["ranked"]))

    def test_the_first_ranked_guide_is_always_a_distinct_site(self):
        data = self._report().to_dict()
        self.assertTrue(data["ranked"][0]["distinct_site"])


class TestTheSensitivityReportIsWhatItSaysAndReachesSomebody(unittest.TestCase):
    """It was computed for every guide and exported nowhere.

    The one question a weighted ranking raises is whether a guide is first
    because of the evidence or because of how the reader weighted it. This is
    the number that answers it, and until now answering meant parsing the
    rationale prose.

    Its own comment also described it as ``d(score)/d(weight)``, which is a
    different quantity: the stored value is ``weight x normalised value``, and
    the derivative in the weight is the normalised value alone. The tests below
    pin the semantics the field actually has, so the description and the code
    cannot drift apart again.
    """

    _ranked_cache: list = []

    @classmethod
    def _report(cls):
        """One design, shared. It was one per test method."""
        if cls._ranked_cache:
            return cls._ranked_cache

        import random
        import tempfile
        from pathlib import Path

        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(9)
        sequence = "".join(rng.choice("ACGU") for _ in range(300))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.fa"
            path.write_text(">t1" + chr(10) + sequence + chr(10), encoding="utf-8")
            cls._ranked_cache = run_request(
                DesignRequest(fasta_path=str(path), task="knockdown",
                              assay="cell_culture_knockdown",
                              effector="RfxCas13d", use_structure=False,
                              use_signal=False)
            ).report["ranked"]
        return cls._ranked_cache

    def test_every_ranked_row_carries_it(self):
        rows = self._report()
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(guide=row["guide_id"]):
                self.assertIn("sensitivity", row)
                self.assertTrue(row["sensitivity"])

    def test_the_contributions_add_up_to_the_score(self):
        """A breakdown that does not sum to the thing it breaks down is worse
        than no breakdown."""
        for row in self._report():
            with self.subTest(guide=row["guide_id"]):
                self.assertAlmostEqual(
                    sum(row["sensitivity"].values()), row["score"], places=3
                )

    def test_it_names_the_same_objectives_the_rationale_does(self):
        """Two renderings of one calculation; they must not disagree."""
        for row in self._report():
            with self.subTest(guide=row["guide_id"]):
                from_prose = {line.split("=", 1)[0] for line in row["rationale"]}
                self.assertEqual(set(row["sensitivity"]), from_prose)

    def test_it_is_a_contribution_and_not_a_derivative(self):
        """Doubling a weight doubles that objective's contribution, while the
        derivative in that weight would not move at all. That is how the
        description and the value can be told apart.

        Driven through ``rank_by_preference`` because no shipped door lets a
        user set weights -- ``run_design`` takes them and neither the CLI nor
        ``DesignRequest`` passes them on.
        """
        pool = [
            _assessment_with_coverage(f"g{i}", coverage)
            for i, coverage in enumerate((0.2, 0.5, 0.9))
        ]
        single = rank_by_preference(pool, {"required_coverage": 3.0})
        double = rank_by_preference(pool, {"required_coverage": 6.0})

        before = {r.assessment.guide_id: r.sensitivity for r in single}
        after = {r.assessment.guide_id: r.sensitivity for r in double}
        moved = 0
        for guide_id, contributions in before.items():
            baseline = contributions.get("required_coverage", 0.0)
            if not baseline:
                continue
            moved += 1
            self.assertAlmostEqual(
                after[guide_id]["required_coverage"], baseline * 2.0, places=6
            )
        self.assertGreater(moved, 0, "no guide had a coverage contribution to move")


def _assessment_with_coverage(guide_id: str, coverage: float) -> GuideAssessment:
    from cas13guide.contract.enums import ReasonCode
    from cas13guide.models.evidence import Candidate, GateResult

    return GuideAssessment(
        guide_id=guide_id,
        candidate=Candidate(guide_id=guide_id, spacer="A" * 22,
                            source_target_id="t1", start=0),
        qc=clean_qc(guide_id),
        metrics=MetricVector(coverage, 0.0, 0.0, 0.0, coverage, 1.0),
        gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
    )


if __name__ == "__main__":
    unittest.main()
