"""Panel design: submodular greedy under spacing and diversity constraints."""

import dataclasses
import unittest

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
from cas13guide.pipeline.optimizer import select_panel

SPACER = "GCAUUGCUGUGGAGCUGAAGUC"


def guide(guide_id, coverage: dict, start: int = 0, off_loci=()):
    """A passing assessment with the given per-required-target LCB coverage."""
    evidence: list[PairEvidence] = []

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

    for locus in off_loci:
        evidence.append(
            PairEvidence(
                guide_id=guide_id,
                locus_id=locus,
                context="reference",
                hit=make_hit(guide_id, locus),
                activity=Interval(0.05, 0.0, 0.1),
                expression=Interval(0.2, 0.1, 0.3),
                hazard_weight=0.5,
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


TWO_TARGETS = (
    TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
    TargetSpec(target_id="t2", sequence=SEQ_B, target_class=TargetClass.REQUIRED),
)


class TestPanelSelection(unittest.TestCase):
    def test_greedy_picks_complementary_guides(self):
        spec = make_spec(targets=TWO_TARGETS)
        pool = [
            guide("gA", {"t1": 0.9, "t2": 0.0}, start=0),
            guide("gB", {"t1": 0.0, "t2": 0.9}, start=100),
            guide("gC", {"t1": 0.85, "t2": 0.0}, start=200),
        ]
        panel = select_panel(pool, spec, size=2)
        self.assertEqual({g.guide_id for g in panel.guides}, {"gA", "gB"})
        self.assertGreater(panel.coverage["t2"], 0.5)

    def test_coverage_is_monotone_in_panel_size(self):
        spec = make_spec(targets=TWO_TARGETS)
        pool = [
            guide("gA", {"t1": 0.7, "t2": 0.0}, start=0),
            guide("gB", {"t1": 0.0, "t2": 0.7}, start=100),
            guide("gC", {"t1": 0.5, "t2": 0.2}, start=200),
        ]
        objectives = [select_panel(pool, spec, size=k).objective for k in (1, 2, 3)]
        self.assertLessEqual(objectives[0], objectives[1])
        self.assertLessEqual(objectives[1], objectives[2])

    def test_diminishing_returns(self):
        """Submodularity in practice: each added guide helps less than the last."""
        spec = make_spec(targets=TWO_TARGETS)
        pool = [
            guide("gA", {"t1": 0.8, "t2": 0.8}, start=0),
            guide("gB", {"t1": 0.8, "t2": 0.8}, start=100),
            guide("gC", {"t1": 0.8, "t2": 0.8}, start=200),
        ]
        objectives = [select_panel(pool, spec, size=k).objective for k in (1, 2, 3)]
        self.assertGreater(objectives[1] - objectives[0], objectives[2] - objectives[1])

    def test_spacing_constraint_rejects_shifted_copies(self):
        """A panel must not be three overlapping windows of one site."""
        spec = make_spec(targets=TWO_TARGETS)
        pool = [
            guide("gA", {"t1": 0.9, "t2": 0.9}, start=10),
            guide("gB", {"t1": 0.9, "t2": 0.9}, start=12),
            guide("gC", {"t1": 0.9, "t2": 0.9}, start=14),
        ]
        panel = select_panel(pool, spec, size=3, min_spacing=15)
        self.assertEqual(panel.size, 1)
        self.assertTrue(panel.dropped)  # and it reports the truncation

    def test_diversity_constraint_rejects_shared_off_target_profiles(self):
        spec = make_spec(targets=TWO_TARGETS)
        pool = [
            guide("gA", {"t1": 0.9, "t2": 0.0}, start=0, off_loci=("OFF1",)),
            guide("gB", {"t1": 0.0, "t2": 0.9}, start=100, off_loci=("OFF1",)),
        ]
        self.assertEqual(select_panel(pool, spec, size=2, max_shared_off=0).size, 1)
        self.assertEqual(select_panel(pool, spec, size=2, max_shared_off=5).size, 2)

    def test_coverage_reads_the_lower_bound_not_the_upper(self):
        """Panel coverage is a benefit claim, so it uses LCB.

        Reading UCB here would let a wide, uncertain interval masquerade as
        strong coverage -- the exact failure the bound algebra exists to stop.
        """
        spec = make_spec(targets=TWO_TARGETS[:1])
        wide = GuideAssessment(
            guide_id="gWide",
            candidate=Candidate(guide_id="gWide", spacer=SPACER, source_target_id="t1", start=0),
            qc=clean_qc("gWide"),
            metrics=MetricVector(0.1, 0.0, 0.0, 0.0, 0.1, 1.0),
            gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
            evidence=(
                PairEvidence(
                    guide_id="gWide",
                    locus_id="L_t1",
                    context="reference",
                    hit=make_hit("gWide", "L_t1", "t1", TargetClass.REQUIRED),
                    activity=Interval(0.55, 0.10, 0.99),  # wide and uncertain
                    expression=Interval(0.5, 0.5, 0.5),
                    hazard_weight=0.0,
                ),
            ),
        )
        panel = select_panel([wide], spec, size=1)
        self.assertAlmostEqual(panel.coverage["t1"], 0.10)

    def test_guarantee_is_reported_honestly(self):
        """The constrained branch claims no bound, because it has none.

        This used to assert the string contained "1/2". It checked the text and
        never the property, and the property is false -- see
        test_panel_guarantee.py, which brute-forces the optimum and exhibits an
        instance reaching 0.404 of it.
        """
        spec = make_spec(targets=TWO_TARGETS)
        pool = [guide("gA", {"t1": 0.9, "t2": 0.9}, start=0)]
        constrained = select_panel(pool, spec, size=1, min_spacing=15).guarantee
        self.assertNotIn("1/2", constrained)
        self.assertIn("NO approximation guarantee", constrained)
        unconstrained = select_panel(pool, spec, size=1, min_spacing=0, max_shared_off=10**9)
        self.assertIn("1 - 1/e", unconstrained.guarantee)


if __name__ == "__main__":
    unittest.main()
