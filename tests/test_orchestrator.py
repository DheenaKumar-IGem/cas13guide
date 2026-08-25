"""Control flow: the backtrack loops must fire, and must always terminate."""

import random
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import Universe
from cas13guide.pipeline.orchestrator import DesignContext, run_design
from cas13guide.services.calibration import CalibrationService
from cas13guide.services.indexes import ExhaustiveScanIndex
from cas13guide.services.registry import (
    HeuristicCas13Model,
    ModelRegistry,
    NullAccessibilityModel,
)
from cas13guide.services.transcriptome import Locus, Transcriptome

RNG = random.Random(17)


def seeded_calibration(spec, n=120):
    cs = CalibrationService()
    for regime in ("mm0", "mm1-2", "mm3+"):
        key = CalibrationService.stratum_key(
            spec.profile.effector.value, spec.profile.assay.value, regime
        )
        for i in range(n):
            truth = RNG.random()
            # a group per simulated transcript: without one the stratum has no
            # stated provenance and can no longer certify a bound
            cs.add_observation(
                key, min(1.0, max(0.0, truth + RNG.gauss(0, 0.1))), truth,
                group=f"T{i % 12:02d}",
            )
    return cs


def world(include_target: bool) -> Transcriptome:
    t = Transcriptome("GRCh38", "v44")
    if include_target:
        t.add(Locus("L_T1", SEQ_A, Universe.CODING, gene="TGT", expression=100.0))
    else:
        t.add(
            Locus(
                "L_OTHER",
                "".join(RNG.choice("AUGC") for _ in range(300)),
                Universe.CODING,
                expression=10.0,
            )
        )
    return t


def run(spec, transcriptome, step=1, events=None):
    universes = spec.context.declared_universes
    ctx = DesignContext(
        "reference",
        transcriptome,
        ExhaustiveScanIndex(transcriptome, universes),
        {"L_T1": "t1"},
    )
    return run_design(
        spec,
        [ctx],
        ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel()),
        seeded_calibration(spec),
        panel_size=2,
        initial_step=step,
        on_event=events.append if events is not None else None,
    )


class TestHappyPath(unittest.TestCase):
    def test_run_completes_and_produces_a_panel(self):
        spec = make_spec()
        report = run(spec, world(include_target=True))
        self.assertGreater(sum(report.verdict_counts.values()), 0)
        self.assertGreaterEqual(report.panel.size, 1)
        self.assertEqual(report.spec_fingerprint, spec.fingerprint())
        self.assertTrue(report.validation_plan)
        self.assertTrue(report.provenance["records"])

    def test_controls_are_generated_and_independently_verified(self):
        """Re-run the search ourselves rather than trusting the flag.

        ``verified_no_hits`` is set by the code under test, so asserting it
        alone would pass even if verification were removed. Controls must also
        be checked at the *pipeline's* envelope, not a shallower one.
        """
        spec = make_spec()
        transcriptome = world(include_target=True)
        report = run(spec, transcriptome)
        index = ExhaustiveScanIndex(transcriptome, spec.context.declared_universes)
        depth = spec.thresholds.required_search_mismatches

        self.assertEqual(len(report.controls), report.panel.size)
        panel_spacers = {g.spacer for g in report.panel.guides}
        for control in report.controls:
            self.assertTrue(control.verified_no_hits)
            self.assertNotEqual(control.spacer, "")
            self.assertNotIn(control.spacer, panel_spacers)
            # Composition-matched: same multiset of bases as its partner.
            partner = next(
                g for g in report.panel.guides if g.guide_id == control.matched_to
            )
            self.assertEqual(sorted(control.spacer), sorted(partner.spacer))
            # And genuinely non-targeting at the envelope the guides were judged at.
            self.assertEqual(index.search("check", control.spacer, depth).hits, ())

    def test_pareto_frontier_is_reported(self):
        spec = make_spec()
        report = run(spec, world(include_target=True))
        self.assertTrue(report.pareto_frontier)
        panel_ids = {g.guide_id for g in report.panel.guides}
        self.assertTrue(panel_ids.issubset(set(report.pareto_frontier)))

    def test_no_failed_guide_reaches_the_ranking(self):
        spec = make_spec()
        report = run(spec, world(include_target=True))
        for entry in report.to_dict()["ranked"]:
            self.assertNotEqual(entry["verdict"], "FAIL")


class TestBacktrackLoops(unittest.TestCase):
    def test_search_escalation_fires_and_stops_at_budget(self):
        """C self-loop: a suspicious zero-hit rate escalates, then gives up."""
        spec = make_spec()
        events: list[str] = []
        report = run(spec, world(include_target=False), events=events)

        escalations = [e for e in events if "escalating envelope" in e]
        self.assertEqual(len(escalations), spec.budget.max_search_escalations)
        self.assertTrue(any("budget exhausted" in s for s in report.shortfalls))

    def test_shortfall_is_reported_not_silently_swallowed(self):
        spec = make_spec()
        report = run(spec, world(include_target=False))
        self.assertTrue(report.shortfalls)
        # And the control flow decisions are on the record.
        kinds = {r["name"] for r in report.provenance["records"]}
        self.assertIn("search_escalation", kinds)
        self.assertIn("telemetry", kinds)

    def test_panel_retile_loop_fires_and_is_bounded(self):
        """G->A loop: fires when no feasible panel exists, and terminates.

        Asserting only ``<= budget`` would be satisfied by never looping at
        all -- deleting the retile branch entirely would pass. Pin the exact
        count: step 4 -> 2 -> 1 is two retiles.
        """
        spec = make_spec()
        events: list[str] = []
        run(spec, world(include_target=False), step=4, events=events)
        retiles = [e for e in events if "re-tiling" in e]
        self.assertEqual(len(retiles), spec.budget.max_panel_retiles)
        self.assertIn("step=2", retiles[0])
        self.assertIn("step=1", retiles[1])

    def test_shortfalls_are_recorded_once_not_once_per_iteration(self):
        spec = make_spec()
        report = run(spec, world(include_target=False), step=4)
        search_shortfalls = [s for s in report.shortfalls if "search escalation" in s]
        self.assertEqual(len(search_shortfalls), 1)

    def test_terminates_even_when_nothing_works(self):
        """The real invariant: bounded loops, always an answer."""
        spec = make_spec()
        report = run(spec, world(include_target=False), step=8)
        self.assertIsNotNone(report)
        self.assertTrue(report.shortfalls)


class TestUniverseGateEndToEnd(unittest.TestCase):
    def test_declaring_too_few_universes_fails_everything(self):
        spec = make_spec(universes=frozenset({Universe.CODING}))
        report = run(spec, world(include_target=True))
        self.assertEqual(report.verdict_counts["PASS"], 0)
        self.assertGreater(report.verdict_counts["FAIL"], 0)
        self.assertTrue(
            any("universes" in str(r).lower() for r in report.provenance["records"])
        )


if __name__ == "__main__":
    unittest.main()
