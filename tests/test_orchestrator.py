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
from cas13guide.contract.spec import TargetSpec
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


def world(include_target: bool, adequate: bool = True) -> Transcriptome:
    """The corpus. ``adequate`` decides whether it covers the assay's universes.

    It used to hold a single CODING locus while every spec declared five, which
    no test could notice: the index reported the universes it was *asked* for,
    so a corpus of one biotype and a declaration of five agreed. Now the corpus
    is the evidence, and a fixture that wants a run to be adequate has to
    supply one.
    """
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
    return cover_universes(t) if adequate else t


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

    def test_every_panel_member_cleared_the_gate(self):
        """The invariant that protects a reader ordering these oligos.

        This used to assert the panel was a *subset of the frontier*, and that
        was the constraint rather than the observation: `select_panel` was
        handed `pareto_front(passing)`. Non-domination is computed over
        `metrics.objectives()`, which has no positional axis, while the panel's
        binding constraint is `min_spacing = spacer_length`. So the pool was
        filtered by a criterion orthogonal to feasibility, and over 72 designs
        run twice with only the pool changed, 27 differed -- 27 better, 0
        worse. On a 240 nt target the frontier collapsed to one guide and the
        panel to one member at coverage 0.7647, where the full PASS set fills
        three at 0.9870: a miss probability of 0.2353 against 0.0130.

        Being dominated on aggregate objectives does not make a guide unfit to
        order. It makes it not the best on those axes, which is a different
        claim from the one a panel makes.
        """
        report = run(make_spec(), world(include_target=True))
        verdicts = {row["guide_id"]: row["verdict"]
                    for row in report.to_dict()["ranked"]}
        self.assertTrue(report.panel.guides)
        for guide in report.panel.guides:
            self.assertEqual(verdicts.get(guide.guide_id), "PASS", guide.guide_id)

    def test_the_panel_outgrows_the_frontier_when_it_has_to(self):
        """A panel larger than the frontier is impossible if it is drawn from it.

        Asserting the panel is a subset of the PASS set would hold either way,
        since the frontier is a subset of it too. This target is chosen so the
        frontier collapses to a single guide while a hundred certified guides
        sit at spaced sites: the panel is 2 and the frontier is 1, which is
        arithmetic no frontier-restricted selector can produce.

        Non-domination is computed over `metrics.objectives()`, which has no
        positional axis; the panel's binding constraint is
        `min_spacing = spacer_length`, which is nothing else. Over 72 designs
        run twice with only the pool changed, 27 differed -- 27 better, 0
        worse, worst case a miss probability of 0.2353 against 0.0130.
        """
        import random

        rng = random.Random(3081)
        sequence = "".join(rng.choice("AUGC") for _ in range(150))
        spec = make_spec(targets=(
            TargetSpec(target_id="t1", sequence=sequence,
                       target_class=TargetClass.REQUIRED, expression=100.0),
        ))
        transcriptome = Transcriptome("GRCh38", "v44")
        transcriptome.add(Locus("L_T1", sequence, Universe.CODING, gene="TGT",
                                expression=100.0))
        report = run(spec, cover_universes(transcriptome))

        passing = {row["guide_id"] for row in report.to_dict()["ranked"]
                   if row["verdict"] == "PASS"}
        self.assertGreater(len(passing), 1, "fixture sanity: guides must pass")
        self.assertEqual(
            len(report.pareto_frontier), 1,
            "fixture no longer collapses the frontier, so it cannot "
            "distinguish the two pools",
        )
        self.assertGreater(len(report.panel.guides), len(report.pareto_frontier))
        panel_ids = {g.guide_id for g in report.panel.guides}
        self.assertTrue(panel_ids.issubset(passing))

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

    def _partial_world(self, keep: int) -> Transcriptome:
        """A world holding only the first ``keep`` bases of the target.

        Candidates tiling inside the kept prefix find a hit; the rest do not,
        so the zero-hit fraction is a dial. 42 candidates are tiled, and each
        extra base kept converts exactly one zero-hit candidate into a hit.
        """
        t = Transcriptome("GRCh38", "v44")
        t.add(Locus("L_T1", SEQ_A[:keep], Universe.CODING, gene="TGT",
                    expression=100.0))
        return t

    def test_escalation_fires_just_above_its_threshold(self):
        """ZERO_HIT_SUSPICION is 0.5, and the test above sits at 1.0.

        A fraction of 1.0 escalates whatever the threshold is, so raising it
        to 0.6 -- an ordinary-looking re-tune -- left the whole suite green
        while changing whether a run widens its envelope. 38 kept bases put
        25 of 42 candidates at zero hits, which is 0.5952: above 0.5 and
        below 0.6, so this fails if the threshold moves that far.
        """
        events: list[str] = []
        run(make_spec(), self._partial_world(38), events=events)
        self.assertTrue(any("escalating envelope" in e for e in events))

    def test_and_not_at_it(self):
        """42 kept bases give exactly 21 of 42, and the comparison is `>`.

        Half the candidates finding nothing is not yet suspicious. Pinning
        this is what stops the threshold drifting to `>=` as well as what
        stops it drifting in value.
        """
        events: list[str] = []
        run(make_spec(), self._partial_world(42), events=events)
        self.assertFalse(any("escalating envelope" in e for e in events))

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
        """The corpus holds them; the declaration narrows the index past them.

        This is the shape fix #3 was written for -- a clean off-target report
        bought by not looking -- and it is a FAIL because the run could have
        looked. The fixture has to supply the sequences for that to be true:
        while the world held one CODING locus, declaring too few universes
        skipped nothing, and the test passed on a technicality.
        """
        spec = make_spec(universes=frozenset({Universe.CODING}))
        report = run(spec, world(include_target=True))
        self.assertEqual(report.verdict_counts["PASS"], 0)
        self.assertGreater(report.verdict_counts["FAIL"], 0)
        self.assertTrue(
            any("universes" in str(r).lower() for r in report.provenance["records"])
        )

    def test_a_corpus_that_never_held_them_is_a_review_not_a_failure(self):
        """No verdict on a guide can lift a gap in the evidence supplied.

        Same declaration, same assay, same shortfall against
        ASSAY_REQUIRED_UNIVERSES -- and a corpus with nothing in those
        universes to skip. Failing a guide for that says the guide is bad; it
        is the run that is uninformative, identically for every candidate, and
        the web form has no way to supply a background at all.
        """
        spec = make_spec(universes=frozenset({Universe.CODING}))
        report = run(spec, world(include_target=True, adequate=False))
        self.assertEqual(report.verdict_counts["FAIL"], 0)
        self.assertGreater(report.verdict_counts["REVIEW"], 0)
        self.assertEqual(report.verdict_counts["PASS"], 0)


if __name__ == "__main__":
    unittest.main()
