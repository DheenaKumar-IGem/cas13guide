"""Stage G control guides: matched, and verified at the pipeline's own depth."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import ReasonCode, TargetClass, Universe, Verdict
from cas13guide.metrics.bounds import Interval
from cas13guide.models.evidence import (
    Candidate,
    GateResult,
    GuideAssessment,
    MetricVector,
    PairEvidence,
)
from cas13guide.pipeline.g_output import generate_controls
from cas13guide.pipeline.optimizer import Panel
from cas13guide.services.indexes import ExhaustiveScanIndex
from cas13guide.services.transcriptome import Locus, Transcriptome
from cas13guide.util.seq import revcomp

SPACER = "GCAUUGCUGUGGAGCUGAAGUC"
FILLER = "AUACGAUCAGGUACCAUGACUUGACCAUGGAUCAGUUACGGAUCCAAUGCAUCAGGUACAUC"


def one_guide_panel():
    assessment = GuideAssessment(
        guide_id="g1",
        candidate=Candidate(guide_id="g1", spacer=SPACER, source_target_id="t1", start=0),
        qc=clean_qc("g1"),
        metrics=MetricVector(0.9, 0.0, 0.0, 0.0, 0.9, 1.0),
        gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
        evidence=(
            PairEvidence(
                guide_id="g1",
                locus_id="L_t1",
                context="reference",
                hit=make_hit("g1", "L_t1", "t1", TargetClass.REQUIRED),
                activity=Interval(0.9, 0.9, 0.95),
                expression=Interval(0.5, 0.5, 0.5),
                hazard_weight=0.0,
            ),
        ),
    )
    return Panel(guides=(assessment,), coverage={"t1": 0.9}, objective=0.9, guarantee="test")


def world(extra: str | None = None) -> Transcriptome:
    t = Transcriptome("GRCh38", "v44")
    t.add(Locus("L_t1", SEQ_A, Universe.CODING, expression=100.0))
    if extra is not None:
        t.add(Locus("OFF1", FILLER + extra + FILLER, Universe.CODING, expression=500.0))
    return t


def plant_site(spacer: str, n_mismatch: int) -> str:
    """A binding site for ``spacer`` carrying exactly ``n_mismatch`` mismatches."""
    site = list(revcomp(spacer))
    # Spread them so neither terminal seed survives intact.
    for i in [3, 18, 10][:n_mismatch]:
        site[i] = "C" if site[i] != "C" else "G"
    return "".join(site)


class TestControlGuides(unittest.TestCase):
    def setUp(self):
        self.spec = make_spec()
        self.panel = one_guide_panel()
        self.universes = self.spec.context.declared_universes

    def _controls(self, transcriptome, **kw):
        index = ExhaustiveScanIndex(transcriptome, self.universes)
        return generate_controls(self.panel, self.spec, index, seed=0, **kw)

    def test_control_is_composition_matched_and_not_the_original(self):
        [control] = self._controls(world())
        self.assertTrue(control.verified_no_hits)
        self.assertNotEqual(control.spacer, SPACER)
        self.assertEqual(sorted(control.spacer), sorted(SPACER))

    def test_a_three_mismatch_site_disqualifies_the_control(self):
        """The envelope must match the pipeline's, not a shallower literal.

        Verifying at <=2mm would keep a control that the pipeline itself would
        report as a credible off-target -- a 'non-targeting' control that
        targets something.
        """
        [clean] = self._controls(world())
        decoy = plant_site(clean.spacer, 3)

        [after] = self._controls(world(decoy))
        self.assertNotEqual(
            after.spacer,
            clean.spacer,
            "control with a 3-mismatch site was still accepted; "
            "the verification envelope is too shallow",
        )

    def test_a_shallow_envelope_would_have_kept_it(self):
        """Pins that the previous behaviour really was different."""
        [clean] = self._controls(world())
        decoy = plant_site(clean.spacer, 3)
        [shallow] = self._controls(world(decoy), envelope=2)
        self.assertEqual(shallow.spacer, clean.spacer)

    def test_no_acceptable_control_is_reported_honestly(self):
        """When every scramble hits something, say so rather than shipping one."""
        [clean] = self._controls(world())
        decoy = plant_site(clean.spacer, 0)
        controls = self._controls(world(decoy), attempts=1)
        for control in controls:
            if not control.verified_no_hits:
                self.assertEqual(control.spacer, "")
                self.assertIn("no composition-matched scramble", control.note)



class TestControlsClearTheSameQcAsRealGuides(unittest.TestCase):
    """A control the pipeline would FAIL is not a usable control.

    ``generate_controls`` re-implemented a two-line subset of stage B -- a
    hardcoded homopolymer limit and the poly-U check -- so a scramble carrying a
    GC violation, a self-folding stem or a direct-repeat interaction could ship
    as a "matched" control while the gate would have rejected the same sequence
    as deterministically incompatible.
    """

    def setUp(self):
        self.spec = make_spec()
        self.panel = one_guide_panel()
        self.universes = self.spec.context.declared_universes

    def test_every_generated_control_passes_stage_b_qc(self):
        from cas13guide.models.evidence import Candidate
        from cas13guide.pipeline.b_qc import qc_candidate

        index = ExhaustiveScanIndex(world(), self.universes)
        controls = generate_controls(self.panel, self.spec, index, seed=0)
        self.assertTrue(controls)
        for control in controls:
            if not control.verified_no_hits:
                continue
            probe = Candidate(
                guide_id=control.guide_id,
                spacer=control.spacer,
                source_target_id="t1",
                start=0,
            )
            result = qc_candidate(probe, self.spec)
            self.assertTrue(
                result.passed,
                f"control {control.spacer} would FAIL the gate: {result.flags}",
            )

    def test_composition_is_still_matched_exactly(self):
        index = ExhaustiveScanIndex(world(), self.universes)
        [control] = generate_controls(self.panel, self.spec, index, seed=0)
        self.assertEqual(sorted(control.spacer), sorted(SPACER))


if __name__ == "__main__":
    unittest.main()
