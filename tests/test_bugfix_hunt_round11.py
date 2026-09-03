"""An eleventh pass, aimed at the tenth's own diff.

Three of round ten's eleven findings were regressions introduced by round nine,
so this round pointed a finder straight at round ten's changes and spent the
rest on places no round had covered: BLAST coordinate handling, the preset
tiling path, the browser door, the panel posterior, conservation.

Fourteen candidates, ten refuted, four confirmed. The count is falling -- 8, then
11 with its critic, now 4 -- and the one that matters here is not a small one.
"""

import pathlib
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from helpers import *  # noqa: F401,F403


class TestThePosteriorIsWindowedWhereTheGuideSits(unittest.TestCase):
    """FR6.3's headline number was conditioned on the wrong point of the scale.

    ``for_panel`` passed ``benefit_lower(evidence.activity)`` -- the conformal
    *lower bound* -- to ``guide_success``, which selects calibration rows with
    ``abs(predicted - x) <= bandwidth``. But ``add_observation`` stores the
    model's raw point prediction, so the window landed a full conformal
    quantile below the rows it was meant to sit among. Like was not compared
    with like.

    The module docstring states both halves of why that is wrong in its first
    paragraph: the rows are the ones "whose prediction sits near this guide's",
    and "nothing here reads a conformal quantile". ``benefit_lower`` *is* the
    prediction minus that quantile, and it was the only thing positioning the
    window.

    It was not a small bias. On a perfectly calibrated screen with q = 0.30, a
    guide predicted at 0.60 -- a level where every calibration row cleared the
    threshold -- came back with posterior mean 0.0012 instead of 0.9988. The
    answer was inverted, and it is the number ``TargetConfidence.lower`` and
    every printed ``success_mean`` rest on.
    """

    THRESHOLD_CROSSING = 0.5

    def _member(self, point: float, q: float):
        from cas13guide.contract.enums import ReasonCode, TargetClass, Verdict
        from cas13guide.contract.spec import TargetSpec
        from cas13guide.metrics.bounds import Interval
        from cas13guide.models.evidence import (
            Candidate, GateResult, GuideAssessment, MetricVector, PairEvidence,
        )
        from cas13guide.pipeline.optimizer import Panel
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.panel_confidence import BayesianPanelConfidence

        spec = make_spec(targets=(TargetSpec(
            target_id="t1", sequence="A" * 120,
            target_class=TargetClass.REQUIRED, expression=100.0),))

        # Perfect calibration: observed == predicted across [0, 1]. So the
        # truth for any guide is simply whether its prediction clears the
        # assay threshold, and the posterior has no excuse to disagree.
        calibration = CalibrationService()
        key = CalibrationService.stratum_key(
            spec.profile.effector.value, spec.profile.assay.value, "mm0")
        for i in range(2000):
            p = i / 1999
            calibration.add_observation(key, p, p, group=f"T{i % 12:02d}")

        hit = make_hit("g1", "L1", target_id="t1",
                       target_class=TargetClass.REQUIRED)
        evidence = PairEvidence(
            guide_id="g1", locus_id="L1", context="reference", hit=hit,
            activity=Interval(point=point, lower=max(0.0, point - q),
                              upper=min(1.0, point + q)),
            expression=Interval(point=1.0, lower=1.0, upper=1.0),
            hazard_weight=1.0,
        )
        assessment = GuideAssessment(
            guide_id="g1",
            candidate=Candidate(guide_id="g1", spacer="A" * 22,
                                source_target_id="t1", start=0),
            qc=clean_qc("g1"),
            metrics=MetricVector(point, 1.0, 0.0, 0.0, point, 1.0),
            gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
            evidence=(evidence,),
        )
        panel = Panel(guides=(assessment,), coverage={"t1": point},
                      objective=point, guarantee="test", dropped=())
        model = BayesianPanelConfidence(calibration, spec)
        return model.for_panel(panel).by_target()["t1"].members[0]

    def test_the_window_sits_at_the_models_prediction(self):
        member = self._member(point=0.60, q=0.30)
        self.assertAlmostEqual(member.prediction, 0.60, places=6)

    def test_a_guide_above_the_threshold_gets_a_posterior_above_it(self):
        """The inverted case. Predicted 0.60 on a perfectly calibrated screen
        whose threshold is 0.50: every neighbouring row succeeded."""
        member = self._member(point=0.60, q=0.30)
        self.assertGreater(member.mean, 0.9, f"{member.successes}/{member.trials}")

    def test_a_guide_below_the_threshold_still_gets_a_low_posterior(self):
        """The fix must not simply push everything up."""
        member = self._member(point=0.20, q=0.10)
        self.assertLess(member.mean, 0.1, f"{member.successes}/{member.trials}")

    def test_the_answer_does_not_move_with_the_conformal_quantile(self):
        """The interval's width is reported beside this number; charging it a
        second time by shifting the window is what made it meaningless."""
        narrow = self._member(point=0.70, q=0.05)
        wide = self._member(point=0.70, q=0.40)
        self.assertAlmostEqual(narrow.prediction, wide.prediction, places=6)
        self.assertAlmostEqual(narrow.mean, wide.mean, places=3)


class TestABlastAmbiguityCodeIsALocatedRefusal(unittest.TestCase):
    """A single IUPAC code in a subject aborted the run with a bare ValueError.

    Refusing is right -- skipping the row would hide a real off-target behind
    an ambiguity code. The defect was the exception *type*. blastn exiting
    non-zero already raises ``BlastUnavailable``, which becomes a
    ``DesignRequestError`` and exit 2; this raised ``ValueError``, matched no
    handler, and escaped as a traceback with exit 1 -- the status documented as
    "the run completed and nothing survived the gate".

    The sequence comes from a third-party database the user cannot edit, and
    RefSeq/nt/TSA routinely carry R, Y, M, K, S, W, B, D, H and V.
    """

    def _row(self, subject: str, sequence: str) -> str:
        spacer_len = len(sequence)
        return "\t".join([
            "g1", subject, "100.0", str(spacer_len), "0", "0",
            "1", str(spacer_len), "55", str(54 + spacer_len),
            "1e-9", "44.1", "plus", sequence,
        ])

    def test_it_raises_the_type_the_caller_already_handles(self):
        from cas13guide.services.blast import BlastOffTargetIndex, BlastUnavailable

        index = BlastOffTargetIndex(database="db", universes=frozenset())
        rows = self._row("NM_2", "RACGUACGUACGUACGUACGUA")
        with mock.patch.object(BlastOffTargetIndex, "_run", return_value=rows):
            with self.assertRaises(BlastUnavailable) as caught:
                index.search("g1", "GCAUUGCUGUGGAGCUGAAGUC", 4)
        message = str(caught.exception)
        self.assertIn("--blast-db", message, "the message must name the flag to act on")
        self.assertIn("NM_2", message, "and the subject responsible")

    def test_blast_unavailable_is_still_what_a_missing_blastn_raises(self):
        """The two failures of one subsystem should reach the user the same
        way; that they did not is the whole finding."""
        from cas13guide.services.blast import BlastUnavailable

        self.assertTrue(issubclass(BlastUnavailable, RuntimeError))

    def test_run_request_turns_it_into_a_located_refusal(self):
        """The handler that makes the exit status right."""
        import inspect

        from cas13guide.services import workflow

        source = inspect.getsource(workflow.run_request)
        self.assertIn("except BlastUnavailable", source)


class TestTheServerAnnouncesThePortItBound(unittest.TestCase):
    """`serve --port 0` printed -- and opened -- an address that cannot work.

    The URL was built from the *requested* port. `--port 0` is the conventional
    "pick a free one", argparse accepts it without restriction, and the server
    duly bound an ephemeral port while announcing `http://127.0.0.1:0/`. The
    working address appeared nowhere. This is the door aimed at people who will
    not use a command line, and the one line of output they get was wrong.
    """

    def test_the_server_binds_a_real_port_when_asked_for_zero(self):
        from cas13guide.services.webapp import make_server

        server = make_server("127.0.0.1", 0)
        try:
            self.assertNotEqual(server.server_address[1], 0)
        finally:
            server.server_close()

    def test_the_announced_url_is_built_from_the_bound_port(self):
        import inspect

        from cas13guide.services import webapp

        source = inspect.getsource(webapp.serve)
        self.assertIn("server.server_address[1]", source)
        self.assertNotIn('url = f"http://{host}:{port}/"', source)


class TestVariantsDoNotRewriteTheProvenance(unittest.TestCase):
    """`--variants` replaced the recorded build and annotation.

    Every variant world was built as `Transcriptome("variants", record.name)`,
    and `run_design` files `transcriptome.version` in the ledger for every
    context. So adding a flag about isoforms discarded the user's
    `--genome-build` and `--annotation` -- including for the context named
    "reference", whose loci are identical to the non-variant run's -- and filed
    a FASTA header in the annotation slot. NFR4 makes provenance the thing that
    outlives the session, and this is the only line saying which annotation was
    searched.
    """

    BUILD, ANNOTATION = "T2T-CHM13v2.0", "RefSeq-110"

    def _versions(self, with_variants: bool):
        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(4)
        sequence = "".join(rng.choice("ACGU") for _ in range(200))
        variant = list(sequence)
        variant[40] = "A" if variant[40] != "A" else "G"

        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            (work / "t.fa").write_text(f">TGT\n{sequence}\n", encoding="utf-8")
            extra = {}
            if with_variants:
                (work / "v.fa").write_text(
                    ">VAR1\n" + "".join(variant) + "\n", encoding="utf-8")
                extra["variants_path"] = str(work / "v.fa")
            outcome = run_request(DesignRequest(
                fasta_path=str(work / "t.fa"), task="knockdown",
                assay="cell_culture_knockdown", effector="RfxCas13d",
                use_structure=False, use_signal=False,
                genome_build=self.BUILD, annotation=self.ANNOTATION, **extra))
        return {
            record["name"]: record["version"]
            for record in outcome.report["provenance"]["records"]
            if record.get("kind") == "transcriptome"
        }

    def test_a_plain_run_records_what_the_user_asked_for(self):
        versions = self._versions(with_variants=False)
        self.assertEqual(set(versions.values()), {f"{self.BUILD}/{self.ANNOTATION}"})

    def test_adding_variants_does_not_change_it(self):
        versions = self._versions(with_variants=True)
        self.assertGreater(len(versions), 1, "fixture must add a variant context")
        for name, version in versions.items():
            with self.subTest(context=name):
                self.assertEqual(version, f"{self.BUILD}/{self.ANNOTATION}")

    def test_no_record_name_is_filed_as_an_annotation(self):
        versions = self._versions(with_variants=True)
        self.assertNotIn("variants/VAR1", set(versions.values()))


if __name__ == "__main__":
    unittest.main()
