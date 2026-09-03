"""What the tenth pass's eight dimensions missed, found by asking.

After the finders and the skeptics, one more agent was asked a different
question: not "is there a bug here" but "what did those eight dimensions not
look at, and what would you expect to find there". It named six places and went
and checked the most promising one. Three of its leads were real.

The first is the worst thing found in this round. The other two are the shape
this codebase keeps producing: a fix applied to one field and not to the
identical field beside it.
"""

import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

from helpers import *  # noqa: F401,F403

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _fasta(directory: pathlib.Path, records: dict[str, str]) -> pathlib.Path:
    path = directory / "input.fa"
    path.write_text(
        "".join(f">{k}\n{v}\n" for k, v in records.items()), encoding="utf-8"
    )
    return path


class TestAGeneNameCanBePrinted(unittest.TestCase):
    """A name the FASTA reader is built to accept killed the CLI.

    ``read_fasta`` decodes with ``utf-8-sig``, so a UTF-8 header is supported
    input by design. The record name becomes a ``target_id`` the CLI prints to
    a stream left at the platform default -- cp1252 on Windows -- and the print
    raises. The trigger is ordinary nomenclature: NF-kappaB1, IFN-gamma1,
    TGF-beta1, HIF1-alpha.

    Where it landed made it worse than a bad character. The crash came *after*
    the design succeeded and *before* the ``--out`` block, so a run with a full
    ranking and a panel wrote no files and returned 1 -- the status this
    command's own epilog defines as "the run completed and nothing survived the
    gate". A wrapper read a complete design as an empty one. The browser door
    rendered the same bytes without complaint, so the two doors that "cannot
    drift into behaving differently" did.
    """

    #: NF-kappa-B1, as written in every paper that mentions it.
    NAME = "NF-κB1"

    def test_the_run_completes_and_writes_its_exports(self):
        rng = random.Random(8)
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            fasta = _fasta(work, {
                self.NAME: "".join(rng.choice("ACGU") for _ in range(200))
            })
            out = work / "out"
            result = subprocess.run(
                [sys.executable, "-m", "cas13guide.cli", "design",
                 "--fasta", str(fasta), "--expression", "0", "--top", "1",
                 "--out", str(out), "--stem", "u"],
                cwd=ROOT, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(result.returncode, 0, result.stderr[-800:])
            written = sorted(p.name for p in out.iterdir())
        self.assertEqual(len(written), 4, written)

    def test_the_name_survives_into_the_export(self):
        from cas13guide.services.fasta import ranked_to_csv
        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(8)
        with tempfile.TemporaryDirectory() as directory:
            fasta = _fasta(pathlib.Path(directory), {
                self.NAME: "".join(rng.choice("ACGU") for _ in range(200))
            })
            outcome = run_request(DesignRequest(
                fasta_path=str(fasta), task="knockdown",
                assay="cell_culture_knockdown", effector="RfxCas13d",
                use_structure=False, use_signal=False))
        self.assertIn(self.NAME, ranked_to_csv(outcome.report))

    def test_it_shows_what_was_there_rather_than_a_question_mark(self):
        """``backslashreplace``, not ``replace``: a console that cannot render
        the character should print an escape naming it, which is recoverable,
        rather than ``?``, which is not."""
        import inspect

        from cas13guide import cli

        source = inspect.getsource(cli._make_output_printable)
        self.assertIn("backslashreplace", source)
        self.assertNotIn('errors="replace"', source)


class TestControlsAreHeldToTheQcTheRunUsed(unittest.TestCase):
    """``generate_controls`` QC'd its shuffles without the folding engine.

    The run QCs real guides with the ViennaRNA MFE and records
    ``qc/vienna-mfe/v1`` in the ledger for the whole run; the control probe
    fell back to the stem-length proxy, which catches about a third of what a
    real fold does. So a control could pass here and be FAILed by the QC that
    actually ran -- which is the divergence the comment directly above that
    call says it removed.
    """

    def _panel_of_one(self):
        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.models.evidence import GateResult, GuideAssessment, MetricVector
        from cas13guide.pipeline.optimizer import Panel

        member = GuideAssessment(
            guide_id="g1",
            candidate=make_candidate("g1"),
            qc=clean_qc("g1"),
            metrics=MetricVector(1.0, 1.0, 0.0, 0.0, 1.0, 1.0),
            gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
        )
        return Panel(guides=(member,), coverage={"t1": 1.0}, objective=1.0,
                     guarantee="test", dropped=())

    class _NoHits:
        """An index where nothing binds, so only QC decides."""

        version = "test/none"

        def any_hit(self, spacer, depth):
            return False

    def test_the_fold_engine_reaches_the_control_probe(self):
        """A fold that calls every sequence self-folding must leave no control
        standing. Without it the proxy accepts them and the controls ship."""
        from cas13guide.pipeline.g_output import generate_controls

        panel, spec = self._panel_of_one(), make_spec()

        without = generate_controls(panel, spec, self._NoHits(), seed=1)
        with_fold = generate_controls(panel, spec, self._NoHits(), seed=1,
                                      fold=lambda sequence: -30.0)

        self.assertTrue(without[0].verified_no_hits,
                        "fixture must find a control when QC is permissive")
        self.assertFalse(
            with_fold[0].verified_no_hits,
            "a control the run's own QC would FAIL was shipped anyway",
        )
        self.assertIn("scramble", with_fold[0].note)

    def test_the_orchestrator_passes_it(self):
        """The engine existed in ``run_design`` all along; the defect was that
        it stopped at the guides."""
        import inspect

        from cas13guide.pipeline import orchestrator

        source = inspect.getsource(orchestrator.run_design)
        self.assertRegex(source, r"generate_controls\([^)]*fold=self_fold")


class TestNoFoldIsRunForAnAnswerNobodyReads(unittest.TestCase):
    """``duplex_dg_per_nt`` has one reader, and it skips non-intended pairs.

    The same waste as accessibility in the field beside it, left behind when
    that one was fixed: a duplex fold run for every off-target hit and
    discarded. It grows with the background corpus while the intended folds
    stay fixed, so the discarded share rises with the size of the search.
    """

    class _NoExpression:
        def expression_tpm(self, locus_id):
            return None

    def _world(self, hit):
        from cas13guide.contract.enums import Universe
        from cas13guide.services.transcriptome import Locus, Transcriptome

        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("L1", hit.target_site, Universe.CODING, expression=10.0))
        return world

    def test_an_off_target_pair_costs_no_fold(self):
        from cas13guide.pipeline.d_pair_evidence import evidence_for_hit
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.registry import (
            HeuristicCas13Model, ModelRegistry, NullAccessibilityModel,
        )

        class _CountingEnergy:
            version = "counting/v1"
            calls = 0

            def duplex_dg_per_nt(self, spacer, site):
                type(self).calls += 1
                return -1.5

        energy = _CountingEnergy()
        hit = make_hit("g1", "L1")  # no target_class -> not intended
        evidence = evidence_for_hit(
            hit, "GCAUUGCUGUGGAGCUGAAGUC", make_spec(),
            ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel(),
                          energy=energy),
            CalibrationService(), self._world(hit), self._NoExpression())

        self.assertIsNone(evidence.duplex_dg_per_nt)
        self.assertEqual(_CountingEnergy.calls, 0,
                         "a fold was run for a value nothing reads")

    def test_an_intended_pair_still_carries_it(self):
        """The saving must not cost the number that is actually read."""
        from cas13guide.contract.enums import TargetClass
        from cas13guide.pipeline.d_pair_evidence import evidence_for_hit
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.registry import (
            HeuristicCas13Model, ModelRegistry, NullAccessibilityModel,
        )

        class _Energy:
            version = "fixed/v1"

            def duplex_dg_per_nt(self, spacer, site):
                return -1.5

        hit = make_hit("g1", "L1", target_id="t1",
                       target_class=TargetClass.REQUIRED)
        evidence = evidence_for_hit(
            hit, "GCAUUGCUGUGGAGCUGAAGUC", make_spec(),
            ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel(),
                          energy=_Energy()),
            CalibrationService(), self._world(hit), self._NoExpression())

        self.assertEqual(evidence.duplex_dg_per_nt, -1.5)


if __name__ == "__main__":
    unittest.main()
