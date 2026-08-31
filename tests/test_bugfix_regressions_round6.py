"""Sixth review pass: the subcommands, the file-writing side, and a namespace.

Five passes had exercised ``design`` and ``serve``. This one went at what was
still never run: ``train`` and ``demo``, the ``--out``/``--stem`` paths, what
happens when a user-controlled record name reaches an output surface, and
whether state survives between runs in one process.

The critical find is a namespace collision -- ``"reference"`` is both a
reserved context key and a name a user may give a FASTA record -- and it is
the kind only fuzzing finds: every module involved is individually correct.

Three of the eight came from the previous passes' own fixes, again.
"""

import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def _sequence(seed_value: int, length: int) -> str:
    import random

    rng = random.Random(seed_value)
    return "".join(rng.choice("ACGU") for _ in range(length))


class TestReferenceIsAReservedContextName(unittest.TestCase):
    """A variant record named ``reference`` replaced the reference context.

    Context names mix one reserved literal with user-supplied FASTA record
    names, and nothing separated them. ``orchestrator`` files per-guide
    evidence under ``ctx.name`` and ``e_aggregate`` reads the intended target's
    evidence back with ``by_context.get("reference")``, so two contexts under
    one key meant the variant's evidence overwrote the target's and every
    downstream number for the REQUIRED target was computed against the variant.

    Measured: renaming one variant record and changing nothing else took a run
    from PASS=96, coverage 0.7647 to PASS=0, coverage 0.0 -- with a shortfall
    claiming no feasible panel covers a target that was never evaluated.
    ``_refuse_duplicate_names`` cannot see it: the record names really are
    distinct, and the collision is with a literal that is not a record at all.
    """

    def test_variant_contexts_refuses_the_reserved_name(self):
        from cas13guide.contract.enums import Universe
        from cas13guide.services.conservation import variant_contexts
        from cas13guide.services.fasta import SequenceRecord

        with self.assertRaises(ValueError) as caught:
            variant_contexts(
                [SequenceRecord("GENE1", SEQ_A), SequenceRecord("reference", SEQ_B)],
                frozenset(Universe),
                target_id="GENE1",
            )
        self.assertIn("reference", str(caught.exception))

    def test_the_first_record_may_of_course_be_called_that(self):
        """It is the reference. Only a *variant* is refused."""
        from cas13guide.contract.enums import Universe
        from cas13guide.services.conservation import variant_contexts
        from cas13guide.services.fasta import SequenceRecord

        contexts = variant_contexts(
            [SequenceRecord("reference", SEQ_A), SequenceRecord("v1", SEQ_B)],
            frozenset(Universe),
            target_id="reference",
        )
        self.assertEqual([c.name for c in contexts], ["reference", "v1"])

    def test_the_refusal_reaches_the_cli_as_a_located_message(self):
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            run_request,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "t.fa"
            target.write_text(f">GENE1\n{_sequence(1, 200)}\n", encoding="utf-8")
            variants = root / "v.fa"
            variants.write_text(f">reference\n{_sequence(2, 200)}\n", encoding="utf-8")
            with self.assertRaises(DesignRequestError) as caught:
                run_request(
                    DesignRequest(fasta_path=str(target), variants_path=str(variants),
                                  use_structure=False, use_signal=False)
                )
        self.assertIn("--variants", str(caught.exception))


class TestTheWebFormKeepsItsTwoSourcesApart(unittest.TestCase):
    """The form folded the upload into the paste before the guard could see it.

    ``_read_records`` began refusing more than one target source, with a
    comment naming the web form as the reason -- and ``_request_from_fields``
    collapsed ``sequence`` and ``fasta__file`` into one field, so ``--fasta``
    could never appear in the refusal and a filled textarea silently discarded
    an uploaded file. Reproduced over real HTTP: the paste+upload response was
    byte-identical to the paste-only one, with the uploaded record's name
    nowhere on the page.
    """

    def test_the_upload_has_its_own_field(self):
        from cas13guide.services.webapp import _request_from_fields

        request = _request_from_fields({"sequence": "AUGC", "fasta__file": ">x\nGGGG"})
        self.assertEqual(request.sequence_text, "AUGC")
        self.assertEqual(request.fasta_text, ">x\nGGGG")

    def test_an_upload_alone_still_designs(self):
        from cas13guide.services.webapp import _request_from_fields
        from cas13guide.services.workflow import run_request

        request = _request_from_fields(
            {"sequence": "", "fasta__file": f">uploaded\n{_sequence(4, 120)}\n",
             "effector": "RfxCas13d", "assay": "cell_culture_knockdown", "task": "knockdown"}
        )
        outcome = run_request(request)
        self.assertEqual(outcome.records[0].name, "uploaded")

    def test_both_at_once_is_refused(self):
        from cas13guide.services.webapp import _request_from_fields
        from cas13guide.services.workflow import DesignRequestError, run_request

        request = _request_from_fields(
            {"sequence": _sequence(5, 120), "fasta__file": f">x\n{_sequence(6, 120)}\n"}
        )
        with self.assertRaises(DesignRequestError):
            run_request(request)


class TestTrainRefusesUnreadableScreensLegibly(unittest.TestCase):
    """``cli._train`` catches DatasetError and nothing else.

    ``path.exists()`` is not "readable as a UTF-8 text file", so a directory or
    a screen exported from Excel as UTF-16 escaped as PermissionError or
    UnicodeDecodeError -- a traceback, and exit 1, which is the code that
    command uses to mean "loaded, and rejected by the acceptance gate". The
    FASTA door already converted exactly this class into a located message.
    """

    def test_a_directory_is_a_dataset_error(self):
        from cas13guide.services.ingest import DatasetError, load_screen

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(DatasetError) as caught:
                load_screen(directory)
        self.assertIn("cannot open", str(caught.exception))

    def test_a_utf16_screen_is_a_dataset_error(self):
        from cas13guide.services.ingest import DatasetError, load_screen

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utf16.csv"
            path.write_bytes(
                "spacer,activity,transcript\nGCAUUGCUGUGGAGCUGAAGUC,0.5,T1\n".encode("utf-16")
            )
            with self.assertRaises(DatasetError) as caught:
                load_screen(path)
        self.assertIn("UTF-8", str(caught.exception))

    def test_a_named_target_site_column_that_is_absent_names_itself(self):
        """It used to drop every row and blame the spacer column -- the same
        misattribution ``dropped_group`` was split out to stop, one column
        over."""
        from cas13guide.services.ingest import ColumnMap, load_screen

        rows = ["spacer,activity,transcript"]
        rows += [f"GCAUUGCUGUGGAGCUGAAGUC,0.5,T{i % 9}" for i in range(40)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.csv"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            _examples, report = load_screen(path, columns=ColumnMap(target_site="site"))
        errors = " ".join(report.errors)
        self.assertIn("site", errors)
        self.assertNotIn("unusable sequence", " ".join(report.warnings))

    def test_a_good_screen_is_unaffected(self):
        from cas13guide.services.ingest import load_screen

        rows = ["spacer,activity,transcript"]
        rows += [f"GCAUUGCUGUGGAGCUGAAGUC,0.{i % 9},T{i % 9}" for i in range(40)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.csv"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            examples, report = load_screen(path)
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(len(examples), 40)


class TestMeasuredKineticsDoNotKillTheRun(unittest.TestCase):
    """``StiffKineticsError`` is raised from inside stage E.

    Long after ``run_request`` validated the JSON, so it escaped untyped and
    the CLI printed a traceback and exited 1 with no report. The trigger is not
    exotic -- ``kinetics.py`` names ``k_on = 1 /nM/s`` as an ordinary measured
    association rate -- so the flag advertised as the way to unlock absolute
    signal punished the user for supplying real constants.
    """

    def test_it_arrives_as_a_located_message_naming_the_flag(self):
        import json

        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            run_request,
        )

        constants = {
            "k_on": 50.0, "k_off": 0.005, "k_cat": 0.5, "k_m": 200.0,
            "complex_nm": 100.0, "target_nm": 1.0, "reporter_nm": 500.0,
            "assay_seconds": 7200.0, "source": "measured", "provisional": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kin.json"
            path.write_text(json.dumps(constants), encoding="utf-8")
            with self.assertRaises(DesignRequestError) as caught:
                run_request(
                    DesignRequest(preset="hsa-miR-21-5p", kinetics_json=str(path))
                )
        message = str(caught.exception)
        self.assertIn("--kinetics-constants", message)
        self.assertIn("stiff solver", message)


class TestTheExportPathsFailLegibly(unittest.TestCase):
    """Every path problem arrived as a pathlib traceback and exit 1.

    Which is the code ``design`` returns for the legitimate "no guide cleared
    the gate" outcome, so a wrapper script could not tell "nothing ranked" from
    "could not write the report".
    """

    def _run(self, *args):
        import subprocess
        import sys

        return subprocess.run(
            [sys.executable, "-m", "cas13guide", "design", *args],
            capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).resolve().parents[1]),
        )

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.target = Path(self.dir.name) / "t.fa"
        self.target.write_text(f">t1\n{_sequence(7, 120)}\n", encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def test_out_naming_a_file_is_refused_before_the_pipeline_runs(self):
        blocker = Path(self.dir.name) / "notadir.txt"
        blocker.write_text("x", encoding="utf-8")
        result = self._run("--fasta", str(self.target), "--out", str(blocker))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("must name a directory", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_stem_that_is_a_path_cannot_escape_the_out_directory(self):
        out = Path(self.dir.name) / "out"
        result = self._run(
            "--fasta", str(self.target), "--out", str(out), "--stem", "../escaped"
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("plain file name", result.stderr)
        self.assertEqual(list(Path(self.dir.name).glob("escaped*")), [])

    def test_a_normal_export_still_works(self):
        out = Path(self.dir.name) / "good"
        result = self._run("--fasta", str(self.target), "--out", str(out), "--top", "1")
        self.assertIn("wrote:", result.stdout)
        self.assertTrue((out / "candidates.csv").exists())


class TestTheExportHeaderCarriesWhatTheLedgerDoes(unittest.TestCase):
    """The previous pass recorded the calibration parameters in the ledger and
    said it had fixed "the export headers" too. It had not.

    ``provenance_lines`` emitted ``kind: name = version`` and never touched
    ``detail``, so two runs whose every number differed still shipped
    byte-identical headers -- the failure the ledger entry was added to fix,
    surviving one level out, in the file that outlives the session.
    """

    def _header(self, seed):
        from cas13guide.services.fasta import ranked_to_csv
        from cas13guide.services.workflow import DesignRequest, run_request

        outcome = run_request(DesignRequest(preset="hsa-miR-21-5p", seed=seed))
        return [
            line for line in ranked_to_csv(outcome.report).splitlines()
            if line.startswith("# calibration")
        ]

    def test_two_seeds_produce_different_headers(self):
        self.assertNotEqual(self._header(1), self._header(2))

    def test_the_header_names_the_seed(self):
        self.assertIn("seed=3", " ".join(self._header(3)))

    def test_the_rendering_is_stable_for_the_same_parameters(self):
        """Sorted, so identical parameters give identical bytes."""
        from cas13guide.services.fasta import _render_detail

        self.assertEqual(
            _render_detail({"b": 2, "a": 1}), _render_detail({"a": 1, "b": 2})
        )
        self.assertNotIn("\n", _render_detail({"note": "one\ntwo"}))


class TestTheDemoDoesNotClaimAVerdictItsRowsLack(unittest.TestCase):
    """The bundled demo printed "top ranked (PASS only)" two lines under its
    own "0 PASS of 108"."""

    def test_the_header_names_what_ranked_actually_is(self):
        source = (
            Path(__file__).resolve().parents[1] / "examples" / "demo_knockdown.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("top ranked (PASS only)", source)
        self.assertIn("gate survivors", source)

    def test_it_carries_the_triage_note_the_cli_carries(self):
        source = (
            Path(__file__).resolve().parents[1] / "examples" / "demo_knockdown.py"
        ).read_text(encoding="utf-8")
        self.assertIn("not a recommendation", source)


if __name__ == "__main__":
    unittest.main()
