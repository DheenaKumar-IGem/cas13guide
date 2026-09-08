"""Seventh review pass: three defects, and one instrument that came back clean.

The pass drove the Arc path end to end, compared the CLI against the web server
field by field, swept the spec thresholds, fed the tool its own exports, and put
the server under genuine concurrent load. Concurrency found nothing that
survived verification -- which, for a threaded dev server that runs a
multi-second design inside the request, is worth recording.

What it did find were three ways a run could alter or omit its own input without
saying so.
"""

import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def _sequence(seed_value: int, length: int) -> str:
    import random

    rng = random.Random(seed_value)
    return "".join(rng.choice("ACGU") for _ in range(length))


class TestAnUntileableTargetSaysSoRatherThanVanishing(unittest.TestCase):
    """A target shorter than the window is dropped, correctly, and silently.

    ``design_candidates`` filters on ``len(sequence) >= spacer_length``, and the
    run then reported only "no feasible panel covers required target(s): t1" --
    which reads as "t1 has no usable guides" when the truth is that nobody
    looked. ``designability()`` produces exactly this explanation, and only the
    preset path with no explicit ``--spacer-length`` ever asked it, so passing a
    length suppressed it. ``--arc`` makes that unavoidable rather than
    incidental: it refuses every length except 30, and all five presets are
    22-24 nt.

    Worse on a multi-record FASTA: a 25 nt REQUIRED record and a 300 nt second
    one produced 271 scored candidates and a full export while the REQUIRED
    target was never examined.
    """

    def _run(self, **kw):
        from cas13guide.services.workflow import DesignRequest, run_request

        return run_request(
            DesignRequest(effector="RfxCas13d", assay="cell_culture_knockdown",
                          task="knockdown", use_structure=False, use_signal=False, **kw)
        )

    def test_a_preset_shorter_than_an_explicit_spacer_length_says_why(self):
        outcome = self._run(preset="hsa-miR-21-5p", spacer_length=30)
        shortfalls = " ".join(outcome.report["shortfalls"])
        self.assertIn("shorter than the 30 nt spacer", shortfalls)
        self.assertIn("never tiled", shortfalls)

    def test_the_reason_code_travels_with_it(self):
        from cas13guide.contract.enums import ReasonCode

        outcome = self._run(preset="hsa-miR-21-5p", spacer_length=30)
        codes = [
            record["detail"].get("shortfall_codes")
            for record in outcome.report["provenance"]["records"]
            if record["name"] == "telemetry"
        ]
        self.assertIn(ReasonCode.TARGET_TOO_SHORT_TO_TILE.value, codes[0])

    def test_a_multi_record_run_names_the_record_nobody_looked_at(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mix.fa"
            path.write_text(
                f">shortfirst\n{_sequence(1, 25)}\n>long\n{_sequence(2, 300)}\n",
                encoding="utf-8",
            )
            outcome = self._run(fasta_path=str(path), spacer_length=30)
        shortfalls = " ".join(outcome.report["shortfalls"])
        self.assertIn("shortfirst", shortfalls)
        self.assertIn("never tiled", shortfalls)

    def test_a_target_that_fits_says_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ok.fa"
            path.write_text(f">t1\n{_sequence(3, 300)}\n", encoding="utf-8")
            outcome = self._run(fasta_path=str(path), spacer_length=30)
        self.assertNotIn("never tiled", " ".join(outcome.report["shortfalls"]))


class TestBothDoorsDecodeAnUploadTheSameWay(unittest.TestCase):
    """The web upload used ``errors="replace"``.

    So a cp1252 FASTA -- routine from Excel, older NCBI dumps, or any header
    with an accented name -- was refused outright by the CLI as unreadable, and
    on the web ran to completion with its record name silently swapped to
    U+FFFD. A sequence-bearing file altered without a word, on the one door
    whose users cannot read a traceback.
    """

    def test_the_shared_helper_refuses_undecodable_bytes(self):
        from cas13guide.services.fasta import SequenceInputError, decode_text

        with self.assertRaises(SequenceInputError) as caught:
            decode_text(">caf\xe9_target\nAUGC\n".encode("latin-1"), "caf.fa")
        self.assertIn("caf.fa", str(caught.exception))
        self.assertIn("not readable as text", str(caught.exception))

    def test_it_consumes_a_byte_order_mark_like_the_file_door(self):
        from cas13guide.services.fasta import decode_text

        self.assertEqual(decode_text(">a\nAUGC\n".encode("utf-8-sig"), "x.fa"), ">a\nAUGC\n")

    def test_the_upload_path_refuses_rather_than_replacing(self):
        from cas13guide.services.fasta import SequenceInputError
        from cas13guide.services.webapp import _parse_multipart

        boundary = "b0undary"
        payload = ">caf\xe9_target\nAUGCAUGCAUGCAUGCAUGC\n".encode("latin-1")
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="fasta"; filename="caf.fa"\r\n\r\n'
        ).encode("ascii") + payload + f"\r\n--{boundary}--\r\n".encode("ascii")
        with self.assertRaises(SequenceInputError):
            _parse_multipart(body, f"multipart/form-data; boundary={boundary}")

    def test_a_utf8_upload_is_unaffected(self):
        from cas13guide.services.webapp import _parse_multipart

        boundary = "b0undary"
        payload = ">café_target\nAUGCAUGCAUGCAUGCAUGC\n".encode("utf-8")
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="fasta"; filename="caf.fa"\r\n\r\n'
        ).encode("ascii") + payload + f"\r\n--{boundary}--\r\n".encode("ascii")
        fields = _parse_multipart(body, f"multipart/form-data; boundary={boundary}")
        self.assertIn("café_target", fields["fasta__file"])


class TestGapCharactersAreRefusedRatherThanDeleted(unittest.TestCase):
    """``-`` and ``*`` sat in the same bucket as whitespace.

    Deleting a gap is not formatting: it splices the base before the run onto
    the base after it, and the tiler then walks a window across a junction that
    exists in no molecule. Measured on a 400 nt target with a 30-base gapped
    run, 21 of 297 exported guides had a target site occurring nowhere in the
    real sequence. The module's own rule is that nothing is quietly repaired --
    a pasted alignment's ruler digits were refused with a located message while
    the same alignment's gaps were removed without one.
    """

    def test_a_gap_is_located_and_explained(self):
        from cas13guide.services.fasta import SequenceInputError, parse_sequence_input

        with self.assertRaises(SequenceInputError) as caught:
            parse_sequence_input("AUGC-AUGC")
        message = str(caught.exception)
        self.assertIn("position 5", message)
        self.assertIn("alignment gap", message)
        self.assertIn("'N'", message)

    def test_a_masked_base_is_refused_too(self):
        from cas13guide.services.fasta import SequenceInputError, parse_sequence_input

        with self.assertRaises(SequenceInputError):
            parse_sequence_input("AUGC*AUGC")

    def test_the_file_door_refuses_it_as_well(self):
        from cas13guide.services.fasta import SequenceInputError, read_fasta

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gapped.fa"
            path.write_text(">aligned\nAUGC----AUGC\n", encoding="utf-8")
            with self.assertRaises(SequenceInputError) as caught:
                read_fasta(path)
        self.assertIn("alignment gap", str(caught.exception))

    def test_no_spacer_can_now_be_designed_across_a_spliced_junction(self):
        """The consequence, stated as the property that matters."""
        from cas13guide.services.fasta import SequenceInputError, parse_sequence_input

        left, right = _sequence(4, 60), _sequence(5, 60)
        with self.assertRaises(SequenceInputError):
            parse_sequence_input(f"{left}{'-' * 30}{right}")

    def test_whitespace_and_wrapping_are_still_removed(self):
        from cas13guide.services.fasta import parse_sequence_input

        [record] = parse_sequence_input("  aug c\n\tAUGC  \n")
        self.assertEqual(record.sequence, "AUGCAUGC")

    def test_dna_is_still_transcribed(self):
        """The narrowing must not have touched the one repair that is allowed."""
        from cas13guide.services.fasta import parse_sequence_input

        self.assertEqual(parse_sequence_input("ATGCATGC")[0].sequence, "AUGCAUGC")


if __name__ == "__main__":
    unittest.main()
