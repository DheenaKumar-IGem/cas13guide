"""The exit status reports what the run did, not how much of it was printed.

``design`` has three outcomes and two of them are non-zero, which is the whole
reason there are two: 1 means the run worked and refused everything, 2 means
the run could not be completed. A wrapper script that cannot tell those apart
is the only consumer this distinction exists for -- and it was written down
nowhere a wrapper author would look, only in a source comment.

It was also wrong. The status came from ``rows``, the list the terminal table
prints, which has already been through the distinct-site filter and ``--top``.
So ``--top 0`` exited 1 and printed "no guide cleared the gate" beside a CSV it
had just written containing 322 of them. A display flag decided a
machine-readable outcome, and the sentence next to it was false.
"""

import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _design(*args, sequence: str | None = None) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        fasta = work / "t.fa"
        if sequence is None:
            rng = random.Random(9)
            sequence = "".join(rng.choice("ACGU") for _ in range(400))
        fasta.write_text(f">t1\n{sequence}\n", encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-m", "cas13guide.cli", "design",
             "--fasta", str(fasta), "--task", "knockdown",
             "--assay", "cell_culture_knockdown", "--effector", "RfxCas13d",
             "--no-structure", "--no-signal",
             "--out", str(work / "out"), "--stem", "r", *args],
            capture_output=True, text=True,
        )


class TestTheThreeOutcomes(unittest.TestCase):
    def test_zero_when_guides_are_ranked(self):
        result = _design()
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])

    def test_one_when_nothing_survives_the_gate(self):
        """A target too short to tile produces no candidates at all. This is a
        result, not an error, and it must not be confused with one."""
        result = _design(sequence="ACGUACGUAC")
        self.assertEqual(result.returncode, 1, result.stdout[-800:])
        self.assertIn("no guide cleared the gate", result.stdout)

    def test_two_when_the_run_cannot_be_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "cas13guide.cli", "design",
                 "--fasta", str(Path(directory) / "absent.fa"),
                 "--task", "knockdown", "--assay", "cell_culture_knockdown",
                 "--effector", "RfxCas13d"],
                capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2, result.stderr[-800:])
        self.assertNotIn("Traceback", result.stderr)


class TestADisplayFlagDoesNotDecideTheOutcome(unittest.TestCase):
    """The regression this file exists for."""

    def test_top_zero_still_exits_zero(self):
        full, none_shown = _design("--top", "10"), _design("--top", "0")
        self.assertEqual(full.returncode, 0, full.stderr[-800:])
        self.assertEqual(
            none_shown.returncode, 0,
            "a display limit reported the run as having found nothing",
        )

    def test_top_zero_does_not_claim_nothing_cleared_the_gate(self):
        result = _design("--top", "0")
        self.assertNotIn("no guide cleared the gate", result.stdout)
        self.assertIn("none displayed", result.stdout)

    def test_the_exports_are_written_whatever_the_display_limit(self):
        """The message promises "the exports carry them all"."""
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            rng = random.Random(9)
            fasta = work / "t.fa"
            fasta.write_text(
                ">t1\n" + "".join(rng.choice("ACGU") for _ in range(400)) + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-m", "cas13guide.cli", "design",
                 "--fasta", str(fasta), "--task", "knockdown",
                 "--assay", "cell_culture_knockdown", "--effector", "RfxCas13d",
                 "--no-structure", "--no-signal", "--top", "0",
                 "--out", str(work / "out"), "--stem", "r"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-800:])
            body = [
                line for line in (work / "out" / "r.csv").read_text(
                    encoding="utf-8").splitlines()
                if line and not line.startswith("#")
            ]
        self.assertGreater(len(body), 10, "the export was truncated by --top")


class TestTheContractIsDiscoverable(unittest.TestCase):
    def test_help_documents_every_code(self):
        """A contract only a source comment states is not a contract."""
        result = subprocess.run(
            [sys.executable, "-m", "cas13guide.cli", "design", "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("exit codes:", result.stdout)
        for code in ("0", "1", "2"):
            with self.subTest(code=code):
                self.assertRegex(result.stdout, rf"\n  {code}  \S")


if __name__ == "__main__":
    unittest.main()
