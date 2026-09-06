"""The demo is the first thing anybody runs, and nothing ran it.

``cas13guide demo`` executes the whole pipeline and prints a transcript. The
suite invoked it exactly once, with ``--help``, alongside every other
subcommand -- so ``demo.py`` sat at 20% line coverage even with subprocess
tracing on, and the command could have raised on line one without a single test
noticing.

Worse, the README *quotes* that transcript. Two of the lines it quotes had
drifted from what the command prints: "the rows above" for "the rows below",
and a target named ``demo`` that is now ``t_tgt``. A documented transcript that
nothing checks is a claim with no procedure behind it, which is the one thing
this package says it does not ship.

So this runs the demo for real. It reads the transcript the demo prints,
which nothing else does; ``test_the_demo_demonstrates_something.py`` reads the
report it writes.

The README pinning that used to live here has moved to
``test_module_entry_point.py``. The lines it quotes are a shortfall block, and
this file passed them for the wrong reason: the demo printed that block not by
arrangement but because its stand-in calibration could not certify anything, so
the PASS-only panel was always empty. The check now sits beside a run that
arranges an empty panel on purpose.
"""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

def _run_demo() -> subprocess.CompletedProcess:
    """Run in a scratch directory: the demo writes ``report.json`` to cwd."""
    with tempfile.TemporaryDirectory() as directory:
        return subprocess.run(
            [sys.executable, "-m", "cas13guide.cli", "demo"],
            cwd=directory, capture_output=True, text=True,
        )


class TestTheDemoRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _run_demo()

    def test_it_exits_cleanly(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[-2000:])
        self.assertNotIn("Traceback", self.result.stderr)

    def test_it_prints_the_sections_it_is_for(self):
        """A demo that runs but prints nothing useful passes an exit-code
        check and fails the reader.

        ``--- shortfalls ---`` used to be in this list and is not, because it
        prints conditionally: ``if report.shortfalls:``. It was here because
        the demo always fell short -- 0 PASS, so the PASS-only panel was empty,
        so a REQUIRED target had no cover -- and requiring the section is
        requiring the demo to fail. The message itself is guarded in
        ``test_module_entry_point.py``, against a run that arranges an empty
        panel on purpose.
        """
        for section in ("--- verdicts ---", "--- panel ---",
                        "--- validation plan ---"):
            with self.subTest(section=section):
                self.assertIn(section, self.result.stdout)

    def test_it_reports_a_verdict_for_every_candidate(self):
        """Against the count the run announces, which is what the name claims.

        This asserted the three counts summed above zero, which is true of
        ``0 / 81 / 27`` -- the demo's verdicts for as long as its stand-in
        calibration could not certify anything. An assertion that holds however
        badly the thing under it went is not measuring that thing.
        """
        counts = re.search(
            r"\{'PASS': (\d+), 'REVIEW': (\d+), 'FAIL': (\d+)\}",
            self.result.stdout)
        self.assertIsNotNone(counts, "no verdict counts in the demo output")
        announced = re.search(r"A: (\d+) candidates", self.result.stdout)
        self.assertIsNotNone(announced, "the run announces no candidate count")
        self.assertGreater(int(announced.group(1)), 0, "nothing was tiled")
        self.assertEqual(
            sum(int(group) for group in counts.groups()),
            int(announced.group(1)),
            "a candidate was tiled and never given a verdict",
        )


if __name__ == "__main__":
    unittest.main()
