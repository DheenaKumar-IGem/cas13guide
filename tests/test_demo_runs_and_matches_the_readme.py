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

So this runs the demo for real and pins the README against it, in that
direction: the README is checked against the behaviour, never the reverse.
"""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def _run_demo() -> subprocess.CompletedProcess:
    """Run in a scratch directory: the demo writes ``report.json`` to cwd."""
    with tempfile.TemporaryDirectory() as directory:
        return subprocess.run(
            [sys.executable, "-m", "cas13guide.cli", "demo"],
            cwd=directory, capture_output=True, text=True,
        )


def _squash(text: str) -> str:
    """Collapse whitespace, so a line wrapped for the README still matches."""
    return re.sub(r"\s+", " ", text).strip()


class TestTheDemoRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = _run_demo()

    def test_it_exits_cleanly(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[-2000:])
        self.assertNotIn("Traceback", self.result.stderr)

    def test_it_prints_the_sections_it_is_for(self):
        """A demo that runs but prints nothing useful passes an exit-code
        check and fails the reader."""
        for section in ("--- verdicts ---", "--- panel ---",
                        "--- validation plan ---", "--- shortfalls ---"):
            with self.subTest(section=section):
                self.assertIn(section, self.result.stdout)

    def test_it_reports_a_verdict_for_every_candidate(self):
        match = re.search(r"\{'PASS': (\d+), 'REVIEW': (\d+), 'FAIL': (\d+)\}",
                          self.result.stdout)
        self.assertIsNotNone(match, "no verdict counts in the demo output")
        self.assertGreater(sum(int(g) for g in match.groups()), 0,
                           "the demo assessed no guides at all")


class TestTheReadmeQuotesWhatTheDemoPrints(unittest.TestCase):
    """The README shows the shortfall block verbatim. Pin it to reality.

    Checked in one direction only: the README is asserted against the command's
    output, never the command against the README. A transcript is a record of
    what happened, so behaviour is the thing that gets to be right.
    """

    @classmethod
    def setUpClass(cls):
        cls.output = _squash(_run_demo().stdout)
        cls.readme = README.read_text(encoding="utf-8")

    def _quoted_shortfalls(self) -> list[str]:
        """Every ``!`` line inside a fenced block in the README."""
        # The language tag matters: without it the closing fence of one block
        # pairs with the opening of the next, and every block is misaligned.
        blocks = re.findall(r"```[a-z]*\n(.*?)```", self.readme, re.DOTALL)
        quoted = []
        for block in blocks:
            # Continuation lines are indented under their "!" line.
            current: list[str] = []
            for line in block.splitlines():
                if line.lstrip().startswith("!"):
                    if current:
                        quoted.append(" ".join(current))
                    current = [line.strip().lstrip("!").strip()]
                elif current and line.startswith("    "):
                    current.append(line.strip())
                elif current:
                    quoted.append(" ".join(current))
                    current = []
            if current:
                quoted.append(" ".join(current))
        return quoted

    def test_the_readme_actually_quotes_some(self):
        """Otherwise the test below passes over an empty list."""
        self.assertGreaterEqual(len(self._quoted_shortfalls()), 2)

    def test_every_quoted_shortfall_is_one_the_demo_prints(self):
        for quoted in self._quoted_shortfalls():
            with self.subTest(quoted=quoted[:60]):
                self.assertIn(
                    _squash(quoted), self.output,
                    "the README quotes a line the demo does not print",
                )


if __name__ == "__main__":
    unittest.main()
