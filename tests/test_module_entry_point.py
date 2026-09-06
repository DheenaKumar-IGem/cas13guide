"""``python -m cas13guide`` has to work, because PATH often does not.

A console script that is installed but not on PATH is indistinguishable from a
broken install, and that is the common case for ``pip install --user`` on
Windows. This is checked by running a real subprocess rather than by importing
``__main__``: importing it proves the file parses, which was never in doubt --
the failure mode is that the interpreter cannot resolve the package as a module
at all, and only an out-of-process run sees that.
"""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def _squash(text: str) -> str:
    """Collapse whitespace, so a line wrapped for the README still matches."""
    return re.sub(r"\s+", " ", text).strip()


def _readme_shortfall_quotes() -> list[str]:
    """Every ``!`` line inside a fenced block in the README."""
    text = README.read_text(encoding="utf-8")
    # The language tag matters: without it the closing fence of one block pairs
    # with the opening of the next, and every block is misaligned.
    quoted: list[str] = []
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, re.DOTALL):
        current: list[str] = []
        for line in block.splitlines():
            if line.lstrip().startswith("!"):
                if current:
                    quoted.append(" ".join(current))
                current = [line.strip().lstrip("!").strip()]
            elif current and line.startswith("    "):
                # Continuation lines are indented under their "!" line.
                current.append(line.strip())
            elif current:
                quoted.append(" ".join(current))
                current = []
        if current:
            quoted.append(" ".join(current))
    return quoted

#: A 400 nt target with room for real tiling. Fixed rather than random
#: so the verdict mix under the built-in models is the same every run.
SEQUENCE = (
    "CCGUAAUGCCUUUCCCUAACAGAGUUUUUCGAACUCGUGUUGUCGAGCGACGGAAUUAGAUCAGUUAAAUGGCAGAAAACUGGCAGGGCUUUUAGUCGUG"
    "GGAUGAUCAGUGGGUAAAGGUGGCGCGGGGUAACGCGCGCUAAGGCUCAGCUGCAACGCGGAGCUGGUGUGUUAUCCAUUCAUGGCAGACAACUAAUACG"
    "CAUAAGCGUAGCCAACCGCAUUAGCGUAUGAACAAAAUAAUGCGAGUUGGGCGUACAUACAGUUAUAGUGUUUACCGAUCUCAGGGAUAUAGAAUCCUAA"
    "AUCAGAAAUGGAACAAAGCACCCUUGGUGUAUCUCUUCUCCAUUUCCGCCGCGUGCGAGUUCCGCGUCUUCUAUAUAUCCACGCCGCCAGCAGCUAAAAG"
)


def run_module(*args):
    return subprocess.run(
        [sys.executable, "-m", "cas13guide", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestModuleEntryPoint(unittest.TestCase):
    def test_the_package_is_runnable_as_a_module(self):
        result = run_module("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cas13guide", result.stdout)

    def test_it_reaches_the_same_parser_as_the_console_script(self):
        """Same prog name and same subcommands -- one CLI, two invocations."""
        result = run_module("--help")
        for command in ("demo", "design", "serve"):
            self.assertIn(command, result.stdout)

    def test_a_subcommands_help_is_reachable_through_it(self):
        result = run_module("design", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--preset", result.stdout)

    def test_no_arguments_exits_nonzero_rather_than_traceback(self):
        """argparse's own error, not an unhandled exception."""
        result = run_module()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)


class TestTheTerminalSaysWhenThereIsNoRecommendation(unittest.TestCase):
    """A run that ranks rows but recommends nothing must say so.

    Shortfalls used to print only when *nothing* ranked -- the case where the
    reader already knows something went wrong. The dangerous case is the other
    one: rows ranked, the PASS-only panel empty, and the terminal printing a
    full table and not one word about it. The report and the export headers
    carried the shortfall the whole time; only the surface a person actually
    reads did not.

    The empty panel is now *arranged* rather than inherited. This docstring
    used to say every guide lands REVIEW "with the built-in synthetic
    calibration", and so did README's account of it, and both were wrong: the
    universal REVIEW came from EXPRESSION_UNKNOWN counting the run's own
    target, whose abundance no metric reads. Once that was restricted to
    off-target loci the same command returns 272 PASS under the same synthetic
    calibration, and a test resting on the old behaviour would have gone green
    only while the defect lasted.

    So the fixture supplies a background paralog: a full-length copy of the
    target with every seventh base substituted. Every window then has an
    off-target hit in a locus of unknown abundance -- weak enough to downgrade
    rather than fail -- so every survivor is REVIEW for the reason the code
    means by it, the panel is empty, and the terminal has to say so.

    Run out of process on purpose: the defect was in what reaches a terminal,
    and calling the printer directly would not have seen it.
    """

    @classmethod
    def setUpClass(cls):
        substitute = {"A": "C", "C": "A", "G": "U", "U": "G"}
        paralog = "".join(
            substitute[base] if index % 7 == 3 else base
            for index, base in enumerate(SEQUENCE)
        )
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        (root / "target.fa").write_text(f">t1\n{SEQUENCE}\n", encoding="utf-8")
        (root / "paralog.fa").write_text(f">paralog\n{paralog}\n", encoding="utf-8")
        cls.result = run_module(
            "design", "--fasta", str(root / "target.fa"),
            "--background", str(root / "paralog.fa"),
            "--top", "2", "--panel-size", "3",
        )

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_the_fixture_really_leaves_the_panel_empty(self):
        """Or every assertion below would pass on a run that recommended one."""
        self.assertIn("'PASS': 0", self.result.stdout)

    def test_the_run_succeeds_and_ranks_something(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr)
        self.assertIn("rank", self.result.stdout)

    def test_it_says_there_is_no_panel(self):
        self.assertIn("no panel", self.result.stdout)

    def test_it_says_the_ranked_rows_are_not_a_recommendation(self):
        """The distinction the message has to carry: triage, not an answer."""
        self.assertIn("not a recommendation", self.result.stdout)

    def test_the_shortfall_itself_reaches_the_terminal(self):
        self.assertIn("no feasible panel", self.result.stdout)

    def test_the_readme_quotes_this_run_and_not_some_other(self):
        """The README shows this block; pin it against the run that prints it.

        It used to be pinned against ``cas13guide demo``, and passed, because
        the demo printed the same shortfall -- not by arrangement, but because
        its own stand-in calibration could not certify anything and the
        PASS-only panel was therefore always empty. Fixing the demo left the
        quote with no producer. That is how a transcript ends up documenting a
        defect: it was checked, continuously, against the wrong run.

        Checked in one direction only. The README is asserted against the
        command's output, never the command against the README, because a
        transcript is a record of what happened.
        """
        quoted = _readme_shortfall_quotes()
        self.assertGreaterEqual(len(quoted), 2,
                                "the README quotes none, so this is vacuous")
        printed = _squash(self.result.stdout)
        for line in quoted:
            with self.subTest(quoted=line[:55]):
                self.assertIn(
                    _squash(line), printed,
                    "the README quotes a line this run does not print")
