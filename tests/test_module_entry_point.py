"""``python -m cas13guide`` has to work, because PATH often does not.

A console script that is installed but not on PATH is indistinguishable from a
broken install, and that is the common case for ``pip install --user`` on
Windows. This is checked by running a real subprocess rather than by importing
``__main__``: importing it proves the file parses, which was never in doubt --
the failure mode is that the interpreter cannot resolve the package as a module
at all, and only an out-of-process run sees that.
"""

import subprocess
import sys
import unittest

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
    one: with the built-in synthetic calibration every guide lands REVIEW, the
    panel is PASS-only and therefore empty, and the terminal printed a full
    table and not one word about it. The report and the export headers carried
    the shortfall the whole time; only the surface a person actually reads did
    not.

    Run out of process on purpose: the defect was in what reaches a terminal,
    and calling the printer directly would not have seen it.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = run_module(
            "design", "--sequence", SEQUENCE, "--top", "2", "--panel-size", "3"
        )

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
