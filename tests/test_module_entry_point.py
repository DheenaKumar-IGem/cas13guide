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
