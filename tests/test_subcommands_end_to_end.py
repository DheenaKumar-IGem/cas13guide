"""``check`` and ``train``, driven as commands rather than as functions.

Coverage measurement put ``cli.py`` at 63% once subprocess tracing was turned on
(the 7% it reports without that is an artefact -- several tests drive the CLI
through ``subprocess``, and coverage does not follow a child process by
default). What is left uncovered is almost entirely these two subcommands:
``_check`` and ``_train`` are exercised by nothing in the suite.

That is the gap worth closing rather than the number. The sixth review pass
found three real defects in ``train`` -- a traceback on a directory, a traceback
on a UTF-16 screen, and a mistyped column blamed on the wrong column -- and
found them precisely because nobody had ever run it. Those are pinned at the
``load_screen`` level; this pins the commands, which is where a user meets them,
and where an exit code means something.

Driven out of process on purpose: an exit code and a stream are the contract
these commands have with a shell script, and calling ``main()`` directly tests
neither.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "cas13guide", *args],
        capture_output=True, text=True, timeout=900, cwd=str(cwd or REPO),
    )


def screen_csv(path: Path, rows: int = 360, groups: int = 12) -> Path:
    """A screen with enough transcripts and spread to be trainable.

    At least 300 rows, because the acceptance gate refuses a smaller screen
    before it reaches the criteria this exercises. Spacers vary so the model
    has something to learn, and activity tracks a
    weak function of GC content so the acceptance gate sees real signal rather
    than noise -- the point is to exercise the command, not to smuggle a model
    past its own gate.
    """
    import random

    rng = random.Random(5)
    lines = ["spacer,activity,transcript"]
    for index in range(rows):
        spacer = "".join(rng.choice("ACGU") for _ in range(22))
        gc = (spacer.count("G") + spacer.count("C")) / len(spacer)
        activity = min(1.0, max(0.0, gc + rng.gauss(0, 0.08)))
        lines.append(f"{spacer},{activity:.4f},T{index % groups:02d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestTheCheckCommand(unittest.TestCase):
    """The CI gate. Its exit code is the whole point of it."""

    def test_it_passes_and_says_what_it_gated(self):
        result = run("check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK:", result.stdout)
        # The narrowed claim from the eighth pass: it gates coverage, and
        # reports skill without gating it.
        self.assertIn("conformal coverage", result.stdout)
        self.assertIn("skill is reported, not gated", result.stdout)

    def test_it_prints_the_numbers_it_measured(self):
        result = run("check")
        for field in ("spearman", "mae", "coverage"):
            with self.subTest(field=field):
                self.assertIn(field, result.stdout)

    def test_the_seed_is_honoured(self):
        """Same seed, same numbers -- across processes, so hash randomisation
        is in play."""
        first, second = run("check", "--seed", "3"), run("check", "--seed", "3")
        self.assertEqual(first.stdout, second.stdout)

    def test_a_different_seed_moves_the_measurement(self):
        """Otherwise the flag is accepted and ignored, which is its own bug."""
        self.assertNotEqual(run("check", "--seed", "1").stdout,
                            run("check", "--seed", "2").stdout)

    def test_no_traceback_reaches_the_user(self):
        self.assertNotIn("Traceback", run("check").stderr)


class TestTheTrainCommand(unittest.TestCase):
    """Ingest a screen, run the acceptance gate, report."""

    def test_a_good_screen_is_read_and_summarised(self):
        with tempfile.TemporaryDirectory() as directory:
            path = screen_csv(Path(directory) / "screen.csv")
            result = run("train", str(path))
        self.assertIn("read 360 rows", result.stdout)
        self.assertIn("12 transcript group(s)", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_it_reports_each_acceptance_criterion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = screen_csv(Path(directory) / "screen.csv")
            result = run("train", str(path))
        self.assertIn("conformal coverage", result.stdout)
        self.assertTrue(
            "[PASS]" in result.stdout or "[FAIL]" in result.stdout, result.stdout
        )

    def test_every_printed_criterion_agrees_with_its_own_numbers(self):
        """The eighth pass found a criterion printing PASS beside a value below
        the threshold on the same line. Checked here at the command, not at the
        function, because the line is what a user reads."""
        import re

        with tempfile.TemporaryDirectory() as directory:
            path = screen_csv(Path(directory) / "screen.csv")
            result = run("train", str(path))
        pattern = re.compile(r"\[(PASS|FAIL)\]\s+(.+?)\s+([-\d.]+) \(need ([-\d.]+)\)")
        rows = pattern.findall(result.stdout)
        self.assertTrue(rows, result.stdout)
        for mark, name, measured, threshold in rows:
            with self.subTest(criterion=name.strip()):
                self.assertEqual(mark == "PASS", float(measured) >= float(threshold))

    def test_a_missing_file_is_a_located_message(self):
        result = run("train", "no_such_screen.csv")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("cannot read dataset", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_directory_is_a_located_message(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run("train", directory)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("cannot read dataset", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_screen_with_no_grouping_column_is_refused_with_the_reason(self):
        """The one failure this module's docstring calls fatal rather than a
        warning."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nogroup.csv"
            path.write_text(
                "spacer,activity\nGCAUUGCUGUGGAGCUGAAGUC,0.5\n", encoding="utf-8"
            )
            result = run("train", str(path))
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("transcript", result.stdout.lower())
        self.assertNotIn("Traceback", result.stderr)

    def test_a_mistyped_column_names_the_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = screen_csv(Path(directory) / "screen.csv")
            result = run("train", str(path), "--spacer-column", "guide")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("guide", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_the_direction_flag_changes_the_outcome(self):
        """``--lower-is-better`` inverts the ranking; on a screen with real
        signal that cannot leave the report identical, or the flag is accepted
        and ignored."""
        with tempfile.TemporaryDirectory() as directory:
            path = screen_csv(Path(directory) / "screen.csv")
            normal = run("train", str(path))
            flipped = run("train", str(path), "--lower-is-better")
        self.assertNotEqual(normal.stdout, flipped.stdout)


class TestTheCommandSurfaceIsConsistent(unittest.TestCase):
    """Every subcommand answers ``--help`` and none of them traceback."""

    SUBCOMMANDS = ("demo", "check", "train", "design", "serve")

    def test_every_subcommand_has_help(self):
        for name in self.SUBCOMMANDS:
            with self.subTest(subcommand=name):
                result = run(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_every_subcommand_is_listed_in_the_top_level_help(self):
        listing = run("--help").stdout
        for name in self.SUBCOMMANDS:
            with self.subTest(subcommand=name):
                self.assertIn(name, listing)

    def test_an_unknown_subcommand_is_argparses_error_not_a_traceback(self):
        result = run("nonesuch")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
