"""The ranking table drops a column it cannot fill, rather than saying `None`.

Found by reading a run's output as a user rather than by any instrument.
`cas13guide design --no-structure --no-signal` printed access, dG/nt and signal
as `None` on every row -- in a tool whose whole manner is accounting for what
it does not know, the one absence it did not explain.

Every one of them *is* explained, in the limitations printed above the table:
"Collateral (trans) cleavage is not predicted for this assay",
"No structural accessibility was computed". So the word carried nothing the
reader did not already have, and the `robust` column two positions along had
always behaved correctly, collapsing to nothing when `--variants` was absent.

Driven out of process, because this is about what reaches a terminal.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def run_module(*args, timeout=300):
    return subprocess.run(
        [sys.executable, "-m", "cas13guide", *args],
        capture_output=True, text=True, timeout=timeout,
    )


class RankingTableTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import random

        rng = random.Random(23)

        def seq(n):
            return "".join(rng.choice("ACGU") for _ in range(n))

        cls._directory = tempfile.TemporaryDirectory()
        home = Path(cls._directory.name)
        eol = chr(10)
        (home / "t.fa").write_text(">TGT1" + eol + seq(200) + eol,
                                   encoding="utf-8")
        rows = []
        for name, biotype in (("RN18S1", "rrna"), ("MT-CO1", "mito"),
                              ("LINC01", "ncrna"), ("PREM1", "pre_mrna")):
            rows += [">" + name + " biotype=" + biotype, seq(200)]
        (home / "bg.fa").write_text(eol.join(rows) + eol, encoding="utf-8")
        cls.home = home

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def table(self, *extra):
        result = run_module(
            "design", "--fasta", str(self.home / "t.fa"),
            "--background", str(self.home / "bg.fa"), *extra,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-400:])
        lines = result.stdout.splitlines()
        start = next(i for i, line in enumerate(lines) if "rank" in line
                     and "spacer" in line)
        return lines[start], lines[start + 1:start + 4]


class TestAColumnWithNothingInItIsNotPrinted(RankingTableTestCase):
    def test_no_row_ever_says_none(self):
        """The complaint, directly."""
        for flags in ((), ("--no-signal",), ("--no-structure", "--no-signal"),
                      ("--assay", "cell_culture_knockdown")):
            with self.subTest(flags=flags):
                header, rows = self.table(*flags)
                for row in rows:
                    self.assertNotIn("None", row)

    def test_with_everything_available_every_column_is_there(self):
        header, rows = self.table()
        for column in ("access", "dG/nt", "signal"):
            self.assertIn(column, header)
        self.assertTrue(rows)

    def test_a_knockdown_assay_drops_only_the_signal_column(self):
        """Collateral is a hazard there, not a readout, so the model is not
        asked -- but the fold still runs."""
        header, _rows = self.table("--assay", "cell_culture_knockdown")
        self.assertNotIn("signal", header)
        self.assertIn("access", header)
        self.assertIn("dG/nt", header)

    def test_without_vienna_the_two_it_provides_go_together(self):
        header, _rows = self.table("--no-structure", "--no-signal")
        self.assertNotIn("access", header)
        self.assertNotIn("dG/nt", header)
        self.assertNotIn("signal", header)

    def test_the_columns_that_always_apply_stay(self):
        """Dropping the optional ones must not disturb the rest."""
        for flags in ((), ("--no-structure", "--no-signal")):
            with self.subTest(flags=flags):
                header, rows = self.table(*flags)
                for column in ("rank", "spacer", "score", "PFS"):
                    self.assertIn(column, header)
                self.assertTrue(rows[0].split())


class TestTheRunSaysWhyAColumnIsMissing(RankingTableTestCase):
    """Dropping a column is only honest because the reason is printed above it.

    `--no-structure` removes the duplex free energy as well as accessibility --
    same ViennaRNA dependency, same run -- and the warning named only the
    first, which left `dG/nt` absent for a reason the run never gave.
    """

    def _stderr(self, *extra):
        result = run_module(
            "design", "--fasta", str(self.home / "t.fa"),
            "--background", str(self.home / "bg.fa"), *extra,
        )
        return result.stderr + result.stdout

    def test_the_no_structure_warning_names_both_things_it_removes(self):
        text = self._stderr("--no-structure")
        self.assertIn("No structural accessibility was computed", text)
        self.assertIn("duplex free energy", text)
        self.assertIn("duplex_dg_per_nt", text)

    def test_a_knockdown_run_says_collateral_is_not_predicted(self):
        text = self._stderr("--assay", "cell_culture_knockdown")
        self.assertIn("Collateral (trans) cleavage is not predicted", text)


if __name__ == "__main__":
    unittest.main()
