"""QC measures three things about every spacer and reported none of them.

Found by asking, across every dataclass in the package, which fields are
constructed and then read by nothing -- the shape ``GateResult.detail`` had.
Of ``QCResult``'s fields only ``passed``, ``flags``, ``warnings``,
``manufacturability`` and ``pfs_preference`` were consumed anywhere. The three
measured numbers were not:

* ``gc`` -- the first number anybody ordering an oligo writes down, and the
  gate FAILs outside [0.30, 0.70], so where a *surviving* guide sits inside
  that band is exactly the thing a reader wants and could not get;
* ``max_homopolymer`` -- what makes a spacer hard to synthesise;
* ``self_fold_score`` -- the self-structure proxy.

The third is exported as ``self_fold_stem`` rather than under its field name,
because it is not always what gated the guide: with ViennaRNA installed the
SPACER_SELF_FOLDING flag comes from a real fold energy and this number stays
the proxy's stem length. Naming it for what it measures keeps a reader from
reading it as the reason.
"""

import csv
import io
import random
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

FIELDS = ("gc", "max_homopolymer", "self_fold_stem")


def _report():
    from cas13guide.services.workflow import DesignRequest, run_request

    rng = random.Random(9)
    sequence = "".join(rng.choice("ACGU") for _ in range(300))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "t.fa"
        path.write_text(f">t1\n{sequence}\n", encoding="utf-8")
        return run_request(
            DesignRequest(fasta_path=str(path), task="knockdown",
                          assay="cell_culture_knockdown", effector="RfxCas13d",
                          use_structure=False, use_signal=False)
        ).report


class TestTheReportCarriesWhatQcMeasured(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _report()
        cls.rows = cls.report["ranked"]

    def test_every_ranked_row_carries_all_three(self):
        self.assertTrue(self.rows)
        for row in self.rows:
            for field in FIELDS:
                with self.subTest(guide=row["guide_id"], field=field):
                    self.assertIn(field, row)
                    self.assertIsNotNone(row[field])

    def test_the_gc_reported_is_the_gc_of_the_spacer_shown(self):
        """Not merely present: the number has to belong to the sequence beside
        it, which is the only thing that makes it worth exporting.

        Checked over the whole ranking rather than the top of it. The composite
        score penalises |gc - 0.5|, so the highest-ranked guides are all at
        exactly 0.5 -- and a first draft that sampled the top 25 was satisfied
        by hard-coding 0.5, which is precisely the defect it was written for.
        """
        from cas13guide.util.seq import gc_fraction

        for row in self.rows:
            with self.subTest(guide=row["guide_id"]):
                self.assertAlmostEqual(row["gc"], gc_fraction(row["spacer"]), places=3)

    def test_the_ranking_actually_contains_more_than_one_gc(self):
        """Otherwise the test above cannot distinguish a real value from a
        constant, whatever it samples."""
        self.assertGreater(len({row["gc"] for row in self.rows}), 3)

    def test_the_homopolymer_reported_is_the_run_in_the_spacer_shown(self):
        from cas13guide.util.seq import max_homopolymer

        for row in self.rows:
            with self.subTest(guide=row["guide_id"]):
                self.assertEqual(row["max_homopolymer"], max_homopolymer(row["spacer"]))

    def test_every_surviving_guide_is_inside_the_gc_band(self):
        """The gate FAILs outside it, so this is a cross-check between the
        number now exported and the rule that let the guide through."""
        from cas13guide.pipeline.b_qc import GC_MAX, GC_MIN

        for row in self.rows:
            with self.subTest(guide=row["guide_id"]):
                self.assertGreaterEqual(row["gc"], GC_MIN)
                self.assertLessEqual(row["gc"], GC_MAX)


class TestTheExportsCarryThemToo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = _report()

    def test_all_three_are_csv_columns_after_the_established_ones(self):
        from cas13guide.services.fasta import RANKED_COLUMNS

        for field in FIELDS:
            with self.subTest(field=field):
                self.assertIn(field, RANKED_COLUMNS)
                self.assertGreater(
                    RANKED_COLUMNS.index(field),
                    RANKED_COLUMNS.index("distinct_site"),
                    "appended, never inserted",
                )

    def test_the_csv_rows_carry_the_values(self):
        from cas13guide.services.fasta import ranked_to_csv

        body = "\n".join(
            line for line in ranked_to_csv(self.report).splitlines()
            if not line.startswith("#")
        )
        rows = list(csv.DictReader(io.StringIO(body)))
        self.assertTrue(rows)
        for row in rows[:25]:
            for field in FIELDS:
                with self.subTest(field=field):
                    self.assertTrue(row[field], f"{field} exported empty")

    def test_the_defline_carries_the_gc(self):
        """The FASTA claims to carry "the numbers a wet-lab notebook needs
        beside the sequence", and GC is the first of them."""
        from cas13guide.services.fasta import ranked_to_fasta

        deflines = [
            line for line in ranked_to_fasta(self.report, limit=5).splitlines()
            if line.startswith(">")
        ]
        self.assertTrue(deflines)
        for defline in deflines:
            with self.subTest(defline=defline[:40]):
                self.assertRegex(defline, r"\bgc=0?\.\d+")

    def test_a_report_without_the_fields_still_exports(self):
        from cas13guide.services.fasta import ranked_to_fasta

        text = ranked_to_fasta({"ranked": [
            {"guide_id": "g1", "spacer": "ACGU", "rank": 1, "score": 1.0,
             "verdict": "PASS"},
        ]})
        self.assertIn(">g1", text)
        self.assertNotIn("gc=", text)


if __name__ == "__main__":
    unittest.main()
