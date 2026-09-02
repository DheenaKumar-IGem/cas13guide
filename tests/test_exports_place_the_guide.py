"""An exported guide has to say where it binds.

Round three fixed the accessibility track for drawing guides from one record at
another record's coordinates, and the sentence it turned on was "``start``
alone cannot place a guide". ``source_target_id`` was added to the report dict
then -- and reached the track, and nothing else.

So the three surfaces a user actually keeps said which guide, what it scored
and how it was judged, and never which target it was for. On a multi-record
design that is not a cosmetic gap: two guides at the same coordinate can be on
different records, and recovering the answer from a CSV means searching each
spacer back against the input FASTA by hand. The FASTA export's own docstring
claims it is "useful on its own rather than only as a companion to the CSV",
which it was not.

Found by asking which fields the report produces that no export carries.
"""

import csv
import io
import random
import re
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def _multi_record_report():
    """Two records sharing an 80 nt stretch, so both are ranked."""
    from cas13guide.services.workflow import DesignRequest, run_request

    rng = random.Random(9)
    primary = "".join(rng.choice("ACGU") for _ in range(300))
    second = (
        "".join(rng.choice("ACGU") for _ in range(60))
        + primary[100:180]
        + "".join(rng.choice("ACGU") for _ in range(60))
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "t.fa"
        path.write_text(f">t1\n{primary}\n>t2\n{second}\n", encoding="utf-8")
        return run_request(
            DesignRequest(fasta_path=str(path), task="knockdown",
                          assay="cell_culture_knockdown", effector="RfxCas13d",
                          use_structure=False, use_signal=False)
        ).report


class TestTheCsvPlacesEveryGuide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cas13guide.services.fasta import ranked_to_csv

        cls.report = _multi_record_report()
        body = "\n".join(
            line for line in ranked_to_csv(cls.report).splitlines()
            if not line.startswith("#")
        )
        cls.rows = list(csv.DictReader(io.StringIO(body)))

    def test_both_columns_exist_and_came_after_the_established_ones(self):
        from cas13guide.services.fasta import RANKED_COLUMNS

        for column in ("source_target_id", "start"):
            with self.subTest(column=column):
                self.assertIn(column, RANKED_COLUMNS)
                self.assertGreater(
                    RANKED_COLUMNS.index(column),
                    RANKED_COLUMNS.index("distinct_site"),
                    "appended, never inserted",
                )

    def test_the_run_ranks_both_records_so_this_is_a_test(self):
        self.assertEqual(
            {row["source_target_id"] for row in self.rows}, {"t1", "t2"},
            "the fixture must rank guides from both records or it proves nothing",
        )

    def test_every_row_carries_a_target_and_a_coordinate(self):
        self.assertTrue(self.rows)
        for row in self.rows:
            with self.subTest(guide=row["guide_id"]):
                self.assertTrue(row["source_target_id"])
                self.assertTrue(row["start"])
                self.assertGreaterEqual(int(row["start"]), 0)


class TestTheFastaDeflinePlacesEveryGuide(unittest.TestCase):
    def test_the_defline_carries_the_target_and_the_start(self):
        from cas13guide.services.fasta import ranked_to_fasta

        text = ranked_to_fasta(_multi_record_report(), limit=5)
        deflines = [line for line in text.splitlines() if line.startswith(">")]
        self.assertTrue(deflines)
        for defline in deflines:
            with self.subTest(defline=defline[:40]):
                self.assertRegex(defline, r"\btarget=\S+")
                self.assertRegex(defline, r"\bstart=\d+")

    def test_a_report_without_the_fields_still_exports(self):
        """An older report dict has neither key, and an export that omits a
        field is fine where one that raises is not."""
        from cas13guide.services.fasta import ranked_to_fasta

        text = ranked_to_fasta({"ranked": [
            {"guide_id": "g1", "spacer": "ACGU", "rank": 1, "score": 1.0,
             "verdict": "PASS"},
        ]})
        self.assertIn(">g1", text)
        self.assertNotIn("target=", text)
        self.assertNotIn("start=", text)


class TestTheReportTablePlacesEveryGuide(unittest.TestCase):
    def test_the_site_column_shows_the_record_and_the_coordinate(self):
        from cas13guide.services.report_html import render_report

        html = render_report(_multi_record_report())
        self.assertIn(">Site<", html)
        self.assertRegex(html, r"<td[^>]*>t\d+:\d+</td>")

    def test_the_column_sorts_on_the_coordinate_not_the_rendered_text(self):
        """"t1:1000" sorts before "t1:99" as text, which is the wrong order for
        the one thing anybody sorts this column for."""
        from cas13guide.services.report_html import render_report

        html = render_report({"verdict_counts": {}, "ranked": [
            {"guide_id": "g1", "spacer": "ACGU", "rank": 1, "verdict": "PASS",
             "source_target_id": "t1", "start": 1000},
            {"guide_id": "g2", "spacer": "ACGU", "rank": 2, "verdict": "PASS",
             "source_target_id": "t1", "start": 99},
        ]})
        sorts = re.findall(r'<td data-sort="([^"]*)">t1:(\d+)</td>', html)
        self.assertEqual(len(sorts), 2)
        for sort_key, shown in sorts:
            with self.subTest(shown=shown):
                self.assertEqual(sort_key, shown, "sorted on the label, not the number")

    def test_every_column_still_has_its_glossary_entry(self):
        """The table and the glossary are declared separately, so a column
        added to one and not the other renders a heading nobody can read."""
        from cas13guide.services.report_html import SCORE_GLOSSARY, TABLE_COLUMNS

        for column in TABLE_COLUMNS:
            self.assertIn(column, SCORE_GLOSSARY)

    def test_a_row_without_a_start_renders_an_empty_cell(self):
        from cas13guide.services.report_html import render_report

        html = render_report({"verdict_counts": {}, "ranked": [
            {"guide_id": "g1", "spacer": "ACGU", "rank": 1, "verdict": "PASS"},
        ]})
        self.assertIn(">Site<", html)
        self.assertNotIn("None:", html)


if __name__ == "__main__":
    unittest.main()
