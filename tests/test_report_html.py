"""The HTML report (FR8.2, FR8.4, NFR5, NFR6)."""

import html.parser
import re
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.services.report_html import (
    SCORE_GLOSSARY,
    TABLE_COLUMNS,
    AccessibilityTrack,
    render_report,
    write_report,
)

REPORT = {
    "spec_fingerprint": "abc123",
    "verdict_counts": {"PASS": 2, "REVIEW": 1, "FAIL": 4},
    "review_rate": 0.1429,
    "ranked": [
        {
            "rank": 1, "guide_id": "g_one", "spacer": "AUGCAUGCAUGCAUGCAUGCAU",
            "score": 7.5, "verdict": "PASS", "review_tier": None, "start": 10,
            "spans_junction": False, "pfs_3prime": "A", "pfs_preferred": True,
            "accessibility": 0.61, "duplex_dg_per_nt": -1.9, "predicted_signal": 0.14,
            "reasons": ["constraints_satisfied"], "qc_warnings": [],
            "metrics": {"required_coverage": 0.81, "worst_off_target": 0.0,
                        "cumulative_burden": 0.0, "robustness": 0.79},
        },
        {
            "rank": 2, "guide_id": "g_two", "spacer": "GCAUGCAUGCAUGCAUGCAUGC",
            "score": 5.1, "verdict": "REVIEW", "review_tier": "R2", "start": 40,
            "spans_junction": True, "pfs_3prime": "G", "pfs_preferred": False,
            "accessibility": 0.22, "duplex_dg_per_nt": -2.4, "predicted_signal": 0.05,
            "reasons": ["expression_unknown"], "qc_warnings": ["pfs_disfavoured"],
            "metrics": {"required_coverage": 0.55, "worst_off_target": 0.3,
                        "cumulative_burden": 0.4, "robustness": 0.51},
        },
    ],
    "controls": [
        {"guide_id": "ctrl_one", "spacer": "UACGUACGUACGUACGUACGUA",
         "matched_to": "g_one", "gc": 0.5, "verified_no_hits": True},
    ],
    "shortfalls": ["search escalation budget exhausted"],
    "validation_plan": ["Transfect the panel alongside controls."],
    "provenance": {"records": [{"kind": "model", "name": "activity",
                                "version": "model/heuristic-cas13/v1"}]},
}


class _Balanced(html.parser.HTMLParser):
    VOID = {"meta", "br", "input", "img", "hr", "link", "rect", "polyline", "text", "path"}

    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.mismatched: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            self.mismatched.append(tag)
            self.stack.remove(tag)


class TestRenderedPage(unittest.TestCase):
    def setUp(self):
        self.text = render_report(REPORT, title="T", subtitle="S")

    def test_the_markup_is_balanced(self):
        parser = _Balanced()
        parser.feed(self.text)
        self.assertEqual(parser.stack, [])
        self.assertEqual(parser.mismatched, [])

    def test_it_is_self_contained(self):
        """A report that fetches a script stops working offline or in 3 years."""
        self.assertEqual(re.findall(r'(?:src|href)\s*=\s*["\']https?://', self.text), [])
        self.assertNotIn("cdn.", self.text)
        self.assertIn("<style>", self.text)
        self.assertIn("<script>", self.text)

    def test_every_ranked_guide_has_a_row(self):
        self.assertEqual(self.text.count("<tr data-verdict="), len(REPORT["ranked"]))

    def test_rows_carry_their_verdict_for_filtering(self):
        self.assertIn('data-verdict="PASS"', self.text)
        self.assertIn('data-verdict="REVIEW"', self.text)

    def test_every_cell_carries_a_sort_key(self):
        """FR8.2 sorting is by data-sort, so a missing one silently mis-sorts."""
        rows = re.findall(r"<tr data-verdict=.*?</tr>", self.text, re.S)
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(row.count("<td"), len(TABLE_COLUMNS))
            self.assertEqual(row.count("data-sort="), len(TABLE_COLUMNS))

    def test_metrics_are_pulled_out_of_the_nested_dict(self):
        """Coverage and robustness live under `metrics`, not at the top level."""
        self.assertIn("0.81", self.text)     # required_coverage
        self.assertIn("0.79", self.text)     # robustness

    def test_missing_values_render_as_a_dash_not_none(self):
        self.assertNotIn(">None<", self.text)
        self.assertIn("—", self.text)

    def test_html_in_the_data_is_escaped(self):
        """Sequence names come from user files, so they are untrusted input."""
        hostile = {
            **REPORT,
            "ranked": [{**REPORT["ranked"][0],
                        "spacer": "<script>alert(1)</script>",
                        "pfs_3prime": "A&B<C"}],
            "controls": [{**REPORT["controls"][0],
                          "guide_id": "<img src=x onerror=alert(1)>"}],
            "shortfalls": ["<b>bold</b>"],
        }
        text = render_report(hostile)
        self.assertNotIn("<script>alert(1)</script>", text)
        self.assertNotIn("<img src=x", text)
        self.assertNotIn("<b>bold</b>", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("A&amp;B&lt;C", text)

    def test_a_hostile_guide_label_cannot_break_out_of_the_svg(self):
        track = AccessibilityTrack(
            name="x", profile=[0.5, 0.5, 0.5],
            guides=[(0, 2, "</title><script>alert(1)</script>")],
        )
        text = render_report(REPORT, tracks=[track])
        self.assertNotIn("<script>alert(1)</script>", text)
        self.assertIn("&lt;/title&gt;", text)


class TestScoreExplanations(unittest.TestCase):
    """NFR6 -- a reviewer should not have to read the source."""

    def test_every_displayed_column_is_explained(self):
        for column in TABLE_COLUMNS:
            self.assertIn(column, SCORE_GLOSSARY, f"{column} has no explanation")

    def test_the_glossary_is_rendered(self):
        text = render_report(REPORT)
        self.assertEqual(text.count("<dt>"), len(TABLE_COLUMNS))
        for column in TABLE_COLUMNS:
            self.assertIn(SCORE_GLOSSARY[column][0], text)

    def test_the_bound_asymmetry_is_explained_not_just_shown(self):
        coverage = SCORE_GLOSSARY["required_coverage"][1].lower()
        hazard = SCORE_GLOSSARY["worst_off_target"][1].lower()
        self.assertIn("lower", coverage)
        self.assertIn("upper", hazard)

    def test_the_signal_caveat_is_stated(self):
        self.assertIn("within this run", SCORE_GLOSSARY["predicted_signal"][1])

    def test_caller_notes_are_shown_prominently(self):
        text = render_report(REPORT, notes=["calibration is SYNTHETIC"])
        self.assertIn("calibration is SYNTHETIC", text)
        self.assertIn('class="note"', text)


class TestAccessibilityTrackVisual(unittest.TestCase):
    """FR8.4 -- a visual for the top-ranked candidates."""

    def _track(self):
        return AccessibilityTrack(
            name="target (60 nt)",
            profile=[i / 60 for i in range(60)],
            guides=[(10, 22, "#1 g_one"), (40, 22, "#2 g_two")],
        )

    def test_the_track_is_drawn_as_inline_svg(self):
        text = render_report(REPORT, tracks=[self._track()])
        self.assertIn('<svg class="track"', text)
        self.assertIn("<polyline", text)

    def test_guides_are_marked_and_labelled(self):
        text = render_report(REPORT, tracks=[self._track()])
        svg = re.search(r"<svg class=\"track\".*?</svg>", text, re.S).group(0)
        self.assertEqual(svg.count("<title>"), 2)   # page <title> is not in here
        self.assertIn("#1 g_one", svg)
        self.assertIn("#2 g_two", svg)

    def test_an_empty_profile_says_so_rather_than_drawing_nothing(self):
        text = render_report(
            REPORT, tracks=[AccessibilityTrack(name="x", profile=[])]
        )
        self.assertIn("no accessibility profile", text)

    def test_probabilities_outside_the_unit_interval_are_clamped(self):
        """A bad profile must not draw a line off the top of the chart."""
        track = AccessibilityTrack(name="x", profile=[-1.0, 0.5, 2.0])
        text = render_report(REPORT, tracks=[track])
        points = re.search(r'points="([^"]+)"', text).group(1)
        ys = [float(pair.split(",")[1]) for pair in points.split()]
        self.assertTrue(all(0 <= y <= 92 for y in ys), ys)


class TestEmptyAndDegenerateReports(unittest.TestCase):
    def test_no_candidates_explains_itself(self):
        text = render_report({"verdict_counts": {"PASS": 0, "REVIEW": 0, "FAIL": 9},
                              "ranked": [], "shortfalls": ["nothing cleared the gate"]})
        self.assertIn("No guide cleared the gate", text)
        self.assertIn("nothing cleared the gate", text)

    def test_a_bare_report_still_renders(self):
        parser = _Balanced()
        parser.feed(render_report({}))
        self.assertEqual(parser.stack, [])

    def test_write_report_creates_missing_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_report(REPORT, Path(directory) / "nested" / "r.html")
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)


class TestTheOffTargetTableSaysHowMuchOfItThereIs(unittest.TestCase):
    """"The 50 highest-hazard off-target loci found" reads as all of them.

    `_off_target_rows` caps the table at fifty and the caption reported
    `len(off_targets)`, so fifty of fifty and fifty of four thousand printed
    the same sentence -- and "found" names the whole set, not the shown part.
    The validation plan directly underneath tells a reader to sequence from
    this table, which makes how much of it they are looking at the first thing
    they need to know.
    """

    @staticmethod
    def _rows(n):
        return [
            {"guide_id": f"g{i}", "locus": f"L{i}", "gene": f"GENE{i}",
             "biotype": "coding", "target_class": "unannotated",
             "mismatches": 2, "wobbles": 0, "cigar": "MMM",
             "ucb_activity": 0.4, "ucb_expression": 0.5, "hazard_weight": 0.6,
             "hazard": 0.12, "expression_unknown": False, "ood": False,
             "context": "reference"}
            for i in range(n)
        ]

    def _render(self, shown, total):
        report = dict(REPORT)
        report["off_targets"] = self._rows(shown)
        report["off_target_total"] = total
        return render_report(report)

    def test_a_complete_table_says_so(self):
        page = self._render(shown=7, total=7)
        self.assertIn("All 7 off-target loci found", page)
        self.assertNotIn("highest-hazard of", page)

    def test_a_truncated_table_names_both_numbers(self):
        page = self._render(shown=50, total=273)
        self.assertIn("The 50 highest-hazard of 273 off-target loci found", page)
        self.assertIn("The other 223 are not shown", page)

    def test_it_does_not_send_the_reader_to_a_file_that_has_no_more(self):
        """The report dict carries the same capped rows, and report.json is
        not written unless asked for. Naming either would trade one false
        impression for another."""
        page = self._render(shown=50, total=273)
        self.assertNotIn("report.json", page)

    def test_it_says_the_hidden_rows_are_still_inside_the_metrics(self):
        """Otherwise "not shown anywhere" reads as "not counted"."""
        page = self._render(shown=50, total=273)
        self.assertIn("all 273 are already inside", page)
        self.assertIn("cumulative_burden", page)

    def test_a_report_without_the_total_still_renders(self):
        """Older reports, and any caller building the dict by hand."""
        report = dict(REPORT)
        report["off_targets"] = self._rows(4)
        report.pop("off_target_total", None)
        page = render_report(report)
        self.assertIn("All 4 off-target loci found", page)


class TestThePlanAndThePageAgreeAboutTheTable(unittest.TestCase):
    """The plan may only send a reader somewhere the page actually goes.

    `report_html` renders the explorer `if off_targets`, and the comment
    beside it says why the table was added at all: "A page that instructs a
    reader to consult a table it does not contain is worse than one that says
    nothing, because the reader goes looking." That fixed the case where rows
    existed and nothing rendered them. With no rows there is still no table,
    and the plan said "Sequence the top off-target loci from the explorer
    table" regardless.

    Driven through `build_report`, because the defect is the agreement between
    two functions and calling either alone cannot see it.
    """

    @staticmethod
    def _page(with_rows: bool):
        report = dict(REPORT)
        rows = [
            {"guide_id": "g_one", "locus": "L1", "gene": "GENE1",
             "biotype": "coding", "target_class": "unannotated",
             "mismatches": 2, "wobbles": 0, "cigar": "MMM",
             "ucb_activity": 0.4, "ucb_expression": 0.5, "hazard_weight": 0.6,
             "hazard": 0.12, "expression_unknown": False, "ood": False,
             "context": "reference"}
        ]
        report["off_targets"] = rows if with_rows else []
        report["off_target_total"] = len(report["off_targets"])
        report["validation_plan"] = [
            "Sequence the top off-target loci from the explorer table; "
            "prioritize rows with high UCB(activity) whose expression is "
            "currently unknown."
            if with_rows else
            "There is no off-target explorer on this page: no site was found "
            "in the sequences this run was given."
        ]
        return render_report(report)

    def test_with_rows_the_table_is_there_to_be_read(self):
        page = self._page(with_rows=True)
        self.assertIn("Off-target explorer", page)
        self.assertIn("Sequence the top off-target loci", page)

    def test_without_rows_nothing_sends_the_reader_looking(self):
        page = self._page(with_rows=False)
        self.assertNotIn("Off-target explorer", page)
        self.assertNotIn("Sequence the top off-target loci", page)


class TestAnEmptyExplorerIsNotACleanOne(unittest.TestCase):
    """What the plan says instead must not read as a specificity result.

    An empty off-target table is the shape of a good answer and the substance
    of no answer: a run given one sequence finds nothing because there was
    nothing loaded to find. That is the failure this whole package is built
    against, so the sentence that replaces the missing-table line has to name
    it rather than leave a blank.
    """

    def test_the_plan_says_why_the_table_is_missing(self):
        from cas13guide.pipeline.g_output import _validation_plan
        from cas13guide.pipeline.optimizer import Panel

        panel = Panel(guides=(), coverage={}, objective=0.0, guarantee="none")
        plan = _validation_plan(panel, (), make_spec(), off_target_rows=())
        # "off-target homology" appears in the collateral step too.
        [line] = [step for step in plan if "off-target explorer" in step]
        self.assertIn("no site was found", line)
        self.assertIn("about the corpus", line)
        self.assertIn("background", line)
        self.assertNotIn("explorer table;", line)


if __name__ == "__main__":
    unittest.main()
