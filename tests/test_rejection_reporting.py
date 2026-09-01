"""A refusal has to say why.

``GateResult.detail`` carries a numeric explanation for every verdict the gate
reaches -- "QC flags: gc_out_of_range", "forbidden hit UCB(activity) 0.412 >
0.200", "unsearched universes required by assay: ...". It is composed in eight
places in ``f_decide``.

Nothing read it. A grep for a read of ``.detail`` across ``src/`` returned
nothing at all: the strings were built and dropped on the floor. The visible
consequence was a run reporting ``FAIL: 41`` with no way to learn what was
wrong with any of the 41 -- ``ranked`` holds no FAIL rows by design, and that
was the only place a reason could have surfaced.

The fix carries the detail out on the ranked rows and groups the failures by
reason. These tests pin both, and pin the grouping's bounds and order, because
an explanation that changes shape between runs is not much of an explanation.
"""

import random
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def _run_with_failures():
    """A design that fails a meaningful number of guides on QC."""
    from cas13guide.services.workflow import DesignRequest, run_request

    rng = random.Random(7)
    sequence = "".join(rng.choice("ACGU") for _ in range(240))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "t.fa"
        path.write_text(f">t1\n{sequence}\n>t2\n{sequence[60:180]}\n", encoding="utf-8")
        return run_request(
            DesignRequest(fasta_path=str(path), task="knockdown",
                          assay="cell_culture_knockdown", effector="RfxCas13d",
                          use_structure=False, use_signal=False)
        ).report


class TestTheFailedGuidesAreAccountedFor(unittest.TestCase):
    def setUp(self):
        self.report = _run_with_failures()
        self.failed = self.report["verdict_counts"]["FAIL"]
        if not self.failed:
            self.skipTest("fixture produced no failures, so it checks nothing")

    def test_every_failed_guide_is_in_a_group(self):
        """The counts have to add up, or the section is worse than nothing."""
        total = sum(group["count"] for group in self.report["rejected"])
        self.assertEqual(total, self.failed)

    def test_no_failed_guide_is_in_the_ranking(self):
        """The other half of the same promise: explained, not recommended."""
        self.assertTrue(
            all(row["verdict"] != "FAIL" for row in self.report["ranked"])
        )

    def test_each_group_says_what_the_gate_measured(self):
        for group in self.report["rejected"]:
            with self.subTest(reasons=group["reasons"]):
                self.assertTrue(group["reasons"], "a group with no reason explains nothing")
                self.assertTrue(group["examples"])
                for example in group["examples"]:
                    self.assertTrue(
                        example["detail"],
                        f"{example['guide_id']} failed with no detail: the gate "
                        f"reached a verdict it could not describe",
                    )

    def test_the_examples_are_bounded(self):
        """Grouping on the detail strings would put nearly every guide in a
        group of one, since they carry per-guide numbers -- reproducing the
        wall of rows this replaces."""
        from cas13guide.pipeline.g_output import REJECTION_EXAMPLES

        for group in self.report["rejected"]:
            self.assertLessEqual(len(group["examples"]), REJECTION_EXAMPLES)

    def test_the_order_is_stable_and_puts_the_dominant_cause_first(self):
        rows = self.report["rejected"]
        self.assertEqual(
            rows, sorted(rows, key=lambda row: (-row["count"], row["reasons"]))
        )

    def test_the_ranked_rows_carry_the_gates_numbers(self):
        """REVIEW is the common verdict, and "needs a human" is not actionable
        without the quantity that triggered it."""
        rows = self.report["ranked"]
        self.assertTrue(rows)
        for row in rows:
            self.assertIn("detail", row)
        self.assertTrue(
            any(row["detail"] for row in rows),
            "no ranked guide carried any detail, so the field proves nothing",
        )


class TestTheGroupingItself(unittest.TestCase):
    """Unit-level, so the shape is pinned without paying for a full run."""

    def _assessment(self, guide_id, verdict, reasons, detail):
        from cas13guide.models.evidence import GateResult, GuideAssessment, MetricVector

        return GuideAssessment(
            guide_id=guide_id,
            candidate=make_candidate(guide_id),
            qc=clean_qc(guide_id),
            metrics=MetricVector(0.5, 0.0, 0.0, 0.0, 0.5, 1.0),
            gate=GateResult(verdict, tuple(reasons), detail=tuple(detail)),
        )

    def test_groups_are_ordered_by_count_then_reason(self):
        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.pipeline.g_output import _rejection_rows

        one, many = ReasonCode.CREDIBLE_FORBIDDEN_HIT, ReasonCode.DETERMINISTIC_INCOMPATIBLE
        rows = _rejection_rows([
            self._assessment("b", Verdict.FAIL, [one], ["off-allele 0.4 > 0.2"]),
            *[self._assessment(f"a{i}", Verdict.FAIL, [many], ["QC flags: gc_out_of_range"])
              for i in range(4)],
        ])
        self.assertEqual([row["count"] for row in rows], [4, 1])
        self.assertEqual(rows[0]["reasons"], [many.value])

    def test_a_run_with_no_failures_reports_no_groups(self):
        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.pipeline.g_output import _rejection_rows

        rows = _rejection_rows([
            self._assessment("a", Verdict.PASS, [ReasonCode.CONSTRAINTS_SATISFIED], ["ok"]),
            self._assessment("b", Verdict.REVIEW, [ReasonCode.EXPRESSION_UNKNOWN], ["1 locus"]),
        ])
        self.assertEqual(rows, ())

    def test_examples_are_the_same_guides_whatever_order_they_arrive_in(self):
        """Chosen by guide id rather than by position, so two runs that build
        the assessment list differently still print the same examples."""
        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.pipeline.g_output import _rejection_rows

        pool = [
            self._assessment(f"g{i}", Verdict.FAIL,
                             [ReasonCode.DETERMINISTIC_INCOMPATIBLE], [f"flag {i}"])
            for i in range(6)
        ]
        forward = _rejection_rows(list(pool))
        backward = _rejection_rows(list(reversed(pool)))
        self.assertEqual(forward, backward)


class TestTheReportRendersIt(unittest.TestCase):
    def test_the_section_appears_and_the_tooltip_points_at_it(self):
        from cas13guide.services.report_html import render_report

        html = render_report({
            "verdict_counts": {"PASS": 0, "REVIEW": 1, "FAIL": 2},
            "ranked": [],
            "rejected": [{
                "reasons": ["deterministic_incompatible"],
                "count": 2,
                "examples": [{"guide_id": "g1", "detail": ["QC flags: gc_out_of_range"]}],
            }],
        })
        self.assertIn("<h2>Why guides were rejected</h2>", html)
        self.assertIn("gc_out_of_range", html)

    def test_a_report_without_the_section_still_renders(self):
        """Older report dicts have no ``rejected`` key at all."""
        from cas13guide.services.report_html import render_report

        html = render_report({"verdict_counts": {}, "ranked": []})
        self.assertNotIn("<h2>Why guides were rejected</h2>", html)

    def test_the_detail_is_escaped(self):
        from cas13guide.services.report_html import render_report

        html = render_report({
            "verdict_counts": {}, "ranked": [],
            "rejected": [{
                "reasons": ["<b>x</b>"], "count": 1,
                "examples": [{"guide_id": "<img>", "detail": ["a > b & c"]}],
            }],
        })
        self.assertNotIn("<b>x</b>", html)
        self.assertNotIn("<img>", html)
        self.assertIn("a &gt; b &amp; c", html)


if __name__ == "__main__":
    unittest.main()
