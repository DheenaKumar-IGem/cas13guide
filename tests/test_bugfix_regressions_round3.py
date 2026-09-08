"""Third review pass. Both survivors were in the multi-record FASTA path.

Twenty-five defects survived the first pass, four the second, two this one --
and the two angles most likely to find something new, differential testing
across the four search indexes and property testing of the gate/Pareto/rank
order, found nothing that survived. That is the shape of convergence, and it is
worth recording alongside what was fixed.

Both survivors are the same oversight in two places: a run holds more than one
target record, and two consumers assumed it held one.
"""

import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def _two_record_fasta(directory: Path) -> tuple[Path, str, str]:
    """A primary and a second record sharing an 80 nt stretch.

    The shared stretch is what makes the second record a *useful* DESIRED
    target: guides tiled there score on both.
    """
    import random

    rng = random.Random(9)
    primary = "".join(rng.choice("ACGU") for _ in range(300))
    second = (
        "".join(rng.choice("ACGU") for _ in range(60))
        + primary[100:180]
        + "".join(rng.choice("ACGU") for _ in range(60))
    )
    path = directory / "two.fa"
    path.write_text(f">t1\n{primary}\n>t2\n{second}\n", encoding="utf-8")
    return path, primary, second


def _variants_fasta(directory: Path, primary: str) -> Path:
    variant = list(primary)
    variant[10] = "A" if variant[10] != "A" else "C"
    path = directory / "v.fa"
    path.write_text(">V1\n" + "".join(variant) + "\n", encoding="utf-8")
    return path


class TestTheAccessibilityTrackMarksOnlyItsOwnTarget(unittest.TestCase):
    """FR8.4 drew guides from other records at their own coordinates.

    ``outcome.profile`` is the primary record's, and the track is named and
    sized for it -- but the marks came from an unfiltered top-N of the ranking,
    and candidates are tiled from every REQUIRED *and DESIRED* target. So a
    guide from record 2 was drawn at its own offset along record 1's profile:
    silently shifted, or, when the offset ran past the end, silently clipped out
    of the picture altogether. Measured: all five default marks foreign, one of
    them 6 nt beyond a 150 nt track.
    """

    def test_the_ranked_row_says_which_target_its_start_belongs_to(self):
        """``start`` alone cannot place a guide, and the comment said it could."""
        from cas13guide.services.workflow import DesignRequest, run_request

        with tempfile.TemporaryDirectory() as directory:
            path, _primary, _second = _two_record_fasta(Path(directory))
            outcome = run_request(
                DesignRequest(fasta_path=str(path), task="knockdown",
                              assay="cell_culture_knockdown", effector="RfxCas13d",
                              use_structure=False, use_signal=False)
            )
        rows = outcome.report["ranked"]
        self.assertTrue(rows)
        for row in rows[:20]:
            self.assertIn("source_target_id", row)
        self.assertEqual(
            {row["source_target_id"] for row in rows}, {"t1", "t2"},
            "fixture must rank guides from both records to be a test",
        )

    def test_only_the_primary_record_s_guides_are_tracked(self):
        from cas13guide.services.report_html import track_rows

        ranked = [
            {"rank": 1, "guide_id": "gA", "start": 10, "source_target_id": "t2"},
            {"rank": 2, "guide_id": "gB", "start": 20, "source_target_id": "t1"},
            {"rank": 3, "guide_id": "gC", "start": 30, "source_target_id": "t2"},
            {"rank": 4, "guide_id": "gD", "start": 40, "source_target_id": "t1"},
        ]
        self.assertEqual(
            [row["guide_id"] for row in track_rows(ranked, "t1", 5)], ["gB", "gD"]
        )

    def test_the_limit_is_applied_after_the_filter(self):
        """Otherwise a page of foreign guides leaves the track empty."""
        from cas13guide.services.report_html import track_rows

        ranked = [
            {"guide_id": f"g{i}", "source_target_id": "t2"} for i in range(10)
        ] + [{"guide_id": "mine", "source_target_id": "t1"}]
        self.assertEqual(
            [row["guide_id"] for row in track_rows(ranked, "t1", 5)], ["mine"]
        )

    def test_a_row_without_the_field_is_kept(self):
        """An older report dict has no source_target_id; drawing nothing at all
        would be a worse answer than what was already being drawn."""
        from cas13guide.services.report_html import track_rows

        ranked = [{"guide_id": "old", "start": 5}]
        self.assertEqual([row["guide_id"] for row in track_rows(ranked, "t1", 5)], ["old"])

    def test_a_mark_past_the_end_stays_inside_the_viewbox(self):
        """x2 was clamped and x1 was not, so an out-of-range mark left the
        picture rather than being drawn wrongly -- the worse failure, because
        nothing shows it happened."""
        import re

        from cas13guide.services.report_html import AccessibilityTrack, render_report

        track = AccessibilityTrack(
            name="T1 (40 nt)",
            profile=tuple([0.5] * 40),
            guides=[(320, 22, "#1 g_far")],
        )
        html = render_report({"verdict_counts": {}, "ranked": []}, tracks=[track])
        viewbox = re.search(r'viewBox="0 0 (\d+) (\d+)"', html)
        self.assertIsNotNone(viewbox)
        width = float(viewbox.group(1))
        for x in re.findall(r'<rect x="([\d.]+)" y="\d+" width="[\d.]+" height', html):
            with self.subTest(x=x):
                self.assertLessEqual(float(x), width)


class TestVariantsKeepTheOtherTargetRecords(unittest.TestCase):
    """``--variants`` replaced the context list and dropped records 1+.

    A multi-record FASTA makes records 1+ DESIRED targets, and the contract
    still listed them -- but they existed in no context, so
    ``desired_utility`` fell to NO_EVIDENCE for every guide. It weighs 1.0 in
    DEFAULT_WEIGHTS, so the ranking moved too, and those loci stopped being
    searched at all. Nothing said so.
    """

    def _run(self, path, **kw):
        from cas13guide.services.workflow import DesignRequest, run_request

        return run_request(
            DesignRequest(fasta_path=str(path), task="knockdown",
                          assay="cell_culture_knockdown", effector="RfxCas13d",
                          use_structure=False, use_signal=False, **kw)
        )

    def test_desired_utility_survives_adding_a_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            path, primary, _second = _two_record_fasta(Path(directory))
            variants = _variants_fasta(Path(directory), primary)
            without = self._run(path)
            with_variants = self._run(path, variants_path=str(variants))

        def utilities(outcome):
            return [row["metrics"]["desired_utility"] for row in outcome.report["ranked"]]

        baseline = utilities(without)
        self.assertGreater(max(baseline), 0.0, "fixture must exercise desired_utility")
        self.assertGreater(
            max(utilities(with_variants)), 0.0,
            "adding --variants zeroed the DESIRED objective",
        )
        self.assertEqual(
            sum(1 for value in baseline if value > 0),
            sum(1 for value in utilities(with_variants) if value > 0),
        )

    def test_the_extra_target_is_a_target_and_not_an_off_target(self):
        """Routing it through ``background`` would be wrong the other way:
        background loci are deliberately absent from ``locus_to_target``, so a
        DESIRED target would arrive as an off-target hazard."""
        from cas13guide.contract.enums import Universe
        from cas13guide.services.conservation import variant_contexts
        from cas13guide.services.fasta import SequenceRecord

        contexts = variant_contexts(
            [SequenceRecord("t1", SEQ_A), SequenceRecord("V1", SEQ_A)],
            frozenset(Universe),
            target_id="t1",
            extra_targets=[SequenceRecord("t2", SEQ_B)],
        )
        for context in contexts:
            with self.subTest(context=context.name):
                self.assertIn("t2", context.transcriptome.loci)
                self.assertEqual(context.locus_to_target.get("t2"), "t2")
                self.assertEqual(context.locus_to_target.get(context.name)
                                 or context.locus_to_target.get("t1"), "t1")

    def test_the_primary_is_not_re_added_as_its_own_extra_target(self):
        """Each variant context holds ONE realisation of the primary target.

        ``run_request`` passes ``records[1:]``, so the primary is never in that
        list -- but a caller passing the whole record set must not end up with
        the reference sequence sitting inside every variant world beside the
        variant, which would make robustness measure the original rather than
        the variant.
        """
        from cas13guide.contract.enums import Universe
        from cas13guide.services.conservation import variant_contexts
        from cas13guide.services.fasta import SequenceRecord

        contexts = variant_contexts(
            [SequenceRecord("t1", SEQ_A), SequenceRecord("V1", SEQ_B)],
            frozenset(Universe),
            target_id="t1",
            extra_targets=[SequenceRecord("t1", SEQ_A)],
        )
        by_name = {context.name: context for context in contexts}
        self.assertEqual(set(by_name["reference"].transcriptome.loci), {"t1"})
        self.assertEqual(set(by_name["V1"].transcriptome.loci), {"V1"})
        self.assertEqual(by_name["V1"].locus_to_target, {"V1": "t1"})


if __name__ == "__main__":
    unittest.main()
