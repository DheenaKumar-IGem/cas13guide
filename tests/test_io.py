"""Sequence input and export (FR1.1, FR1.2, FR1.4, FR8.3), and NFR3.

The front door. If a malformed paste produces a stack trace, or a silently
altered sequence, nothing downstream can recover -- so these tests are as much
about the *messages* as about the parsing.
"""

import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.services.fasta import (
    RANKED_COLUMNS,
    SequenceInputError,
    SequenceRecord,
    controls_to_fasta,
    format_fasta,
    parse_fasta,
    parse_sequence_input,
    ranked_to_csv,
    ranked_to_fasta,
    read_fasta,
    write_exports,
    write_fasta,
)


class TestRawSequenceInput(unittest.TestCase):
    """FR1.1 -- a pasted sequence, in whatever shape it arrives."""

    def test_plain_rna(self):
        [record] = parse_sequence_input("AUGCAUGCAUGC")
        self.assertEqual(record.sequence, "AUGCAUGCAUGC")
        self.assertEqual(record.name, "query")

    def test_lowercase_and_whitespace_are_formatting_not_content(self):
        [record] = parse_sequence_input("  augc AUGC\n\taugc  \n")
        self.assertEqual(record.sequence, "AUGCAUGCAUGC")

    def test_dna_is_transcribed(self):
        [record] = parse_sequence_input("ATGCATGC")
        self.assertEqual(record.sequence, "AUGCAUGC")
        self.assertNotIn("T", record.sequence)

    def test_n_is_kept_because_it_is_a_real_base(self):
        [record] = parse_sequence_input("AUGCNAUGC")
        self.assertEqual(record.sequence, "AUGCNAUGC")

    def test_alignment_gap_characters_are_stripped(self):
        [record] = parse_sequence_input("AUG--CAU*GC")
        self.assertEqual(record.sequence, "AUGCAUGC")


class TestFastaInput(unittest.TestCase):
    """FR1.2 -- FASTA text and files."""

    FASTA = ">mir21 hsa-miR-21-5p\nUAGCUUAUCAGACUG\nAUGUUGACU\n>mir29c\nUAGCACCAUUUGAAAUCGGUUA\n"

    def test_multi_record_with_wrapped_lines(self):
        records = parse_fasta(self.FASTA)
        self.assertEqual([r.name for r in records], ["mir21", "mir29c"])
        self.assertEqual(records[0].length, 24)      # wrapped lines rejoined
        self.assertEqual(records[0].description, "hsa-miR-21-5p")
        self.assertEqual(records[1].description, "")

    def test_parse_sequence_input_detects_fasta_by_its_header(self):
        records = parse_sequence_input(self.FASTA)
        self.assertEqual(len(records), 2)

    def test_duplicate_names_are_kept_not_merged(self):
        """Usually a concatenated file; silently deduplicating would hide it."""
        records = parse_fasta(">a\nAUGC\n>a\nGCAU\n")
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0].sequence, records[1].sequence)

    def test_file_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "in.fasta"
            original = parse_fasta(self.FASTA)
            write_fasta(path, original)
            self.assertEqual(read_fasta(path), original)

    def test_wrapping_is_reversible(self):
        record = SequenceRecord("long", "AUGC" * 40)
        text = format_fasta([record], line_width=60)
        self.assertGreater(len(text.splitlines()), 2)
        self.assertEqual(parse_fasta(text)[0].sequence, record.sequence)


class TestMalformedInputSaysWhatAndWhere(unittest.TestCase):
    """FR1.4 / NFR3 -- reject clearly rather than crash or quietly repair."""

    def _message(self, text):
        with self.assertRaises(SequenceInputError) as caught:
            parse_sequence_input(text)
        return str(caught.exception)

    def test_a_bad_character_is_located(self):
        message = self._message("AUGCXAUGC")
        self.assertIn("'X'", message)
        self.assertIn("position 5", message)
        self.assertIn("A, C, G, U", message)

    def test_a_bad_character_inside_fasta_names_the_record(self):
        message = self._message(">r1\nAUGCZZ\n")
        self.assertIn("r1", message)
        self.assertIn("'Z'", message)

    def test_a_misplaced_header_is_diagnosed_as_such(self):
        """The most common paste error deserves better than "unexpected '>'"."""
        message = self._message("AUGC\n>r1\nAUGC")
        self.assertIn("FASTA header in the middle", message)

    def test_empty_input(self):
        self.assertIn("no sequence supplied", self._message("   \n  "))

    def test_empty_header(self):
        self.assertIn("empty header", self._message(">\nAUGC\n"))

    def test_header_with_no_sequence(self):
        self.assertIn("no sequence characters", self._message(">r1\n\n"))

    def test_a_missing_file_is_reported_not_raised_as_oserror(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SequenceInputError):
                read_fasta(Path(directory) / "nope.fasta")

    def test_nothing_is_silently_repaired(self):
        """A protein sequence must not become a plausible-looking RNA."""
        for junk in ("MKVLAAGIVGL", "AUGC123AUGC", "AUGC?AUGC"):
            with self.assertRaises(SequenceInputError):
                parse_sequence_input(junk)


def data_lines(text: str) -> list[str]:
    """The rows a consumer sees, i.e. everything past the provenance comments."""
    return [line for line in text.splitlines() if not line.startswith("#")]


class TestExport(unittest.TestCase):
    """FR8.3 -- CSV and FASTA export of ranked candidates."""

    REPORT = {
        "spec_fingerprint": "abc123",
        "verdict_counts": {"PASS": 0, "REVIEW": 1, "FAIL": 2},
        "provenance": {"records": [
            {"kind": "model", "name": "activity", "version": "model/heuristic-cas13/v1"},
            {"kind": "search", "name": "bound", "version": "index/exhaustive-scan/v1"},
        ]},
        "shortfalls": [],
        "ranked": [
            {
                "rank": 1, "guide_id": "g_abc", "spacer": "AUGCAUGCAUGCAUGCAUGCAU",
                "score": 7.0, "verdict": "REVIEW", "review_tier": "R2",
                "spans_junction": False, "pfs_3prime": "A", "pfs_preferred": True,
                "accessibility": 0.48, "duplex_dg_per_nt": -1.9,
                "predicted_signal": 0.13,
                "reasons": ["expression_unknown"], "qc_warnings": ["pfs_disfavoured"],
                "metrics": {"required_coverage": 0.67, "worst_off_target": 0.0,
                            "cumulative_burden": 0.0},
            }
        ],
        "controls": [
            {"guide_id": "ctrl_g_abc", "spacer": "UACGUACGUACGUACGUACGUA",
             "matched_to": "g_abc", "gc": 0.5, "verified_no_hits": True},
            {"guide_id": "ctrl_none", "spacer": "", "matched_to": "g_x",
             "gc": 0.5, "verified_no_hits": False},
        ],
    }

    def test_csv_has_a_stable_declared_header(self):
        text = ranked_to_csv(self.REPORT)
        header = data_lines(text)[0].split(",")
        for column in RANKED_COLUMNS:
            self.assertIn(column, header)
        self.assertIn("required_coverage", header)

    def test_csv_carries_the_verdict_and_tier(self):
        """A REVIEW row must be identifiable as one in the exported file."""
        row = data_lines(ranked_to_csv(self.REPORT))[1]
        self.assertIn("REVIEW", row)
        self.assertIn("R2", row)

    def test_list_valued_columns_are_flattened_not_repr_d(self):
        row = data_lines(ranked_to_csv(self.REPORT))[1]
        self.assertIn("expression_unknown", row)
        self.assertNotIn("[", row)

    def test_fasta_export_is_parseable_and_annotated(self):
        text = ranked_to_fasta(self.REPORT)
        [record] = parse_fasta(text)
        self.assertEqual(record.name, "g_abc")
        self.assertEqual(record.sequence, "AUGCAUGCAUGCAUGCAUGCAU")
        self.assertIn("rank=1", record.description)
        self.assertIn("predicted_signal=0.13", record.description)

    def test_controls_export_skips_the_ones_that_could_not_be_built(self):
        records = parse_fasta(controls_to_fasta(self.REPORT))
        self.assertEqual([r.name for r in records], ["ctrl_g_abc"])

    def test_write_exports_produces_all_three_files(self):
        with tempfile.TemporaryDirectory() as directory:
            written = write_exports(self.REPORT, directory, stem="cand")
            self.assertEqual(set(written), {"csv", "fasta", "controls"})
            for path in written.values():
                self.assertTrue(path.exists(), path)
                self.assertGreater(path.stat().st_size, 0)

    def test_an_empty_ranking_still_writes_a_valid_header(self):
        text = ranked_to_csv({"ranked": []})
        self.assertEqual(len(data_lines(text)), 1)
        self.assertIn("guide_id", text)


class TestProvenanceTravelsWithTheFile(unittest.TestCase):
    """NFR4 -- an exported file outlives the session that produced it.

    Without a header it is a table of numbers with no model version, no
    parameters and no statement of what was not checked, and it looks exactly
    like a table that has all three.
    """

    def test_the_csv_header_is_comment_prefixed_so_readers_skip_it(self):
        text = ranked_to_csv(TestExport.REPORT)
        self.assertTrue(text.startswith("# "))
        # pandas' comment='#' / R's comment.char='#' see a clean table
        self.assertEqual(data_lines(text)[0].split(",")[0], "rank")

    def test_it_names_the_version_fingerprint_and_models(self):
        text = ranked_to_csv(TestExport.REPORT)
        self.assertIn("cas13guide", text)
        self.assertIn("abc123", text)                     # spec fingerprint
        self.assertIn("model: activity", text)

    def test_limitations_travel_with_the_file(self):
        text = ranked_to_csv(TestExport.REPORT, limitations=["CRITICAL: nothing checked"])
        self.assertIn("limitation: CRITICAL: nothing checked", text)

    def test_the_fasta_header_uses_the_fasta_comment_character(self):
        """'#' would be read as sequence by some parsers; ';' is the FASTA one."""
        text = ranked_to_fasta(TestExport.REPORT, limitations=["INFO: x"])
        self.assertTrue(text.startswith("; "))
        self.assertIn("; limitation: INFO: x", text)

    def test_an_exported_fasta_still_parses(self):
        """The round trip the header could most easily have broken."""
        text = ranked_to_fasta(TestExport.REPORT, limitations=["INFO: x"])
        [record] = parse_fasta(text)
        self.assertEqual(record.name, "g_abc")
        self.assertEqual(record.sequence, "AUGCAUGCAUGCAUGCAUGCAU")

    def test_comment_lines_are_skipped_by_the_parser_generally(self):
        records = parse_fasta("; a comment\n# another\n>r1\nAUGC\n; trailing\n")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sequence, "AUGC")


class TestReviewGuidesReachTheRanking(unittest.TestCase):
    """REVIEW means "look at this", which requires it to be shown.

    Ranking only PASS made the R1/R2/R3 grading invisible: with honest
    out-of-domain flagging the REVIEW bucket is routinely the majority, and a
    run that graded every guide handed back an empty table. The panel stays
    PASS-only; the table a human reads is wider.
    """

    def _report(self, coverage_floor):
        import random

        from cas13guide.contract.enums import Universe
        from cas13guide.pipeline.orchestrator import DesignContext, run_design
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.indexes import ExhaustiveScanIndex
        from cas13guide.services.registry import (
            HeuristicCas13Model, ModelRegistry, NullAccessibilityModel,
        )
        from cas13guide.services.transcriptome import Locus, Transcriptome

        spec = make_spec(required_coverage_min=coverage_floor)
        rng = random.Random(11)
        calibration = CalibrationService()
        for regime in ("mm0", "mm1-2", "mm3+"):
            key = CalibrationService.stratum_key(
                spec.profile.effector.value, spec.profile.assay.value, regime
            )
            for i in range(150):
                truth = rng.random()
                calibration.add_observation(
                    key, min(1.0, max(0.0, truth + rng.gauss(0, 0.1))), truth,
                    group=f"T{i % 12:02d}",
                )
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("L_T1", SEQ_A, Universe.CODING, gene="TGT", expression=100.0))
        # A locus with *unmeasured* expression that actually shares sequence
        # with the target, so guides tiled over that stretch pick up a hit
        # whose expression is unknown -> a deterministic R2 EXPRESSION_UNKNOWN.
        # An unrelated sequence would produce no hits and therefore no review.
        world.add(Locus("L_UNK", SEQ_B[:20] + SEQ_A[:40] + SEQ_B[:20],
                        Universe.NCRNA, gene="UNK", expression=None))
        context = DesignContext(
            "reference", world,
            ExhaustiveScanIndex(world, spec.context.declared_universes),
            {"L_T1": "t1"},
        )
        return run_design(
            spec, [context],
            ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel()),
            calibration, panel_size=2,
        )

    def test_review_guides_appear_in_the_ranking(self):
        report = self._report(coverage_floor=0.50)
        data = report.to_dict()
        verdicts = {row["verdict"] for row in data["ranked"]}
        self.assertTrue(data["ranked"], "ranking should not be empty")
        self.assertIn("REVIEW", verdicts)
        self.assertNotIn("FAIL", verdicts)
        self.assertEqual(
            sum(1 for row in data["ranked"] if row["verdict"] == "REVIEW"),
            data["verdict_counts"]["REVIEW"],
            "every REVIEW guide should be rankable, not just some",
        )

    def test_every_ranked_row_carries_its_verdict_and_tier(self):
        data = self._report(coverage_floor=0.50).to_dict()
        for row in data["ranked"]:
            self.assertIn(row["verdict"], {"PASS", "REVIEW"})
            if row["verdict"] == "REVIEW":
                self.assertIn(row["review_tier"], {"R1", "R2", "R3"})

    def test_failed_guides_still_never_reach_the_ranking(self):
        data = self._report(coverage_floor=0.99).to_dict()
        for row in data["ranked"]:
            self.assertNotEqual(row["verdict"], "FAIL")

    def test_the_panel_stays_pass_only(self):
        """The machine's recommendation must remain certified."""
        report = self._report(coverage_floor=0.50)
        passing = {
            row["guide_id"] for row in report.to_dict()["ranked"]
            if row["verdict"] == "PASS"
        }
        for guide in report.panel.guides:
            self.assertIn(guide.guide_id, passing)


if __name__ == "__main__":
    unittest.main()
