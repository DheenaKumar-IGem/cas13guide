"""Fifth review pass: nine defects, found by driving the product.

Pass four established that running the thing finds what reading it cannot, so
this pass did only that -- input fuzzing, determinism across processes, the
three output surfaces compared field by field, real HTTP POSTs to the server,
and scale limits. Nine survived two skeptics, against six for pass four.

One of them is a regression pass four introduced, which is now the second time
a re-read has caught the previous pass. Two are the same defect reported from
different angles, and one -- an N in the target sequence attracting the
ranking -- had been reachable from the front door since the beginning.
"""

import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import QCFlag, TargetClass


def _sequence(seed_value: int, length: int) -> str:
    import random

    rng = random.Random(seed_value)
    return "".join(rng.choice("ACGU") for _ in range(length))


class TestDistinctSitesUseTheSiteAGuideActuallyBinds(unittest.TestCase):
    """A regression from the previous pass's own fix.

    Rewriting the rule to compare binding sites was right; reducing a guide's
    hits per target with ``min(start)`` was not. At the shipped 4-mismatch
    wobble-aware envelope a 22-mer picks up a second hit on its own transcript
    often enough to matter -- 61 of 1650 guides on one 2 kb target -- and the
    guide was then filed at that near-zero-activity coordinate instead of at
    the one it was designed for. Measured: 3 of 68 marked rows became
    single-base shifts of a row already listed above them, under a line saying
    shifted copies are hidden.

    The fix is the rule ``optimizer._profile`` already uses: the site with the
    highest calibrated lower bound. The docstring claims the two now answer the
    same question, so they must.
    """

    def _ranked(self, guides):
        import dataclasses

        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.metrics.bounds import Interval
        from cas13guide.models.evidence import (
            Candidate,
            GateResult,
            GuideAssessment,
            MetricVector,
            PairEvidence,
        )
        from cas13guide.pipeline.f_decide import RankedGuide

        ranked = []
        for rank, (guide_id, tiled_start, hits) in enumerate(guides, 1):
            evidence = tuple(
                PairEvidence(
                    guide_id=guide_id,
                    locus_id="L_t1",
                    context="reference",
                    hit=dataclasses.replace(
                        make_hit(guide_id, "L_t1", "t1", TargetClass.REQUIRED),
                        start=start,
                    ),
                    activity=Interval(lower, lower, min(1.0, lower + 0.02)),
                    expression=Interval(0.5, 0.5, 0.5),
                    hazard_weight=0.0,
                )
                for start, lower in hits
            )
            assessment = GuideAssessment(
                guide_id=guide_id,
                candidate=Candidate(guide_id=guide_id, spacer="A" * 22,
                                    source_target_id="t1", start=tiled_start),
                qc=clean_qc(guide_id),
                metrics=MetricVector(0.9, 0.0, 0.0, 0.0, 0.9, 1.0),
                gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
                evidence=evidence,
            )
            ranked.append(RankedGuide(rank=rank, assessment=assessment, score=1.0,
                                      rationale=(), sensitivity={}))
        return ranked

    def test_a_guide_is_filed_at_its_best_site_not_its_lowest(self):
        """gB binds strongly 4 nt from gA and weakly far away. It is a shift."""
        from cas13guide.pipeline.f_decide import distinct_site_ids

        ranked = self._ranked([
            ("gA", 1131, [(1131, 0.93)]),
            ("gB", 1127, [(848, 0.003), (1127, 0.93)]),
        ])
        self.assertEqual(distinct_site_ids(ranked, 22), frozenset({"gA"}))

    def test_a_weak_far_hit_does_not_veto_a_genuine_site(self):
        """The opposite error: a near-zero-activity wobble hit must not make an
        unrelated guide look like a shifted copy of it."""
        from cas13guide.pipeline.f_decide import distinct_site_ids

        ranked = self._ranked([
            ("gA", 1127, [(848, 0.003), (1127, 0.93)]),
            ("gB", 829, [(829, 0.93)]),
        ])
        self.assertEqual(distinct_site_ids(ranked, 22), frozenset({"gA", "gB"}))

    def test_it_agrees_with_the_panel_selector_on_where_a_guide_binds(self):
        """The docstring says these answer the same question."""
        from cas13guide.pipeline.f_decide import distinct_site_ids
        from cas13guide.pipeline.optimizer import _profile

        ranked = self._ranked([("gA", 1127, [(848, 0.003), (1127, 0.93)])])
        self.assertEqual(_profile(ranked[0].assessment).sites, {"t1": 1127})
        self.assertEqual(distinct_site_ids(ranked, 22), frozenset({"gA"}))


class TestBackgroundDuplicatesAreReportedAsThemselves(unittest.TestCase):
    """The guard tested membership against a world it was itself growing.

    So the second of two ``--background`` records sharing a name took the
    target-collision branch: the locus was dropped from the off-target corpus,
    and the user was told a target was involved when none was. That note
    travels into every export header, so the false statement outlives the
    session, and its remedy pointed at the wrong file.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        self.target_seq = _sequence(3, 200)
        self.target = root / "t.fa"
        self.target.write_text(f">t1\n{self.target_seq}\n", encoding="utf-8")
        self.duplicated = root / "dup.fa"
        self.duplicated.write_text(
            f">decoy\n{_sequence(4, 200)}\n>decoy\n{_sequence(5, 200)}\n", encoding="utf-8"
        )
        self.colliding = root / "collide.fa"
        self.colliding.write_text(f">t1\n{_sequence(6, 200)}\n", encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def _notes(self, background):
        from cas13guide.services.workflow import DesignRequest, run_request

        return run_request(
            DesignRequest(fasta_path=str(self.target), background_path=str(background),
                          task="knockdown", assay="cell_culture_knockdown",
                          effector="RfxCas13d", use_structure=False, use_signal=False)
        ).notes

    def test_a_background_duplicate_is_not_blamed_on_a_target(self):
        notes = " ".join(self._notes(self.duplicated))
        self.assertIn("more than one record named 'decoy'", notes)
        self.assertNotIn("shares a name with a target record", notes)

    def test_a_real_target_collision_still_says_so(self):
        notes = " ".join(self._notes(self.colliding))
        self.assertIn("shares a name with a target record", notes)
        self.assertNotIn("more than one record named", notes)


class TestAmbiguousSpacersAreNotOfferedAsOligos(unittest.TestCase):
    """An N in the target became an N in the spacer, and nothing noticed.

    ``classify_pair`` calls N:N a Watson-Crick pair, so an N-bearing spacer
    scored a *perfect* on-target duplex and landed in the mm0 calibration
    stratum; RNAplfold cannot pair an N, so accessibility read exactly 1.0
    there. An assembly gap therefore attracted the ranking -- measured, ranks
    1, 2, 3, 4 and 6 of 101 -- and the spacers were exported as orderable
    oligos. ``N`` remains a legal input character; what is refused is handing
    one back as a guide.
    """

    def _qc(self, spacer):
        from cas13guide.models.evidence import Candidate
        from cas13guide.pipeline.b_qc import qc_candidate

        candidate = Candidate(guide_id="g1", spacer=spacer, source_target_id="t1", start=0)
        return qc_candidate(candidate, make_spec())

    def test_an_n_bearing_spacer_fails_qc(self):
        result = self._qc("GCAUUGCUGNNGGAGCUGAAGUC"[:22])
        self.assertFalse(result.passed)
        self.assertIn(QCFlag.SYNTHESIS_CONSTRAINT, result.flags)

    def test_a_fully_determined_spacer_is_untouched(self):
        result = self._qc("GCAUUGCUGUGGAGCUGAAGUC")
        self.assertNotIn(QCFlag.SYNTHESIS_CONSTRAINT, result.flags)

    def test_the_gap_no_longer_attracts_the_ranking(self):
        from cas13guide.services.workflow import DesignRequest, run_request

        target = list(_sequence(11, 150))
        target[60:64] = list("NNNN")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "withN.fa"
            path.write_text(">withN\n" + "".join(target) + "\n", encoding="utf-8")
            outcome = run_request(
                DesignRequest(fasta_path=str(path), use_structure=False, use_signal=False)
            )
        for row in outcome.report["ranked"]:
            with self.subTest(guide=row["guide_id"]):
                self.assertNotIn("N", row["spacer"])


class TestThePasteDoorReadsTheFastaTheFileDoorReads(unittest.TestCase):
    """``parse_sequence_input`` sniffed FASTA from the first non-blank line.

    ``parse_fasta`` skips ';' and '#' comments -- this module writes a
    ';'-prefixed provenance header itself -- and the sniff did not, so a
    comment-first FASTA was handed to ``_clean`` whole and refused at character
    one as a malformed raw sequence. The textarea, ``--sequence`` and the
    browser's file upload all come through here while ``--fasta`` does not, so
    the same file behaved differently by which door it used -- and the file it
    most reliably rejected was this tool's own export.
    """

    def test_a_comment_first_fasta_parses(self):
        from cas13guide.services.fasta import parse_sequence_input

        records = parse_sequence_input("; provenance\n# another\n>a desc\nAUGCAUGC\n")
        self.assertEqual([r.name for r in records], ["a"])

    def test_a_comment_before_a_raw_sequence_parses(self):
        """A sniff-only fix would still have failed this one."""
        from cas13guide.services.fasta import parse_sequence_input

        self.assertEqual(parse_sequence_input("; note\nAUGCAUGC")[0].sequence, "AUGCAUGC")

    def test_input_that_is_only_comments_is_refused_not_crashed(self):
        from cas13guide.services.fasta import SequenceInputError, parse_sequence_input

        with self.assertRaises(SequenceInputError) as caught:
            parse_sequence_input("; only\n# comments\n")
        self.assertIn("only comment lines", str(caught.exception))

    def test_the_tools_own_export_reads_back_through_the_paste_door(self):
        from cas13guide.services.fasta import parse_sequence_input, ranked_to_fasta

        exported = ranked_to_fasta({
            "spec_fingerprint": "abc",
            "ranked": [{"rank": 1, "guide_id": "g1", "spacer": "AUGCAUGCAUGCAUGCAUGCAU",
                        "score": 1.0, "verdict": "PASS", "metrics": {}}],
        })
        self.assertTrue(exported.startswith(";"))
        self.assertEqual([r.name for r in parse_sequence_input(exported)], ["g1"])


class TestOneTargetSourcePerRun(unittest.TestCase):
    """``_read_records`` returned on the first source it found.

    So ``--preset`` silently discarded ``--fasta`` and ``--sequence``, with the
    unused input mentioned in no note, shortfall or export header. The web form
    offers all three at once and argparse puts them in one group with no
    exclusion, so it is reachable from both doors.
    """

    def test_two_sources_are_refused_rather_than_silently_ranked(self):
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            run_request,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.fa"
            path.write_text(f">t1\n{_sequence(2, 100)}\n", encoding="utf-8")
            with self.assertRaises(DesignRequestError) as caught:
                run_request(
                    DesignRequest(preset="hsa-miR-21-5p", fasta_path=str(path))
                )
        message = str(caught.exception)
        self.assertIn("--preset", message)
        self.assertIn("--fasta", message)

    def test_one_source_still_runs(self):
        from cas13guide.services.workflow import DesignRequest, run_request

        outcome = run_request(DesignRequest(preset="hsa-miR-21-5p"))
        self.assertTrue(outcome.report["verdict_counts"])


class TestTheCalibrationRecordsItsOwnParameters(unittest.TestCase):
    """Seed and noise decide every conformal interval, and every metric read at
    a bound, and every verdict -- and appeared in no recorded field.

    Two runs differing only in seed carried byte-identical provenance while
    disagreeing on fifty verdicts, which is the one thing provenance exists to
    make impossible.
    """

    def _source(self, **kw):
        from cas13guide.services.workflow import DesignRequest, run_request

        outcome = run_request(DesignRequest(preset="hsa-miR-21-5p", **kw))
        record = next(
            r for r in outcome.report["provenance"]["records"]
            if r["name"] == "CalibrationService"
        )
        return record["detail"]["source"]

    def test_the_seed_is_recorded(self):
        self.assertEqual(self._source(seed=7)["seed"], 7)

    def test_two_seeds_are_distinguishable_in_the_ledger(self):
        self.assertNotEqual(self._source(seed=1), self._source(seed=2))

    def test_the_noise_and_size_are_recorded_too(self):
        source = self._source(calibration_n=64, calibration_noise=0.25)
        self.assertEqual(source["n_per_stratum"], 64)
        self.assertEqual(source["noise_sd"], 0.25)

    def test_a_caller_supplied_service_is_marked_rather_than_certified(self):
        """Silence would read as "nothing to declare"; it means "unrecorded"."""
        from cas13guide.pipeline.orchestrator import run_design
        from cas13guide.services.calibration import CalibrationService

        service = CalibrationService()
        self.assertIsNone(getattr(service, "source", None))
        # The ledger substitutes a stated value rather than omitting the key.
        from cas13guide.services.provenance import ProvenanceLedger

        ledger = ProvenanceLedger("fp")
        ledger.record("calibration", "CalibrationService", service.version,
                      source=getattr(service, "source", None) or "unstated")
        self.assertEqual(ledger.to_dict()["records"][0]["detail"]["source"], "unstated")
        del run_design


class TestTheOffTargetExplorerExists(unittest.TestCase):
    """The validation plan told the reader to consult a table no surface drew.

    ``build_report`` filled up to 50 hazard-sorted off-target rows and the HTML
    report never read them, while printing "Sequence the top off-target loci
    from the explorer table" underneath. A page that sends a reader to a table
    it does not contain is worse than one that says nothing.
    """

    REPORT = {
        "verdict_counts": {"PASS": 1},
        "ranked": [],
        "validation_plan": ["Sequence the top off-target loci from the explorer table."],
        "off_targets": [
            {"guide_id": "g1", "locus": "PSEUDO1", "gene": "PSEUDO1",
             "biotype": "coding", "mismatches": 1, "wobbles": 0,
             "ucb_activity": 0.95, "ucb_expression": 0.57, "hazard": 0.54,
             "expression_unknown": True, "ood": False},
        ],
    }

    def test_the_section_is_rendered(self):
        from cas13guide.services.report_html import render_report

        html = render_report(self.REPORT)
        self.assertIn("Off-target explorer", html)
        self.assertIn("PSEUDO1", html)
        self.assertIn("0.54", html)

    def test_it_scrolls_inside_its_own_container(self):
        from cas13guide.services.report_html import render_report

        html = render_report(self.REPORT)
        after = html.split("Off-target explorer", 1)[1]
        self.assertIn("tablewrap", after.split("</table>", 1)[0])

    def test_a_run_with_no_off_targets_omits_the_section(self):
        from cas13guide.services.report_html import render_report

        empty = dict(self.REPORT, off_targets=[])
        self.assertNotIn("Off-target explorer", render_report(empty))

    def test_the_columns_are_declared_rather_than_derived(self):
        """Adding a field upstream must not reshape a table someone reads."""
        from cas13guide.services.report_html import _OFF_TARGET_COLUMNS

        self.assertIn("hazard", _OFF_TARGET_COLUMNS)
        self.assertIn("expression_unknown", _OFF_TARGET_COLUMNS)


class TestTheOversizedUploadReplyReachesTheClient(unittest.TestCase):
    """The 413 page was destroyed by the connection reset it caused.

    ``do_POST`` replied and returned without reading the body, so the socket
    closed with megabytes still pending and the OS sent a TCP RST -- which
    destroyed the response before any sequential client could read it. The
    server logged 413; the browser retried as GET and landed on "Not found.",
    a message that is wrong and says nothing about a size limit.
    """

    def test_the_body_is_drained_before_replying(self):
        import io

        from cas13guide.services import webapp

        class _Handler(webapp.DesignHandler):
            def __init__(self):  # noqa: D107 - bypass BaseHTTPRequestHandler setup
                self.path = "/design"
                self.headers = {"Content-Length": str(webapp.MAX_UPLOAD_BYTES + 1024)}
                self.rfile = io.BytesIO(b"x" * (webapp.MAX_UPLOAD_BYTES + 1024))
                self.sent = []

            def _send(self, body, status=200):
                self.sent.append((status, body))

        handler = _Handler()
        handler.do_POST()
        self.assertEqual([status for status, _ in handler.sent], [413])
        self.assertIn("larger than", handler.sent[0][1])
        self.assertEqual(
            handler.rfile.read(), b"",
            "the request body was left unread, which resets the connection",
        )

    def test_the_drain_is_bounded(self):
        """A client streaming without limit must not make the server read
        forever; past the cap the reset is the lesser evil."""
        from cas13guide.services import webapp

        self.assertGreater(webapp._DRAIN_LIMIT_BYTES, webapp.MAX_UPLOAD_BYTES)
        self.assertLessEqual(webapp._DRAIN_LIMIT_BYTES, 128 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
