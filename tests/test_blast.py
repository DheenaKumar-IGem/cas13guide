"""BLAST off-target search (FR4.1-FR4.3), and its honest limits.

BLAST+ and a RefSeq database are infrastructure (DR1, DR3) that this repository
does not ship, so the subprocess path is exercised against a stub executable
that emits real-format output. That checks the argv, the parsing, the strand and
length filtering and the wobble re-scoring -- everything except whether NCBI's
aligner behaves, which is not this package's contract to test.
"""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass, Universe
from cas13guide.metrics.aggregate import build_metric_vector
from cas13guide.services.blast import (
    OUTFMT_FIELDS,
    BlastOffTargetIndex,
    BlastUnavailable,
    parse_outfmt6,
)
from cas13guide.util.seq import revcomp

SPACER = "GCAUUGCUGUGGAGCUGAAGUC"
SITE = revcomp(SPACER)                      # the perfect target site, RNA
SITE_DNA = SITE.replace("U", "T")


def row(subject="NM_000546", sseq=SITE_DNA, length=None, mismatch=0, gapopen=0,
        sstart=101, send=None, strand="plus"):
    length = length if length is not None else len(sseq)
    send = send if send is not None else sstart + length - 1
    return "\t".join(str(v) for v in (
        "query", subject, "100.0", length, mismatch, gapopen,
        1, length, sstart, send, "1e-5", 44.1, strand, sseq,
    ))


def stub_blastn(directory: Path, stdout: str, exit_code: int = 0) -> Path:
    """A fake blastn that prints canned output, for the subprocess path."""
    script = Path(directory) / "stub.py"
    script.write_text(
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    if os.name == "nt":
        launcher = Path(directory) / "blastn.bat"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
                            encoding="utf-8")
    else:
        launcher = Path(directory) / "blastn"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n',
                            encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return launcher


class TestParsing(unittest.TestCase):
    def test_a_well_formed_row_round_trips(self):
        [parsed] = parse_outfmt6(row())
        self.assertEqual(parsed.subject, "NM_000546")
        self.assertEqual(parsed.zero_based_start, 100)     # sstart is 1-based
        self.assertEqual(parsed.strand, "plus")
        self.assertTrue(parsed.is_full_length)

    def test_blank_lines_and_comments_are_ignored(self):
        self.assertEqual(len(parse_outfmt6(f"# header\n\n{row()}\n\n")), 1)

    def test_the_wrong_column_count_names_the_required_format(self):
        with self.assertRaises(ValueError) as caught:
            parse_outfmt6("a\tb\tc")
        self.assertIn("qseqid", str(caught.exception))

    def test_gaps_are_stripped_from_the_subject_sequence(self):
        [parsed] = parse_outfmt6(row(sseq=SITE_DNA[:10] + "-" + SITE_DNA[10:]))
        self.assertNotIn("-", parsed.subject_sequence)


class TestCommandConstruction(unittest.TestCase):
    """Short queries need specific flags or almost nothing is reported."""

    def setUp(self):
        self.index = BlastOffTargetIndex(database="refseq_rna",
                                         universes=frozenset({Universe.CODING}))

    def test_it_uses_the_short_query_task(self):
        argv = self.index.command("q.fa")
        self.assertIn("-task", argv)
        self.assertEqual(argv[argv.index("-task") + 1], "blastn-short")

    def test_it_searches_only_the_plus_strand(self):
        """A transcript database holds sense strands; a minus hit is not a site."""
        argv = self.index.command("q.fa")
        self.assertEqual(argv[argv.index("-strand") + 1], "plus")

    def test_it_requests_exactly_the_fields_the_parser_expects(self):
        argv = self.index.command("q.fa")
        self.assertEqual(argv[argv.index("-outfmt") + 1],
                         "6 " + " ".join(OUTFMT_FIELDS))

    def test_low_complexity_masking_is_off(self):
        """DUST would mask exactly the AU-rich sites worth checking."""
        argv = self.index.command("q.fa")
        self.assertEqual(argv[argv.index("-dust") + 1], "no")

    def test_a_missing_executable_is_reported_clearly(self):
        index = BlastOffTargetIndex(database="db", universes=frozenset(),
                                    executable="definitely-not-blastn-xyz")
        with self.assertRaises(BlastUnavailable) as caught:
            index.ensure_available()
        message = str(caught.exception)
        self.assertIn("not on PATH", message)
        self.assertIn("ExhaustiveScanIndex", message)   # names the alternative


class TestSearchAgainstAStub(unittest.TestCase):
    """The subprocess path, with a fake blastn emitting real-format output."""

    def _search(self, stdout, max_mismatches=4, exit_code=0):
        with tempfile.TemporaryDirectory() as directory:
            launcher = stub_blastn(Path(directory), stdout, exit_code)
            index = BlastOffTargetIndex(
                database="refseq_rna",
                universes=frozenset({Universe.CODING}),
                executable=str(launcher),
            )
            return index.search("g1", SPACER, max_mismatches)

    def test_a_perfect_hit_becomes_a_search_hit(self):
        result = self._search(row() + "\n")
        [hit] = result.hits
        self.assertEqual(hit.locus_id, "NM_000546")
        self.assertEqual(hit.start, 100)
        self.assertEqual(hit.target_site, SITE)          # transcribed back to RNA
        self.assertEqual(hit.duplex.n_mismatch, 0)

    def test_minus_strand_alignments_are_dropped(self):
        """They do not correspond to a bindable site on a sense transcript."""
        self.assertEqual(self._search(row(strand="minus") + "\n").hits, ())

    def test_partial_alignments_are_dropped(self):
        """Padding one out to spacer length would be inventing sequence."""
        short = SITE_DNA[:15]
        self.assertEqual(self._search(row(sseq=short, length=15) + "\n").hits, ())

    def test_gapped_alignments_are_dropped(self):
        self.assertEqual(self._search(row(gapopen=1) + "\n").hits, ())

    def test_the_envelope_is_enforced_on_the_re_derived_count(self):
        site = list(SITE_DNA)
        for i in (2, 6, 10, 14, 18):
            site[i] = "C" if site[i] != "C" else "G"
        line = row(sseq="".join(site)) + "\n"
        self.assertEqual(self._search(line, max_mismatches=2).hits, ())
        self.assertTrue(self._search(line, max_mismatches=6).hits)

    def test_wobbles_are_re_scored_not_taken_from_blast(self):
        """BLAST calls G:U a mismatch; for an RNA:RNA duplex it is a base pair.

        Trusting BLAST's column would over-count every wobble and drop real
        hits at a tight envelope.
        """
        site = list(SITE)
        index_c = site.index("C")
        site[index_c] = "U"                       # C->U in target space: a wobble
        dna = "".join(site).replace("U", "T")
        # BLAST would report 1 mismatch here; the duplex has 0.
        result = self._search(row(sseq=dna, mismatch=1) + "\n", max_mismatches=0)
        [hit] = result.hits
        self.assertEqual(hit.duplex.n_mismatch, 0)
        self.assertEqual(hit.duplex.n_wobble, 1)

    def test_a_failing_blastn_is_surfaced_not_swallowed(self):
        with self.assertRaises(BlastUnavailable):
            self._search("", exit_code=2)

    def test_hits_come_back_in_a_deterministic_order(self):
        lines = (
            row(subject="NM_2", sstart=50) + "\n"
            + row(subject="NM_1", sstart=300) + "\n"
            + row(subject="NM_1", sstart=10) + "\n"
        )
        hits = self._search(lines).hits
        self.assertEqual(
            [(h.locus_id, h.start) for h in hits],
            [("NM_1", 9), ("NM_1", 299), ("NM_2", 49)],
        )


class TestItNeverClaimsExhaustiveness(unittest.TestCase):
    """The load-bearing property: a seeded search must not look complete."""

    def setUp(self):
        self.bound = BlastOffTargetIndex(
            database="refseq_rna", universes=frozenset({Universe.CODING})
        ).bound(4)

    def test_the_bound_is_heuristic(self):
        self.assertFalse(self.bound.exhaustive)
        self.assertFalse(self.bound.covers(4))
        self.assertFalse(self.bound.covers(0))
        self.assertIn("heuristic", self.bound.describe())

    def test_it_declares_that_it_is_not_wobble_aware(self):
        self.assertFalse(self.bound.wobble_aware)

    def test_the_note_says_why_recall_is_not_guaranteed(self):
        note = self.bound.note.lower()
        self.assertIn("seeded", note)
        self.assertIn("wobble", note)
        self.assertIn("word_size", note)

    def test_the_gate_turns_that_into_a_review(self):
        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.pipeline.f_decide import gate

        spec = make_spec()
        evidence = (make_evidence(0.9, 0.95, target_id="t1",
                                  target_class=TargetClass.REQUIRED),)
        metrics = build_metric_vector({"reference": evidence}, clean_qc(), spec)
        result = gate(metrics, clean_qc(), evidence, self.bound, frozenset(), spec)
        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIn(ReasonCode.SEARCH_NOT_EXHAUSTIVE, result.reasons)



class TestItIsReachableFromTheCli(unittest.TestCase):
    """A documented feature wired to nothing is a hole with documentation.

    BlastOffTargetIndex was referenced nowhere outside this module and its own
    tests: no CLI flag, no call site, no way for a user to select it however
    much BLAST+ they had installed. DR1/DR3 were never what stood in the way.
    """

    def test_the_request_carries_a_database_path(self):
        from cas13guide.services.workflow import DesignRequest

        self.assertTrue(hasattr(DesignRequest(), "blast_db"))

    def test_the_cli_exposes_it(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "cas13guide", "design", "--help"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertIn("--blast-db", result.stdout)
        self.assertIn("--blast-universes", result.stdout)

    def test_without_a_database_the_exhaustive_scanner_is_used(self):
        """The default has to stay the one that can claim exhaustiveness."""
        from cas13guide.services.indexes import ExhaustiveScanIndex
        from cas13guide.services.transcriptome import Transcriptome
        from cas13guide.services.workflow import DesignRequest, _reference_index

        world = Transcriptome("GRCh38", "v44")
        self.assertIsInstance(
            _reference_index(DesignRequest(), world), ExhaustiveScanIndex
        )

    def test_a_missing_toolkit_is_refused_legibly_not_deep_in_a_run(self):
        from cas13guide.services.transcriptome import Transcriptome
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            _reference_index,
        )

        world = Transcriptome("GRCh38", "v44")
        request = DesignRequest(blast_db="/nonexistent/db")
        import shutil

        if shutil.which("blastn") is not None:
            self.skipTest("blastn is installed, so absence cannot be tested")
        with self.assertRaises(DesignRequestError) as caught:
            _reference_index(request, world)
        self.assertIn("blastn", str(caught.exception))

    def test_an_unknown_universe_names_the_valid_ones(self):
        from cas13guide.services.transcriptome import Transcriptome
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            _reference_index,
        )

        world = Transcriptome("GRCh38", "v44")
        request = DesignRequest(blast_db="/db", blast_universes="not_a_universe")
        with self.assertRaises(DesignRequestError) as caught:
            _reference_index(request, world)
        self.assertIn("coding", str(caught.exception))

    def test_an_empty_universe_assertion_is_refused(self):
        """Declaring nothing would make the adequacy gate check nothing."""
        from cas13guide.services.transcriptome import Transcriptome
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            _reference_index,
        )

        world = Transcriptome("GRCh38", "v44")
        with self.assertRaises(DesignRequestError):
            _reference_index(
                DesignRequest(blast_db="/db", blast_universes="  "), world
            )

class TestTheDatabaseHoldsTheTargetToo(unittest.TestCase):
    """A transcript database contains the target, and BLAST cannot say so.

    `annotate_hit` names a hit by looking its locus id up in
    `locus_to_target`, which is keyed on the records this run assembled. A
    BLAST subject id is an accession from someone else's database, so the
    target's own transcript comes back as an unannotated, zero-mismatch,
    full-length hit -- the site the guide was designed to bind, scored as the
    strongest off-target evidence there is.

    `_reference_index` already unions BLAST with the local scan so the
    *on-target* evidence survives; its comment says exactly why. This is the
    other half, and it is not fixed by reclassifying: a site identical to the
    target's is either the target or a paralog identical across the whole
    site, and nothing in the run can tell which. Dropping it would hide a real
    off-target, so the hazard stays counted and the run says what it cannot
    distinguish.

    Measured against a control database that does not hold the target,
    identical in every other respect: worst_off_target moves from a median of
    0.000 to 0.600 for every guide and all fifty rows of the off-target
    explorer become the guide's own target -- the exact row the validation
    plan tells a reader to prioritise for sequencing.
    """

    @classmethod
    def _stub_self_hit(cls, directory: Path) -> Path:
        """A blastn that answers every query with a perfect hit on itself."""
        script = Path(directory) / "selfhit.py"
        script.write_text(
            "import sys" + chr(10)
            + "argv = sys.argv[1:]" + chr(10)
            + 'q = argv[argv.index("-query") + 1]' + chr(10)
            + 'text = open(q, encoding="utf-8").read().split()' + chr(10)
            + 'seq = "".join(x for x in text if not x.startswith(">"))' + chr(10)
            + "n = len(seq)" + chr(10)
            + 'row = ["query", "NM_000546.6", "100.000", str(n), "0", "0",'
            + ' "1", str(n), "101", str(100 + n), "1e-05", "44.1", "plus", seq]'
            + chr(10)
            + 'sys.stdout.write(chr(9).join(row) + chr(10))' + chr(10),
            encoding="utf-8",
        )
        if os.name == "nt":
            launcher = Path(directory) / "blastn.bat"
            launcher.write_text(
                "@echo off" + chr(13) + chr(10)
                + '"' + sys.executable + '" "' + str(script) + '" %*'
                + chr(13) + chr(10),
                encoding="utf-8")
        else:
            launcher = Path(directory) / "blastn"
            launcher.write_text(
                "#!/bin/sh" + chr(10)
                + 'exec "' + sys.executable + '" "' + str(script) + '" "$@"'
                + chr(10),
                encoding="utf-8")
            launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
        return launcher

    @classmethod
    def setUpClass(cls):
        import random

        from cas13guide.services import blast as blast_module
        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(31)

        def seq(n):
            return "".join(rng.choice("ACGU") for _ in range(n))

        cls._directory = tempfile.TemporaryDirectory()
        home = Path(cls._directory.name)
        eol = chr(10)
        (home / "t.fa").write_text(">TGT1" + eol + seq(300) + eol,
                                   encoding="utf-8")
        rows = []
        for name, biotype in (("RN18S1", "rrna"), ("MT-CO1", "mito"),
                              ("LINC01", "ncrna"), ("PREM1", "pre_mrna")):
            rows += [">" + name + " biotype=" + biotype, seq(300)]
        (home / "bg.fa").write_text(eol.join(rows) + eol, encoding="utf-8")

        launcher = cls._stub_self_hit(home)
        original = blast_module.BlastOffTargetIndex
        blast_module.BlastOffTargetIndex = (
            lambda **kwargs: original(executable=str(launcher), **kwargs)
        )
        try:
            cls.outcome = run_request(DesignRequest(
                fasta_path=str(home / "t.fa"),
                background_path=str(home / "bg.fa"),
                task="knockdown", assay="cell_culture_knockdown",
                effector="RfxCas13d", use_structure=False, use_signal=False,
                blast_db="pretend.nsq", blast_universes="coding",
            ))
        finally:
            blast_module.BlastOffTargetIndex = original

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_the_fixture_really_produced_a_perfect_unmatched_hit(self):
        """Or everything below would hold over a run that had no such hit."""
        perfect = [row for row in self.outcome.report["off_targets"]
                   if row["mismatches"] == 0]
        self.assertTrue(perfect)
        self.assertEqual({row["locus"] for row in perfect}, {"NM_000546.6"})
        self.assertEqual({row["target_class"] for row in perfect},
                         {"unannotated"})

    def test_the_run_names_the_locus_it_cannot_tell_apart(self):
        self.assertIn("NM_000546.6", self.outcome.report["self_identical_loci"])

    def test_the_reader_is_told_before_they_act_on_the_table(self):
        texts = [item["text"] for item in self.outcome.report["limitations"]
                 if item["id"] == "off-target-may-be-the-target"]
        self.assertEqual(len(texts), 1)
        [text] = texts
        self.assertIn("NM_000546.6", text)
        self.assertIn("matches", text)          # one locus, not "match"
        self.assertIn("worst_off_target", text)
        self.assertEqual(
            [item["severity"] for item in self.outcome.report["limitations"]
             if item["id"] == "off-target-may-be-the-target"],
            ["critical"],
        )

    def test_the_hazard_is_left_counted_rather_than_dropped(self):
        """Overstating is the safe direction; hiding a paralog is not."""
        worst = [row["metrics"]["worst_off_target"]
                 for row in self.outcome.report["ranked"]]
        self.assertTrue(worst)
        self.assertGreater(min(worst), 0.0)


if __name__ == "__main__":
    unittest.main()
