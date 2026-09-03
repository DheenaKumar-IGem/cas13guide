"""Index choice is a performance decision -- and it must be the right one."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import Universe
from cas13guide.services.fmindex import FMIndexSearchIndex
from cas13guide.services.index_selection import (
    BUILD_SECONDS_PER_MB,
    CROSSOVER_BASES,
    FM,
    FM_SECONDS_PER_QUERY,
    SCAN_SECONDS_PER_MB_PER_QUERY,
    SCANNER,
    choose_index,
    estimate_seconds,
    plan_index,
    searchable_bases,
)
from cas13guide.services.indexes import ExhaustiveScanIndex
from cas13guide.services.transcriptome import Locus, Transcriptome

UNIVERSES = frozenset({Universe.CODING})


def corpus(total_bases: int, per_locus: int = 50_000) -> Transcriptome:
    t = Transcriptome("GRCh38", "v44")
    n = max(1, total_bases // per_locus)
    size = total_bases // n
    for i in range(n):
        t.add(Locus(f"L{i}", "AUGC" * (size // 4), Universe.CODING, expression=1.0))
    return t


class TestSearchableBases(unittest.TestCase):
    def test_counts_only_declared_universes(self):
        t = Transcriptome("GRCh38", "v44")
        t.add(Locus("C1", "AUGC" * 100, Universe.CODING, expression=1.0))
        t.add(Locus("N1", "AUGC" * 100, Universe.NCRNA, expression=1.0))
        self.assertEqual(searchable_bases(t, UNIVERSES), 400)
        self.assertEqual(
            searchable_bases(t, frozenset({Universe.CODING, Universe.NCRNA})), 800
        )


class TestCostModel(unittest.TestCase):
    """The policy is a pure cost model -- no suffix array is built here."""

    def test_small_corpus_uses_the_scanner(self):
        plan = plan_index(100_000, expected_queries=1000)
        self.assertEqual(plan.kind, SCANNER)
        self.assertIn("does not pay back", plan.rationale)

    def test_transcriptome_scale_uses_the_fm_index(self):
        plan = plan_index(300_000_000, expected_queries=1000)
        self.assertEqual(plan.kind, FM)
        self.assertIn("amortizes", plan.rationale)

    @staticmethod
    def _break_even_queries(bases: int) -> float:
        """Queries at which FM total == scanner total, from the constants.

        ``q * s * M == b * M + q * f``  =>  ``q = b*M / (s*M - f)``. Only finite
        above :data:`CROSSOVER_BASES`, where the scanner's per-query cost finally
        exceeds the FM-index's.
        """
        mb = bases / 1_000_000
        return BUILD_SECONDS_PER_MB * mb / (SCAN_SECONDS_PER_MB_PER_QUERY * mb - FM_SECONDS_PER_QUERY)

    def test_query_volume_flips_the_decision_at_fixed_corpus_size(self):
        """Corpus size alone is not enough -- build cost has to pay back.

        Stated in terms of the derived crossover rather than a hard-coded
        corpus size: the old form pinned 3 Mb, which stopped being a corpus
        where the flip happens once the scanner was vectorized, so the test
        failed for a reason that had nothing to do with the policy it guards.
        """
        bases = int(CROSSOVER_BASES * 1.7)          # comfortably past crossover
        break_even = self._break_even_queries(bases)
        self.assertGreater(break_even, 1.0)
        self.assertEqual(plan_index(bases, expected_queries=1).kind, SCANNER)
        self.assertEqual(
            plan_index(bases, expected_queries=int(break_even * 4)).kind, FM
        )

    def test_below_the_crossover_no_query_volume_can_justify_the_fm_index(self):
        """Under the crossover the scanner is cheaper *per query*, so build never pays."""
        bases = int(CROSSOVER_BASES * 0.5)
        for queries in (1, 100, 10_000, 1_000_000):
            self.assertEqual(plan_index(bases, expected_queries=queries).kind, SCANNER)

    def test_model_reproduces_its_own_amortization_point(self):
        """At the derived break-even the two totals meet, by construction."""
        bases = int(CROSSOVER_BASES * 1.7)
        queries = self._break_even_queries(bases)
        scanner, fm = estimate_seconds(bases, queries)
        self.assertAlmostEqual(scanner, fm, delta=0.01 * scanner)
        # and the decision really does turn there
        self.assertEqual(plan_index(bases, int(queries * 0.5)).kind, SCANNER)
        self.assertEqual(plan_index(bases, int(queries * 2)).kind, FM)

    def test_scan_cost_is_linear_in_corpus_size(self):
        a, _ = estimate_seconds(1_000_000, 100)
        b, _ = estimate_seconds(2_000_000, 100)
        self.assertAlmostEqual(b, 2 * a, delta=1e-6)

    def test_fm_query_cost_does_not_grow_with_corpus_size(self):
        """A property of the *model*, not of the FM index.

        The real per-query cost is flat only above ~1.5 Mb; below that it
        tracks the corpus by two orders of magnitude. Reading this test as a
        statement about the index is the confusion that let plan_index print
        "FM ~2800s" for a 2 kb corpus. See FM_FLAT_ABOVE_BASES.
        """
        _, small = estimate_seconds(1_000_000, 100)
        _, large = estimate_seconds(2_000_000, 100)
        # Only the build term differs; the per-query term is flat.
        self.assertAlmostEqual(
            large - small, BUILD_SECONDS_PER_MB * 1.0, delta=1e-6
        )

    def test_degenerate_inputs_fall_back_to_the_scanner(self):
        for bases, queries in ((0, 1000), (100_000, 0), (-5, 10)):
            self.assertEqual(plan_index(bases, queries).kind, SCANNER)

    def test_rationale_is_always_populated(self):
        for size in (0, 1_000, CROSSOVER_BASES, CROSSOVER_BASES * 100):
            self.assertTrue(plan_index(size).rationale)

    def test_derived_crossover_matches_the_per_query_costs(self):
        """CROSSOVER_BASES is derived from the constants, not independent."""
        mb = CROSSOVER_BASES / 1_000_000
        self.assertAlmostEqual(
            SCAN_SECONDS_PER_MB_PER_QUERY * mb, FM_SECONDS_PER_QUERY, delta=1e-6
        )


class TestConstruction(unittest.TestCase):
    def test_choose_index_builds_what_the_plan_says(self):
        choice = choose_index(corpus(20_000), UNIVERSES)
        self.assertEqual(choice.kind, SCANNER)
        self.assertIsInstance(choice.index, ExhaustiveScanIndex)
        self.assertEqual(choice.searchable_bases, 20_000)
        self.assertTrue(choice.rationale)


class TestChoiceCannotChangeResults(unittest.TestCase):
    def test_both_indexes_are_exhaustive_so_the_choice_is_speed_only(self):
        """The selector must never trade correctness for speed."""
        small = corpus(20_000)
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        scan = ExhaustiveScanIndex(small, UNIVERSES)
        fm = FMIndexSearchIndex(small, UNIVERSES)

        scan_result = scan.search("g", spacer, 4)
        fm_result = fm.search("g", spacer, 4)

        self.assertTrue(scan_result.bound.exhaustive)
        self.assertTrue(fm_result.bound.exhaustive)
        sig = lambda r: {(h.locus_id, h.start, h.duplex.n_mismatch) for h in r.hits}
        self.assertEqual(sig(scan_result), sig(fm_result))


class TestTheRationaleDoesNotQuoteAnUnmodelledNumber(unittest.TestCase):
    """The printed counterfactual was wrong by about a hundredfold.

    ``FM_SECONDS_PER_QUERY`` is documented as "effectively flat in corpus
    size", which is true above roughly a megabyte and a half and false below
    it -- this module's own docstring table says so, with 0.742 s/query at
    600 kb against 2.758 at 3 Mb. Measured on the same machine further down:
    0.027 s/query at 2 kb, 0.163 at 20 kb, 0.858 at 200 kb.

    ``plan_index`` printed that flat constant extrapolated down anyway. At the
    shipped demo's 2,391 bases it said "FM ~2800s" where the real cost is about
    27 seconds. The string is provenance -- stdout, the HTML report, and the
    ledger's ``index_choice`` -- so it was a measured-sounding claim about a
    counterfactual nobody had measured.

    The decision was never wrong, and these tests pin that too: the crossover
    is near 87 Mb, deep in the flat regime, and below it the scanner wins under
    either figure. Only the report is fixed.
    """

    def test_a_small_corpus_gets_no_fm_seconds_figure(self):
        plan = plan_index(2_391, expected_queries=1000)
        self.assertEqual(plan.kind, SCANNER)
        self.assertNotIn("FM ~", plan.rationale)

    def test_it_says_why_instead(self):
        """A refusal to quote a number is only useful if it says what it means."""
        rationale = plan_index(2_391, expected_queries=1000).rationale
        self.assertIn("suffix-array build does not pay back", rationale)
        self.assertIn("only modelled above", rationale)
        self.assertIn("1.5 Mb", rationale)

    def test_a_large_corpus_still_reports_both_numbers(self):
        """Above the calibration floor the estimate is a measurement, and the
        comparison is the whole point of the model."""
        plan = plan_index(3_000_000, expected_queries=1000)
        self.assertEqual(plan.kind, SCANNER)
        self.assertIn("FM ~", plan.rationale)
        self.assertIn("beats FM", plan.rationale)

    def test_the_choice_itself_is_unchanged_on_both_sides(self):
        """This was a reporting fix. If it moved a decision it was the wrong
        fix, and that is worth a test rather than an assurance."""
        for bases in (0, 1, 2_391, 100_000, 1_499_999, 1_500_000, 3_000_000,
                      50_000_000):
            with self.subTest(bases=bases):
                scanner_seconds, fm_seconds = estimate_seconds(bases, 1000)
                expected = FM if fm_seconds < scanner_seconds else SCANNER
                if bases <= 0:
                    expected = SCANNER
                self.assertEqual(plan_index(bases, 1000).kind, expected)

    def test_the_crossover_is_far_above_the_calibration_floor(self):
        """Why leaving the flat constant in the decision is safe: the only
        place the two costs are close is deep inside the regime where the
        constant was measured."""
        from cas13guide.services.index_selection import FM_FLAT_ABOVE_BASES

        self.assertGreater(CROSSOVER_BASES, 10 * FM_FLAT_ABOVE_BASES)

    def test_the_model_records_where_it_is_valid(self):
        """The constant drifted because its validity range lived in prose that
        described a different regime from the one the code applied it to. The
        measurements that bound it belong beside the model."""
        import inspect

        from cas13guide.services import index_selection

        self.assertTrue(hasattr(index_selection, "FM_FLAT_ABOVE_BASES"))
        doc = inspect.getdoc(index_selection) or ""
        self.assertIn("above about 1.5 Mb", doc)
        # the small-corpus rows that show it is not flat down there
        for row in ("2 kb", "20 kb", "200 kb"):
            with self.subTest(row=row):
                self.assertIn(row, doc)


if __name__ == "__main__":
    unittest.main()
