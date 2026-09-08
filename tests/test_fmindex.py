"""FM-index: correct by construction, and identical to the reference scanner."""

import random
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import Universe
from cas13guide.services.fmindex import FMIndex, FMIndexSearchIndex
from cas13guide.services.fmindex import SEPARATOR, TERMINATOR
from cas13guide.services.indexes import ExhaustiveScanIndex
from cas13guide.services.transcriptome import Locus, Transcriptome
from cas13guide.util.seq import revcomp

UNIVERSES = frozenset({Universe.CODING, Universe.NCRNA})


def random_rna(n, rng):
    return "".join(rng.choice("AUGC") for _ in range(n))


def random_world(rng, n_loci=4, size=220):
    t = Transcriptome("GRCh38", "v44")
    for i in range(n_loci):
        t.add(
            Locus(
                f"L{i}",
                random_rna(rng.randint(size // 2, size), rng),
                Universe.CODING if i % 2 == 0 else Universe.NCRNA,
                gene=f"G{i}",
                expression=10.0 * (i + 1),
            )
        )
    return t


def signature(result):
    """Compare hit *sets*, independent of the order an index emits them."""
    return {
        (h.locus_id, h.start, h.target_site, h.duplex.n_mismatch, h.duplex.n_wobble)
        for h in result.hits
    }


class TestRankAndExtend(unittest.TestCase):
    def test_rank_is_exact_at_every_offset(self):
        rng = random.Random(3)
        text = random_rna(300, rng) + SEPARATOR + random_rna(200, rng) + TERMINATOR
        for checkpoint in (1, 7, 64, 512):
            fm = FMIndex(text, checkpoint=checkpoint)
            for c in "AUGC":
                for i in (0, 1, 17, 128, len(text) // 2, len(text)):
                    self.assertEqual(fm.rank(c, i), fm.bwt[:i].count(c), (c, i, checkpoint))

    def test_backward_extension_finds_exact_occurrences(self):
        rng = random.Random(11)
        text = random_rna(400, rng) + "$"
        fm = FMIndex(text)
        pattern = text[100:112]
        lo, hi = fm.full_interval()
        for ch in reversed(pattern):
            lo, hi = fm.extend_backward(ch, lo, hi)
        found = sorted(fm.sa[lo:hi])
        expected = sorted(
            i for i in range(len(text) - len(pattern) + 1)
            if text[i : i + len(pattern)] == pattern
        )
        self.assertEqual(found, expected)


class TestDifferentialAgainstReferenceScanner(unittest.TestCase):
    """The FM-index must return *exactly* the reference scanner's hit set.

    This is what lets it claim ``exhaustive=True``. A faster index that quietly
    drops hits is the single most dangerous failure in this system: it produces
    a clean off-target report rather than an error.
    """

    def test_randomized_equivalence(self):
        rng = random.Random(2024)
        for trial in range(25):
            world = random_world(rng)
            fm = FMIndexSearchIndex(world, UNIVERSES)
            scan = ExhaustiveScanIndex(world, UNIVERSES)

            locus = rng.choice(list(world.loci.values()))
            start = rng.randrange(0, len(locus.sequence) - 22)
            spacer = revcomp(locus.sequence[start : start + 22])

            for k in (0, 1, 2, 3, 4):
                self.assertEqual(
                    signature(fm.search("g", spacer, k)),
                    signature(scan.search("g", spacer, k)),
                    f"trial={trial} k={k}",
                )

    def test_equivalence_on_random_spacers_not_present_in_the_text(self):
        rng = random.Random(99)
        for _ in range(15):
            world = random_world(rng)
            fm = FMIndexSearchIndex(world, UNIVERSES)
            scan = ExhaustiveScanIndex(world, UNIVERSES)
            spacer = random_rna(22, rng)
            for k in (2, 4, 6):
                self.assertEqual(
                    signature(fm.search("g", spacer, k)),
                    signature(scan.search("g", spacer, k)),
                )

    def test_equivalence_on_repetitive_sequence(self):
        """Homopolymers and tandem repeats are where index bugs surface."""
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("R1", "AUAUAUAUAUAUAUAUAUAUAUAUAUAUAUAU", Universe.CODING, expression=5.0))
        world.add(Locus("R2", "GGGGGGGGGGGGGGGGGGGGGGGGGGGG", Universe.CODING, expression=5.0))
        world.add(Locus("R3", "ACGUACGUACGUACGUACGUACGUACGU", Universe.NCRNA, expression=5.0))
        fm = FMIndexSearchIndex(world, UNIVERSES)
        scan = ExhaustiveScanIndex(world, UNIVERSES)
        for spacer in ("AUAUAUAUAUAUAUAUAUAUAU", "CCCCCCCCCCCCCCCCCCCCCC", "ACGUACGUACGUACGUACGUAC"):
            for k in (0, 2, 4):
                self.assertEqual(
                    signature(fm.search("g", spacer, k)),
                    signature(scan.search("g", spacer, k)),
                    (spacer, k),
                )

    def test_wobble_hits_are_found_at_zero_mismatch_budget(self):
        world = Transcriptome("GRCh38", "v44")
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        site = list(revcomp(spacer))
        site[site.index("C")] = "U"  # C->U wobble: still pairs
        world.add(Locus("W1", "AUGCAUGC" + "".join(site) + "AUGCAUGC", Universe.CODING, expression=5.0))
        result = FMIndexSearchIndex(world, UNIVERSES).search("g", spacer, 0)
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0].duplex.n_wobble, 1)
        self.assertEqual(result.hits[0].duplex.n_mismatch, 0)


class TestBoundaryDefencesIndividually(unittest.TestCase):
    """Two independent mechanisms stop cross-locus matches.

    Removing either alone is masked by the other, so each is pinned here --
    otherwise a refactor could quietly delete one and leave the system one
    edit away from inventing off-targets.
    """

    def test_walk_alphabet_excludes_the_separator(self):
        """The alphabet the walk actually iterates, not the module constant.

        This asserted `SEPARATOR not in BASES` and `set(BASES) == set("ACGU")`.
        `BASES` is used nowhere in src/ outside its own definition -- fmindex
        itself says "the alphabet actually walked is derived per index by
        walk_alphabet" -- so the assertion pinned a dead constant, and
        replacing walk_alphabet with one that returns the separator left this
        class reporting OK while the walk could build matches straddling two
        loci.
        """
        from cas13guide.services.fmindex import (
            SEPARATOR,
            TERMINATOR,
            FMIndexSearchIndex,
            walk_alphabet,
        )

        text = "AUG" + SEPARATOR + "CGN" + TERMINATOR
        self.assertNotIn(SEPARATOR, walk_alphabet(text))
        self.assertNotIn(TERMINATOR, walk_alphabet(text))

        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("A1", "AUGCAUGCAUG", Universe.CODING, expression=1.0))
        world.add(Locus("A2", "CAUGCAUGCAU", Universe.CODING, expression=1.0))
        index = FMIndexSearchIndex(world, UNIVERSES)
        self.assertNotIn(SEPARATOR, index.walk_alphabet)
        self.assertNotIn(TERMINATOR, index.walk_alphabet)

    def test_locate_rejects_a_span_running_off_the_end_of_its_locus(self):
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("A1", "AUGCAUGCAUG", Universe.CODING, expression=1.0))
        world.add(Locus("A2", "CAUGCAUGCAU", Universe.CODING, expression=1.0))
        idx = FMIndexSearchIndex(world, UNIVERSES)
        # Offset 0 is inside A1 (11nt); a 22nt span cannot fit.
        self.assertIsNone(idx._locate(0, 22))
        self.assertIsNotNone(idx._locate(0, 11))
        # A start inside A1 with a length that would reach into A2.
        self.assertIsNone(idx._locate(5, 11))


class TestBoundaries(unittest.TestCase):
    def test_matches_never_span_two_loci(self):
        """The separator must be uncrossable, whatever the mismatch budget."""
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("A1", "AUGCAUGCAUG", Universe.CODING, expression=1.0))
        world.add(Locus("A2", "CAUGCAUGCAU", Universe.CODING, expression=1.0))
        fm = FMIndexSearchIndex(world, UNIVERSES)
        scan = ExhaustiveScanIndex(world, UNIVERSES)
        # A 22-mer cannot fit in either 11nt locus, so nothing may be returned
        # even though the concatenated text is long enough.
        for k in (0, 4, 8):
            self.assertEqual(fm.search("g", "AUGCAUGCAUGCAUGCAUGCAU", k).hits, ())
            self.assertEqual(signature(fm.search("g", "AUGCAUGCAUGCAUGCAUGCAU", k)),
                             signature(scan.search("g", "AUGCAUGCAUGCAUGCAUGCAU", k)))

    def test_hit_at_the_very_start_and_end_of_a_locus(self):
        seq = "GCAUUGCUGUGGAGCUGAAGUC"
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("E1", seq, Universe.CODING, expression=1.0))
        fm = FMIndexSearchIndex(world, UNIVERSES)
        [hit] = fm.search("g", revcomp(seq), 0).hits
        self.assertEqual(hit.start, 0)
        self.assertIsNone(hit.five_prime_flank)
        self.assertIsNone(hit.three_prime_flank)

    def test_empty_universe_yields_no_hits_and_a_valid_bound(self):
        world = Transcriptome("GRCh38", "v44")
        fm = FMIndexSearchIndex(world, UNIVERSES)
        result = fm.search("g", "GCAUUGCUGUGGAGCUGAAGUC", 4)
        self.assertEqual(result.hits, ())
        self.assertTrue(result.bound.exhaustive)

    def test_the_bound_names_the_universes_it_holds_not_the_ones_it_was_offered(self):
        """A filter is what an index will admit; a bound is what it searched.

        Reporting the filter is how the universe-adequacy gate was defeated for
        the life of the CLI: the bound named every universe the caller was
        willing to include, so `(required | declared) - searched` cancelled and
        a corpus of one biotype certified a search over nine. It is also what
        a reader is told -- "Off-targets were searched over ..." is this set.
        """
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("C1", random_rna(200, random.Random(1)), Universe.CODING,
                        expression=1.0))
        # The filter admits ncrna as well; the corpus has none.
        fm = FMIndexSearchIndex(world, UNIVERSES)
        bound = fm.search("g", "GCAUUGCUGUGGAGCUGAAGUC", 4).bound
        self.assertEqual(bound.universes, frozenset({Universe.CODING}))
        self.assertIn("coding", bound.describe())
        self.assertNotIn("ncrna", bound.describe())

    def test_the_bound_reports_the_corpus_it_could_have_searched(self):
        """So the gate can tell a skipped universe from an absent one."""
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("C1", random_rna(200, random.Random(2)), Universe.CODING,
                        expression=1.0))
        world.add(Locus("R1", random_rna(200, random.Random(3)), Universe.RRNA,
                        expression=9000.0))
        # rRNA is in the corpus and outside the filter: skipped, not absent.
        fm = FMIndexSearchIndex(world, UNIVERSES)
        bound = fm.search("g", "GCAUUGCUGUGGAGCUGAAGUC", 4).bound
        self.assertEqual(bound.universes, frozenset({Universe.CODING}))
        self.assertEqual(
            bound.corpus_universes, frozenset({Universe.CODING, Universe.RRNA})
        )

    def test_bound_claims_exhaustiveness(self):
        rng = random.Random(7)
        fm = FMIndexSearchIndex(random_world(rng), UNIVERSES)
        bound = fm.search("g", "GCAUUGCUGUGGAGCUGAAGUC", 4).bound
        self.assertTrue(bound.exhaustive)
        self.assertTrue(bound.covers(4))
        self.assertFalse(bound.covers(5))


if __name__ == "__main__":
    unittest.main()
