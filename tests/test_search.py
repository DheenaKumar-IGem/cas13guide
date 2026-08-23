"""Search: the exhaustiveness claim has to mean something."""

import random
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import Universe
from cas13guide.services.indexes import ExhaustiveScanIndex, SeedHeuristicIndex
from cas13guide.services.transcriptome import Locus, Transcriptome
from cas13guide.util.seq import revcomp

SPACER = "GCAUUGCUGUGGAGCUGAAGUC"


def world_with_spread_mismatch_site():
    """A site whose mismatches break *both* terminal seeds."""
    site = list(revcomp(SPACER))
    for i in (3, 18):  # one in each 11nt terminal seed
        site[i] = "C" if site[i] != "C" else "G"
    rng = random.Random(2)
    filler = "".join(rng.choice("AUGC") for _ in range(200))
    t = Transcriptome("GRCh38", "v44")
    t.add(Locus("L1", filler + "".join(site) + filler, Universe.CODING, expression=10.0))
    return t


class TestExhaustiveness(unittest.TestCase):
    def setUp(self):
        self.world = world_with_spread_mismatch_site()
        self.universes = frozenset({Universe.CODING})

    def test_exhaustive_scan_finds_the_spread_mismatch_hit(self):
        idx = ExhaustiveScanIndex(self.world, self.universes)
        result = idx.search("g1", SPACER, 4)
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0].duplex.n_mismatch, 2)
        self.assertTrue(result.bound.exhaustive)

    def test_seed_heuristic_misses_it_and_says_so(self):
        """Why the bound is a claim, not a status: the fast index is wrong here."""
        idx = SeedHeuristicIndex(self.world, self.universes, seed_length=11)
        result = idx.search("g1", SPACER, 4)
        self.assertEqual(len(result.hits), 0)  # genuinely missed
        self.assertFalse(result.bound.exhaustive)  # and honest about it
        self.assertIn("seed", result.bound.note)

    def test_bound_covers_semantics(self):
        idx = ExhaustiveScanIndex(self.world, self.universes)
        bound = idx.search("g1", SPACER, 4).bound
        self.assertTrue(bound.covers(4))
        self.assertFalse(bound.covers(5))  # deeper than what was proven
        self.assertFalse(exhaustive_bound(4, exhaustive=False).covers(4))

    def test_wobble_hits_are_found_and_not_counted_as_mismatches(self):
        site = list(revcomp(SPACER))
        idx_c = site.index("C")
        site[idx_c] = "U"  # C->U: a wobble, still pairs
        t = Transcriptome("GRCh38", "v44")
        t.add(Locus("L1", "".join(site), Universe.CODING, expression=10.0))
        result = ExhaustiveScanIndex(t, frozenset({Universe.CODING})).search("g1", SPACER, 0)
        self.assertEqual(len(result.hits), 1)  # found at zero mismatch budget
        self.assertEqual(result.hits[0].duplex.n_wobble, 1)



class TestAmbiguityBasesAreNotADeadZone(unittest.TestCase):
    """``N`` is a legal base, and no index may quietly skip windows carrying it.

    ``util.seq.VALID_RNA`` admits ``N``, ``RNA_COMPLEMENT`` maps it to itself
    and ``Transcriptome.add`` normalizes it through unchanged, so an assembly
    gap inside an off-target site is ordinary input. The FM walk used to branch
    over a hard-coded ``("A","C","G","U")``, which made every window overlapping
    an ``N`` unreachable -- while the returned bound still said
    ``exhaustive=True``. A real paralog off-target simply vanished and the guide
    read as perfectly specific.
    """

    def _world(self):
        site = revcomp(SPACER)
        gapped = site[:10] + "N" + site[11:]      # an assembly gap in the site
        t = Transcriptome("GRCh38", "v44")
        t.add(Locus("TGT", "AUGCAUGCAU" + site + "AUGCAUGCAU",
                    Universe.CODING, expression=100.0))
        t.add(Locus("PAR", "AUGCAUGCAU" + gapped + "AUGCAUGCAU",
                    Universe.CODING, expression=50.0))
        return t

    @staticmethod
    def _sig(result):
        return sorted((h.locus_id, h.start, h.duplex.n_mismatch) for h in result.hits)

    def test_scanner_sees_the_hit_through_the_gap(self):
        world = self._world()
        hits = self._sig(
            ExhaustiveScanIndex(world, frozenset({Universe.CODING})).search("g", SPACER, 4)
        )
        self.assertIn(("PAR", 10, 1), hits)

    def test_fm_index_agrees_with_the_scanner_over_n(self):
        """The differential claim, on the input that used to break it."""
        from cas13guide.services.fmindex import FMIndexSearchIndex

        world = self._world()
        universes = frozenset({Universe.CODING})
        scan = ExhaustiveScanIndex(world, universes)
        fm = FMIndexSearchIndex(world, universes)
        for k in (0, 1, 2, 4):
            self.assertEqual(
                self._sig(scan.search("g", SPACER, k)),
                self._sig(fm.search("g", SPACER, k)),
                f"FM and scanner disagree over an N-containing locus at k={k}",
            )

    def test_walk_alphabet_is_derived_and_still_excludes_the_separator(self):
        from cas13guide.services.fmindex import SEPARATOR, TERMINATOR, walk_alphabet

        self.assertEqual(
            walk_alphabet("AUGC" + SEPARATOR + "AUGC" + TERMINATOR),
            ("A", "C", "G", "U"),
        )
        self.assertIn("N", walk_alphabet("AUGCN" + SEPARATOR + TERMINATOR))
        # The first of the two locus-boundary defenses must survive derivation.
        text = "AUG" + SEPARATOR + "C" + TERMINATOR
        self.assertNotIn(SEPARATOR, walk_alphabet(text))
        self.assertNotIn(TERMINATOR, walk_alphabet(text))


class TestVectorizedScanMatchesTheReference(unittest.TestCase):
    """The scan is vectorized; its hit set must be identical to the naive one.

    This index is the one allowed to claim ``exhaustive=True``, so optimizing it
    is only admissible if it provably changes nothing. The reference below is
    the original per-character implementation, kept verbatim as an oracle.
    """

    @staticmethod
    def _reference(expected, sequence, k):
        from cas13guide.util.duplex import WOBBLE_SUBSTITUTIONS

        length = len(expected)
        out = []
        for start in range(len(sequence) - length + 1):
            window = sequence[start : start + length]
            mismatches = 0
            for exp_b, obs_b in zip(expected, window):
                if exp_b == obs_b or (exp_b, obs_b) in WOBBLE_SUBSTITUTIONS:
                    continue
                mismatches += 1
                if mismatches > k:
                    break
            else:
                out.append((start, window, mismatches))
        return out

    def test_randomized_equivalence_including_ambiguity_and_short_loci(self):
        from cas13guide.services.indexes import _scan_locus

        rng = random.Random(20260823)
        for _ in range(600):
            alphabet = rng.choice(["AUGC", "AUGCN", "AU", "A", "GC"])
            sequence = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 70)))
            expected = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
            k = rng.randint(0, 6)
            self.assertEqual(
                list(_scan_locus(expected, sequence, k)),
                self._reference(expected, sequence, k),
                f"scan diverged: expected={expected!r} seq={sequence!r} k={k}",
            )

    def test_prune_point_cannot_change_the_hit_set(self):
        """Compaction is an optimization; every prune point must agree.

        ``cost`` only ever increases, so a window dropped at any compaction had
        already blown the budget. Pinning this stops a future retune of
        ``_prune_after`` from silently losing hits.
        """
        import cas13guide.services.indexes as indexes

        rng = random.Random(5)
        sequence = "".join(rng.choice("AUGCN") for _ in range(3000))
        expected = revcomp(SPACER)
        encoded = indexes._encode(sequence)
        original = indexes._prune_after
        try:
            baseline = None
            for point in (1, 2, 4, 8, 12, 20, 40):
                indexes._prune_after = lambda _k, _p=point: _p
                out = list(indexes._scan_encoded(encoded, sequence, expected, 4))
                if baseline is None:
                    baseline = out
                self.assertEqual(out, baseline, f"prune point {point} changed the hit set")
        finally:
            indexes._prune_after = original


class TestExistenceQueryMatchesTheFullSearch(unittest.TestCase):
    """``any_hit`` is an optimization for control generation, not a new rule."""

    def _worlds(self):
        rng = random.Random(31)
        for _ in range(12):
            t = Transcriptome("GRCh38", "v44")
            for i in range(2):
                seq = "".join(rng.choice("AUGCN") for _ in range(rng.randint(40, 200)))
                t.add(Locus(f"L{i}", seq, Universe.CODING, expression=1.0))
            yield t

    def test_agrees_with_bool_of_search_on_both_indexes(self):
        from cas13guide.services.fmindex import FMIndexSearchIndex

        universes = frozenset({Universe.CODING})
        rng = random.Random(77)
        for world in self._worlds():
            spacer = "".join(rng.choice("AUGC") for _ in range(22))
            for index in (
                ExhaustiveScanIndex(world, universes),
                FMIndexSearchIndex(world, universes),
            ):
                for k in (0, 2, 4):
                    self.assertEqual(
                        index.any_hit(spacer, k),
                        bool(index.search("g", spacer, k).hits),
                        f"{type(index).__name__} any_hit disagreed at k={k}",
                    )


class TestBoundFoldsToTheWeakest(unittest.TestCase):
    """``run_search`` must keep the weakest bound, not the last one written."""

    def test_a_late_exhaustive_bound_cannot_overwrite_an_early_heuristic_one(self):
        from cas13guide.models.evidence import Candidate
        from cas13guide.pipeline.c_search import run_search
        from cas13guide.services.indexes import SearchResult

        spec = make_spec()

        class Fickle:
            """Honest per-query bounds: heuristic for the first spacer only."""

            version = "index/fickle/v1"

            def __init__(self):
                self.seen = 0

            def search(self, guide_id, spacer, max_mismatches):
                self.seen += 1
                return SearchResult(
                    hits=(),
                    bound=exhaustive_bound(max_mismatches, exhaustive=self.seen > 1),
                )

        candidates = (
            Candidate(guide_id="g1", spacer=SPACER, source_target_id="t1", start=0),
            Candidate(guide_id="g2", spacer=SPACER, source_target_id="t1", start=1),
        )
        outcome = run_search(candidates, spec, Fickle(), {}, max_mismatches=4)
        self.assertFalse(
            outcome.bound.exhaustive,
            "a later exhaustive bound overwrote an earlier heuristic one",
        )




class TestConcatenatedScanKeepsLocusBoundaries(unittest.TestCase):
    """The scanner scans one concatenated array; boundaries must still hold.

    Scanning locus by locus made a cross-locus hit structurally impossible.
    Scanning a single concatenated array makes it possible, and a cross-locus
    hit is an *invented* off-target reported as a real one -- so the boundary
    handling gets the same belt-and-braces the FM-index has, and each defense
    is pinned separately.
    """

    UNIVERSES = frozenset({Universe.CODING, Universe.NCRNA})

    def _per_locus_reference(self, world, spacer, k):
        """What locus-by-locus scanning produced, in its original order."""
        from cas13guide.services.indexes import _scan_locus

        expected = revcomp(spacer)
        out = []
        for locus in world.in_universes(self.UNIVERSES):
            for start, window, n in _scan_locus(expected, locus.sequence, k):
                out.append((locus.locus_id, start, window, n))
        return out

    def test_a_pattern_that_exists_only_across_a_junction_is_never_reported(self):
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("A1", "AUGCAUGCAUG", Universe.CODING, expression=1.0))
        world.add(Locus("A2", "CAUGCAUGCAU", Universe.CODING, expression=1.0))
        index = ExhaustiveScanIndex(world, self.UNIVERSES)
        spanning = "AUGCAUGCAUGCAUGCAUGCAU"       # 22nt, only across A1+A2
        # Budgets far past the one mismatch a separator costs, so the separator
        # alone cannot be what rejects these.
        for k in (0, 4, 8, 12):
            self.assertEqual(
                index.search("g", revcomp(spanning), k).hits, (),
                f"invented a cross-locus off-target at k={k}",
            )

    def test_the_separator_is_not_a_base(self):
        """First defense: the joiner pairs with nothing."""
        from cas13guide.services.indexes import _CORPUS_SEPARATOR, _allowed_bases

        for base in "AUGCN":
            self.assertNotIn(_CORPUS_SEPARATOR, _allowed_bases(base))

    def test_the_span_check_rejects_a_window_running_past_its_locus(self):
        """Second defense: the exact one, independent of the separator."""
        import numpy as np

        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("A1", "AUGCAUGCAUG", Universe.CODING, expression=1.0))
        world.add(Locus("A2", "CAUGCAUGCAU", Universe.CODING, expression=1.0))
        index = ExhaustiveScanIndex(world, self.UNIVERSES)
        index._built()
        # offset 0 is inside A1 (11nt): an 11nt window fits, a 22nt one cannot.
        kept, _ = index._within_locus(np.array([0], dtype=np.int64), 11)
        self.assertEqual(kept.size, 1)
        kept, _ = index._within_locus(np.array([0], dtype=np.int64), 22)
        self.assertEqual(kept.size, 0)
        # a start inside A1 whose window would reach into A2
        kept, _ = index._within_locus(np.array([5], dtype=np.int64), 11)
        self.assertEqual(kept.size, 0)

    def test_randomized_equivalence_with_locus_by_locus_scanning(self):
        """Same hits, same order, including loci shorter than the pattern."""
        rng = random.Random(90210)
        for _ in range(120):
            world = Transcriptome("GRCh38", "v44")
            for i in range(rng.randint(1, 5)):
                size = rng.choice([rng.randint(1, 21), rng.randint(22, 60), rng.randint(60, 160)])
                alphabet = "AUGCN" if rng.random() < 0.5 else "AUGC"
                world.add(Locus(
                    f"L{i}", "".join(rng.choice(alphabet) for _ in range(size)),
                    Universe.CODING if i % 2 == 0 else Universe.NCRNA, expression=1.0,
                ))
            loci = world.in_universes(self.UNIVERSES)
            if not loci:
                continue
            concatenated = "".join(locus.sequence for locus in loci)
            if len(concatenated) >= 22 and rng.random() < 0.6:
                # a spacer drawn across the junction: maximum boundary pressure
                start = rng.randrange(0, len(concatenated) - 22 + 1)
                spacer = revcomp(concatenated[start : start + 22])
            else:
                spacer = "".join(rng.choice("AUGC") for _ in range(22))
            index = ExhaustiveScanIndex(world, self.UNIVERSES)
            for k in (0, 2, 4, 6):
                got = [
                    (h.locus_id, h.start, h.target_site, h.duplex.n_mismatch)
                    for h in index.search("g", spacer, k).hits
                ]
                self.assertEqual(got, self._per_locus_reference(world, spacer, k))

    def test_loci_shorter_than_the_spacer_contribute_nothing(self):
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("TINY", "AUGC", Universe.CODING, expression=1.0))
        world.add(Locus("ALSO", "AUG", Universe.CODING, expression=1.0))
        index = ExhaustiveScanIndex(world, self.UNIVERSES)
        self.assertEqual(index.search("g", SPACER, 6).hits, ())
        self.assertFalse(index.any_hit(SPACER, 6))



if __name__ == "__main__":
    unittest.main()
