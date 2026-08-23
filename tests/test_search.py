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


if __name__ == "__main__":
    unittest.main()
