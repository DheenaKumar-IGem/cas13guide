"""Duplex algebra: the G-U wobble rule is asymmetric, and that matters."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.util.duplex import (
    TERMINAL_WEIGHT,
    PairKind,
    classify_pair,
    duplex_penalty,
    profile_duplex,
    seed_weight,
)
from cas13guide.util.seq import revcomp


class TestWobble(unittest.TestCase):
    def test_perfect_duplex(self):
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        p = profile_duplex(spacer, revcomp(spacer))
        self.assertTrue(p.is_perfect)
        self.assertEqual(p.cigar, "22WC")

    def test_wobble_substitutions_still_pair(self):
        # In target space only C->U and A->G preserve pairing (G-U wobble).
        self.assertIs(classify_pair("C", "U"), PairKind.WOBBLE)
        self.assertIs(classify_pair("A", "G"), PairKind.WOBBLE)

    def test_wobble_is_asymmetric(self):
        # The reverse substitutions are true mismatches, not wobbles.
        # Treating them as symmetric would silently inflate specificity.
        self.assertIs(classify_pair("U", "C"), PairKind.MISMATCH)
        self.assertIs(classify_pair("G", "A"), PairKind.MISMATCH)

    def test_wobble_not_counted_as_mismatch(self):
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        expected = list(revcomp(spacer))
        idx = expected.index("C")
        expected[idx] = "U"
        p = profile_duplex(spacer, "".join(expected))
        self.assertEqual(p.n_mismatch, 0)
        self.assertEqual(p.n_wobble, 1)

    def test_spacer_coordinate_convention_is_pinned(self):
        """Target index k maps to 1-based spacer position L-k, not k+1.

        Both conventions look plausible and most assertions pass under either,
        so this pins the one the seed model is defined in.
        """
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        site = list(revcomp(spacer))
        site[0] = "C" if site[0] != "C" else "G"  # mutate the FIRST target base
        p = profile_duplex(spacer, "".join(site))
        self.assertEqual(p.mismatch_positions, (22,))  # LAST spacer position

    def test_cigar_is_written_in_spacer_five_to_three_order(self):
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        site = list(revcomp(spacer))
        site[0] = "C" if site[0] != "C" else "G"
        # Target index 0 is the spacer's 3' end, so the mismatch comes last.
        self.assertEqual(profile_duplex(spacer, "".join(site)).cigar, "21WC1MM")

    def test_both_flanks_taper_to_the_same_terminal_weight(self):
        """Each flank is normalized by its own width, not by the wider one.

        The seed span is deliberately asymmetric (the Cas13 seed sits slightly
        3'-shifted), so the two termini are not bit-identical -- but they must
        land on the same floor. Normalizing both tapers by the wider flank
        made a 3' mismatch ~1.7x cheaper than a 5' one, which silently biased
        every activity prediction and left the terminal floor unreachable.
        """
        for length in (22, 28, 30):
            first = seed_weight(1, length)
            last = seed_weight(length, length)
            self.assertAlmostEqual(first, last, delta=0.05)
            # Both are near the floor, and neither dips below it.
            for w in (first, last):
                self.assertGreaterEqual(w, TERMINAL_WEIGHT)
                self.assertLess(w, TERMINAL_WEIGHT + 0.10)

    def test_taper_converges_to_the_terminal_weight(self):
        """The floor is approached as the terminus is approached.

        Position 1 sits at fraction ``0.5/L``, not 0, so it lands just above
        the floor -- and closer to it the longer the spacer.
        """
        for length in (22, 100, 1000):
            self.assertAlmostEqual(seed_weight(1, length), TERMINAL_WEIGHT, delta=0.06)
            self.assertAlmostEqual(seed_weight(length, length), TERMINAL_WEIGHT, delta=0.08)
        self.assertLess(seed_weight(1, 1000), seed_weight(1, 22))  # converging

    def test_weight_is_monotone_from_terminus_into_the_seed(self):
        length = 30
        rising = [seed_weight(p, length) for p in range(1, 12)]
        self.assertEqual(rising, sorted(rising))
        self.assertAlmostEqual(max(rising), 1.0)

    def test_central_mismatch_costs_more_than_terminal(self):
        length = 22
        central = seed_weight(11, length)
        terminal = seed_weight(1, length)
        self.assertGreater(central, terminal)

    def test_penalty_orders_by_position(self):
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        expected = list(revcomp(spacer))

        def mutate(target_index):
            seq = list(expected)
            seq[target_index] = "C" if seq[target_index] != "C" else "G"
            return duplex_penalty(profile_duplex(spacer, "".join(seq)))

        # Target index 11 sits in the spacer's central seed; index 0 is terminal.
        self.assertGreater(mutate(11), mutate(0))



class TestDegenerateInput(unittest.TestCase):
    """A zero-length duplex is vacuous, not a crash in the CIGAR builder."""

    def test_empty_duplex_is_vacuously_perfect(self):
        p = profile_duplex("", "")
        self.assertEqual(p.length, 0)
        self.assertEqual(p.cigar, "")
        self.assertEqual(p.mismatch_positions, ())
        self.assertEqual(p.wobble_positions, ())
        self.assertTrue(p.is_perfect)

    def test_length_one_still_reports_the_wobble_asymmetry(self):
        """The boundary case either side of the empty guard."""
        self.assertEqual(profile_duplex("A", "U").cigar, "1WC")
        # spacer G pairs with target U as a wobble (expected C, observed U)
        self.assertEqual(profile_duplex("G", "U").wobble_positions, (1,))
        self.assertEqual(profile_duplex("G", "U").cigar, "1GU")
        # the reverse is a true mismatch
        self.assertEqual(profile_duplex("U", "U").mismatch_positions, (1,))

    def test_mismatched_lengths_still_raise_a_named_error(self):
        with self.assertRaises(ValueError):
            profile_duplex("AUG", "AU")


class TestSeedSpanHasOneDefinition(unittest.TestCase):
    """The seed window is shared with the feature layout and must not drift.

    ``models/features.py`` derives its ``gc_seed`` window from the same span
    ``seed_weight`` uses. They were two independent literals, so retuning the
    seed in one place would have left the model's features describing a
    different seed than its penalty.
    """

    def test_seed_weight_defaults_to_the_shared_constant(self):
        from cas13guide.util.duplex import SEED_SPAN

        length = 22
        lo, hi = SEED_SPAN
        inside = int(((lo + hi) / 2) * length) + 1
        self.assertEqual(seed_weight(inside, length), 1.0)
        self.assertEqual(
            seed_weight(inside, length), seed_weight(inside, length, seed_span=SEED_SPAN)
        )

    def test_the_feature_layout_reads_the_same_span(self):
        from cas13guide.models.features import FeatureSpec
        from cas13guide.util.duplex import SEED_SPAN

        length = 22
        spec = FeatureSpec.for_length(length)
        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        x = spec.transform(spacer, profile_duplex(spacer, revcomp(spacer)), None)
        lo_frac, hi_frac = SEED_SPAN
        seed_lo = int(lo_frac * length)
        seed_hi = max(seed_lo + 1, int(hi_frac * length))
        expected_gc_seed = (
            sum(1 for b in spacer[seed_lo:seed_hi] if b in "GC") / (seed_hi - seed_lo)
        )
        gc_seed_index = spec.names.index("gc_seed")
        self.assertAlmostEqual(x[gc_seed_index], expected_gc_seed)


class TestComplementRejectsNonRnaCharacters(unittest.TestCase):
    """``str.translate`` leaves unknown characters alone; that must not happen here.

    Silently passing a stray base through as itself would turn a malformed
    sequence into a plausible-looking duplex instead of an error.
    """

    def test_a_foreign_base_raises_rather_than_passing_through(self):
        from cas13guide.util.seq import complement, revcomp

        with self.assertRaises(KeyError):
            complement("AUGX")
        with self.assertRaises(KeyError):
            revcomp("AUGX")

    def test_the_rna_alphabet_including_n_round_trips(self):
        from cas13guide.util.seq import complement, revcomp

        self.assertEqual(complement("AUGCN"), "UACGN")
        self.assertEqual(revcomp("AUGCN"), "NGCAU")
        self.assertEqual(revcomp(revcomp("AUGCN")), "AUGCN")



if __name__ == "__main__":
    unittest.main()
