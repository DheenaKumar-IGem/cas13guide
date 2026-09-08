"""The ortholog constants are data, and data with no test rots quietly.

Everything else in this package is checked by exercising it. These values are
never exercised -- they are simply believed -- and one of them was wrong for the
whole life of the repository: RfxCas13d carried a 45 nt "direct repeat" that
could not fold into a stem-loop. Nothing failed, because nothing looked.

What caught it was a *structural invariant*: a Cas13 direct repeat has to
present a stable hairpin for the protein to grip, so a DR that folds to
-1.1 kcal/mol on a 3 bp stem is not a DR whatever its provenance. That is the
property tested here, and it is the kind of check that transfers -- it would
catch a truncation, a paste of the wrong strand, or a spacer pasted into a DR
field, none of which a length assertion would.
"""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.effectors import EFFECTORS
from cas13guide.contract.enums import Effector
from cas13guide.pipeline.b_qc import longest_self_complement

try:
    import RNA

    HAVE_VIENNA = True
except ImportError:  # pragma: no cover
    HAVE_VIENNA = False


class TestDirectRepeatsAreStructurallyPlausible(unittest.TestCase):
    def test_every_direct_repeat_is_rna(self):
        for effector, params in EFFECTORS.items():
            with self.subTest(effector=effector):
                self.assertTrue(
                    set(params.direct_repeat) <= set("AUGC"),
                    f"{effector} DR has non-RNA characters (T instead of U?)",
                )

    def test_every_direct_repeat_is_thirty_six_nucleotides(self):
        """All three characterised Cas13 repeats here are 36 nt.

        Not a law of nature, but a strong enough regularity that an outlier is
        worth a human look -- which is exactly what it got, and it was wrong.
        """
        for effector, params in EFFECTORS.items():
            with self.subTest(effector=effector):
                self.assertEqual(len(params.direct_repeat), 36)

    def test_every_direct_repeat_has_a_real_stem(self):
        """Sequence-only, so it runs without ViennaRNA installed."""
        for effector, params in EFFECTORS.items():
            with self.subTest(effector=effector):
                stem = longest_self_complement(params.direct_repeat)
                self.assertGreaterEqual(
                    stem, 5, f"{effector} DR has no stem long enough to be a hairpin"
                )

    @unittest.skipUnless(HAVE_VIENNA, "ViennaRNA not installed")
    def test_every_direct_repeat_folds_into_a_stable_hairpin(self):
        """The check that would have caught the bad value on day one.

        -5 kcal/mol is a deliberately loose floor: the three real repeats fold
        to -15.4, -6.7 and -11.3, and the sequence this replaced folded to -1.1.
        The gap is wide enough that the threshold does not need to be precise.
        """
        for effector, params in EFFECTORS.items():
            with self.subTest(effector=effector):
                structure, energy = RNA.fold(params.direct_repeat)
                self.assertLess(energy, -5.0, f"{effector} DR does not fold")
                self.assertGreaterEqual(
                    structure.count("("), 6, f"{effector} DR pairs too few bases"
                )

    def test_the_repeats_are_distinct_from_each_other(self):
        """A copy-paste between orthologs is the other way this data goes bad."""
        repeats = [params.direct_repeat for params in EFFECTORS.values()]
        self.assertEqual(len(set(repeats)), len(repeats))


class TestDirectRepeatIsLoadBearing(unittest.TestCase):
    """Pinned because it is easy to mistake this data for decoration.

    b_qc.py flags a spacer that can base-pair with its own direct repeat, since
    that competes with the crRNA fold. Getting the DR wrong therefore does not
    fail loudly -- it changes verdicts. Measured across 20,000 random 22 nt
    spacers, the old RfxCas13d value flipped the DR-interaction verdict for
    2.8% of them (2.77% +/- 0.18% over five seeds), in both directions but
    about ten to one toward the wrong repeat missing an interaction the right
    one catches.

    The figure was 4.9% while the scan was Watson-Crick only. It moved because
    the scan learned G-U wobbles and MAX_DR_STEM was re-derived with it, not
    because the repeat changed -- which is the point of pinning the fold rather
    than the rate.
    """

    def test_the_dr_interaction_flag_actually_depends_on_the_dr(self):
        from cas13guide.pipeline.b_qc import MAX_DR_STEM, longest_cross_complement

        repeat = EFFECTORS[Effector.RFX_CAS13D].direct_repeat
        # revcomp of a slice of the DR must pair with it well past the threshold
        complementary = "".join(
            {"A": "U", "U": "A", "G": "C", "C": "G"}[base] for base in reversed(repeat[:12])
        )
        self.assertGreater(longest_cross_complement(complementary, repeat), MAX_DR_STEM)

    def test_a_spacer_designed_against_the_dr_is_flagged(self):
        from cas13guide.contract.enums import QCFlag
        from cas13guide.models.evidence import Candidate
        from cas13guide.pipeline.b_qc import qc_candidate

        repeat = EFFECTORS[Effector.RFX_CAS13D].direct_repeat
        spacer = "".join(
            {"A": "U", "U": "A", "G": "C", "C": "G"}[base] for base in reversed(repeat[:22])
        )
        # make_spec() is RfxCas13d by default, which is the ortholog under test
        result = qc_candidate(
            Candidate(guide_id="g", spacer=spacer, source_target_id="t1", start=0), make_spec()
        )
        self.assertIn(QCFlag.DR_SPACER_INTERACTION, result.flags)

    def test_an_ordinary_spacer_is_not_flagged_for_dr_interaction(self):
        """The other direction: a flag that always fires says nothing."""
        from cas13guide.contract.enums import QCFlag
        from cas13guide.models.evidence import Candidate
        from cas13guide.pipeline.b_qc import qc_candidate

        result = qc_candidate(
            Candidate(
                guide_id="g",
                spacer="GCAUUGCUGUGGAGCUGAAGUC",
                source_target_id="t1",
                start=0,
            ),
            make_spec(),
        )
        self.assertNotIn(QCFlag.DR_SPACER_INTERACTION, result.flags)
