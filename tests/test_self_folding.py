"""The self-folding check decides a hard FAIL, so it has to be about folding.

`SPACER_SELF_FOLDING` sets `passed = False`, which f_decide turns into
DETERMINISTIC_INCOMPATIBLE and Verdict.FAIL -- documented as unbuyable-back by
any user weighting. A candidate it flags is deleted from the run.

It was deciding that on a Watson-Crick-only stem count with no loop constraint.
Measured against ViennaRNA over 6,000 random 22-mers, that caught 5.6% of
spacers folding to -8 kcal/mol or tighter. The specific miss below is the one
that makes the point: AGCCGGAGGCAUGGUCUCCGGU is a clean 9 bp hairpin at
-14.4 kcal/mol, and the old scan called it a 6 -- one under the threshold --
because two of the nine pairs are G-U wobbles, which `revcomp` cannot see.

util/duplex.py opens by saying G-U wobbles are tolerated in RNA:RNA duplexes and
that "the rule is derived here once and used everywhere". It was used everywhere
except here, in the one place that deletes candidates outright.
"""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import QCFlag
from cas13guide.models.evidence import Candidate
from cas13guide.pipeline.b_qc import (
    HELIX_PAIRS,
    MAX_SELF_STEM,
    MIN_HAIRPIN_LOOP,
    SELF_FOLD_MAX_DG,
    longest_self_complement,
    qc_candidate,
)

#: A clean 9 bp hairpin with two G-U pairs. ViennaRNA: -14.4 kcal/mol,
#: `.(((((((((...)))))))))`. The Watson-Crick-only scan scored it 6.
WOBBLE_HAIRPIN = "AGCCGGAGGCAUGGUCUCCGGU"

try:
    import RNA

    HAVE_VIENNA = True
except ImportError:  # pragma: no cover
    HAVE_VIENNA = False


class TestTheStemScanSeesRnaPairs(unittest.TestCase):
    def test_gu_wobbles_are_helix_pairs(self):
        for pair in ("GU", "UG"):
            with self.subTest(pair=pair):
                self.assertIn(pair, HELIX_PAIRS)

    def test_the_hairpin_that_used_to_slip_through_is_seen_whole(self):
        """The regression, as a known answer."""
        self.assertEqual(longest_self_complement(WOBBLE_HAIRPIN), 9)

    def test_and_it_is_now_flagged(self):
        self.assertGreater(longest_self_complement(WOBBLE_HAIRPIN), MAX_SELF_STEM)

    def test_a_helix_needs_a_loop_to_turn_through(self):
        """Bases cannot pair across nothing; the scan used to count it."""
        self.assertGreaterEqual(MIN_HAIRPIN_LOOP, 3)
        # perfectly self-complementary but with no room for a loop
        self.assertEqual(longest_self_complement("GC"), 0)
        self.assertEqual(longest_self_complement("GGCC"), 0)

    def test_a_real_hairpin_with_a_loop_is_counted(self):
        # 4 bp stem, 4 nt loop
        self.assertGreaterEqual(longest_self_complement("GGGGAAAACCCC"), 4)

    def test_an_unstructured_spacer_scores_zero(self):
        self.assertEqual(longest_self_complement("AAAAAAAAAAAAAAAAAAAAAA"), 0)

    def test_dna_is_read_as_rna(self):
        self.assertEqual(
            longest_self_complement("GGGGTTTTCCCC"),
            longest_self_complement("GGGGUUUUCCCC"),
        )


class TestTheRealFoldIsPreferred(unittest.TestCase):
    """A proxy for a free energy should stand down when the energy is available."""

    def _qc(self, spacer, fold=None):
        return qc_candidate(
            Candidate(guide_id="g", spacer=spacer, source_target_id="t1", start=0),
            make_spec(),
            fold,
        )

    def test_a_supplied_fold_decides_the_flag(self):
        flagged = self._qc("AAAAAAAAAAAAAAAAAAAAAA", fold=lambda s: -20.0)
        self.assertIn(QCFlag.SPACER_SELF_FOLDING, flagged.flags)

    def test_a_supplied_fold_can_also_clear_a_spacer_the_proxy_flags(self):
        """The proxy's false alarms are half its flags; the fold overrides them."""
        self.assertGreater(longest_self_complement(WOBBLE_HAIRPIN), MAX_SELF_STEM)
        cleared = self._qc(WOBBLE_HAIRPIN, fold=lambda s: -0.5)
        self.assertNotIn(QCFlag.SPACER_SELF_FOLDING, cleared.flags)

    def test_a_broken_folder_falls_back_rather_than_crashing(self):
        """QC must not take the run down because a folding engine misbehaved."""
        def boom(_sequence):
            raise RuntimeError("engine exploded")

        result = self._qc(WOBBLE_HAIRPIN, fold=boom)
        self.assertIn(QCFlag.SPACER_SELF_FOLDING, result.flags)

    def test_without_a_folder_the_proxy_still_decides(self):
        self.assertIn(QCFlag.SPACER_SELF_FOLDING, self._qc(WOBBLE_HAIRPIN).flags)

    @unittest.skipUnless(HAVE_VIENNA, "ViennaRNA not installed")
    def test_the_orchestrators_folder_agrees_with_viennarna(self):
        from cas13guide.pipeline.orchestrator import _self_fold_energy

        fold = _self_fold_energy()
        self.assertIsNotNone(fold)
        self.assertAlmostEqual(fold(WOBBLE_HAIRPIN), -14.4, places=1)

    @unittest.skipUnless(HAVE_VIENNA, "ViennaRNA not installed")
    def test_the_threshold_flags_the_known_hairpin(self):
        from cas13guide.pipeline.orchestrator import _self_fold_energy

        self.assertLessEqual(_self_fold_energy()(WOBBLE_HAIRPIN), SELF_FOLD_MAX_DG)


class TestTheProxyIsHonestlyDocumented(unittest.TestCase):
    """It is still weak, and the constant says so rather than implying rigour."""

    def test_the_threshold_documents_its_measured_recall(self):
        import cas13guide.pipeline.b_qc as module

        source = module.__dict__.get("__doc__") or ""
        import inspect

        text = inspect.getsource(module)
        self.assertIn("28.9%", text)
        self.assertIn("5.6%", text)
