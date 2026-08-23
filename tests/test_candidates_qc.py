"""Stages A and B: tiling, deduplication and the deterministic filters."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import AssayType, Effector, QCFlag, TargetClass
from cas13guide.contract.spec import (
    ContextScope,
    EffectorAssayProfile,
    ResolvedDesignSpec,
    TargetContract,
    TargetSpec,
)
from cas13guide.models.evidence import Candidate
from cas13guide.pipeline.a_candidates import design_candidates, tile_target
from cas13guide.pipeline.b_qc import qc_candidate
from cas13guide.util.seq import revcomp


class TestTiling(unittest.TestCase):
    def test_spacer_is_the_reverse_complement_of_the_target_site(self):
        target = TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED)
        cands = tile_target(target, spacer_length=22)
        first = cands[0]
        self.assertEqual(first.spacer, revcomp(SEQ_A[0:22]))

    def test_window_count_is_exact(self):
        target = TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED)
        cands = tile_target(target, spacer_length=22)
        self.assertEqual(len(cands), len(SEQ_A) - 22 + 1)

    def test_identical_spacers_deduplicate_to_one_guide(self):
        dup = SEQ_A[:30]
        spec = make_spec(
            targets=(
                TargetSpec(target_id="t1", sequence=dup, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="t2", sequence=dup, target_class=TargetClass.DESIRED),
            )
        )
        cands = design_candidates(spec)
        self.assertEqual(len(cands), len({c.spacer for c in cands}))
        self.assertEqual(len(cands), len(dup) - spec.spacer_length + 1)

    def test_candidate_budget_is_respected(self):
        spec = make_spec()
        spec = spec.model_copy(
            update={"budget": spec.budget.model_copy(update={"max_candidates": 5})}
        )
        self.assertLessEqual(len(design_candidates(spec)), 5)

    def test_junction_spanning_guides_are_marked(self):
        target = TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED)
        cands = tile_target(target, spacer_length=22, junctions=(10,))
        spanning = [c for c in cands if c.spans_junction]
        self.assertTrue(spanning)
        for c in spanning:
            self.assertLess(c.start, 10)
            self.assertGreater(c.start + 22, 10)


class TestDeterministicQC(unittest.TestCase):
    def test_polyu_terminator_is_flagged(self):
        spec = make_spec()
        cand = Candidate("g", "GCAUUGCUGUGUUUUGCUGAAG", "t1", 0)
        self.assertIn(QCFlag.POLY_U_TERMINATOR, qc_candidate(cand, spec).flags)

    def test_homopolymer_run_is_flagged(self):
        spec = make_spec()
        cand = Candidate("g", "GCAUUGCUGGGGGGCUGAAGUC", "t1", 0)
        result = qc_candidate(cand, spec)
        self.assertIn(QCFlag.HOMOPOLYMER_RUN, result.flags)
        self.assertFalse(result.passed)

    def test_gc_out_of_range_is_flagged(self):
        spec = make_spec()
        cand = Candidate("g", "GCGCGCGCGCGCGCGCGCGCGC", "t1", 0)
        self.assertIn(QCFlag.GC_OUT_OF_RANGE, qc_candidate(cand, spec).flags)

    def test_clean_spacer_passes(self):
        spec = make_spec()
        cand = Candidate("g", "GCAUUGCUGUGGAGCUGAAGUC", "t1", 0)
        result = qc_candidate(cand, spec)
        self.assertTrue(result.passed, result.flags)
        self.assertGreater(result.manufacturability, 0.5)

    def test_unsupported_spacer_length_is_flagged(self):
        spec = make_spec()  # RfxCas13d supports 22/23/30
        cand = Candidate("g", "GCAUUGCUGUGGAGCUGAA", "t1", 0)  # 19nt
        self.assertIn(QCFlag.SPACER_LENGTH_UNSUPPORTED, qc_candidate(cand, spec).flags)


class TestPFSPolicy(unittest.TestCase):
    """LwaCas13a prefers a non-G 3' PFS; Cas13d has no PFS requirement."""

    def _spec(self, effector):
        return ResolvedDesignSpec(
            contract=TargetContract(
                task=make_spec().contract.task,
                targets=(
                    TargetSpec(
                        target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED
                    ),
                ),
            ),
            profile=EffectorAssayProfile(
                effector=effector, assay=AssayType.CELL_CULTURE_KNOCKDOWN
            ),
            context=ContextScope(
                organism="Homo sapiens",
                genome_build="GRCh38",
                annotation_version="v44",
                declared_universes=FULL_UNIVERSES,
            ),
        )

    def test_lwacas13a_deprioritizes_a_g_at_the_three_prime_pfs(self):
        """The 3' H bias is a *preference*: it ranks down, it does not exclude.

        This used to be a hard ``PFS_INCOMPATIBLE`` flag, which the gate turns
        into a deterministic FAIL -- so roughly a quarter of every candidate set
        was deleted outright on the strength of a reported base bias, and a
        target whose only accessible sites were G-flanked became undesignable.
        """
        spec = self._spec(Effector.LWA_CAS13A)
        spacer = "GCAUUGCUGUGGAGCUGAAGUCCAUCGA"  # 28nt, the Cas13a length
        disfavoured = qc_candidate(
            Candidate("g", spacer, "t1", 0, three_prime_flank="G"), spec
        )
        preferred = qc_candidate(
            Candidate("g", spacer, "t1", 0, three_prime_flank="A"), spec
        )

        # not excluded ...
        self.assertTrue(disfavoured.passed)
        self.assertNotIn(QCFlag.PFS_INCOMPATIBLE, disfavoured.flags)
        # ... but recorded and scored down
        self.assertIn(QCFlag.PFS_DISFAVOURED, disfavoured.warnings)
        self.assertLess(disfavoured.pfs_preference, preferred.pfs_preference)
        self.assertGreater(disfavoured.pfs_preference, 0.0)

        self.assertTrue(preferred.passed)
        self.assertEqual(preferred.warnings, ())
        self.assertEqual(preferred.pfs_preference, 1.0)

    def test_the_deprioritization_actually_reaches_the_ranking(self):
        """A soft penalty nothing reads is not a penalty."""
        from cas13guide.metrics.aggregate import build_metric_vector
        from cas13guide.pipeline.f_decide import DEFAULT_WEIGHTS

        spec = self._spec(Effector.LWA_CAS13A)
        spacer = "GCAUUGCUGUGGAGCUGAAGUCCAUCGA"
        disfavoured = qc_candidate(
            Candidate("g", spacer, "t1", 0, three_prime_flank="G"), spec
        )
        preferred = qc_candidate(
            Candidate("g", spacer, "t1", 0, three_prime_flank="A"), spec
        )
        mv_bad = build_metric_vector({"reference": ()}, disfavoured, spec)
        mv_good = build_metric_vector({"reference": ()}, preferred, spec)

        self.assertLess(
            mv_bad.objectives()["pfs_preference"],
            mv_good.objectives()["pfs_preference"],
        )
        self.assertIn("pfs_preference", DEFAULT_WEIGHTS)
        self.assertGreater(DEFAULT_WEIGHTS["pfs_preference"], 0.0)

    def test_a_hard_pfs_requirement_still_excludes(self):
        """PspCas13b's 5' constraint is modelled as a requirement, not a bias."""
        spec = self._spec(Effector.PSP_CAS13B)
        spacer = "GCAUUGCUGUGGAGCUGAAGUCCAUCGAUG"  # 30nt, the Cas13b length
        bad = Candidate("g", spacer, "t1", 0, five_prime_flank="C")
        good = Candidate("g", spacer, "t1", 0, five_prime_flank="A")
        self.assertIn(QCFlag.PFS_INCOMPATIBLE, qc_candidate(bad, spec).flags)
        self.assertFalse(qc_candidate(bad, spec).passed)
        self.assertNotIn(QCFlag.PFS_INCOMPATIBLE, qc_candidate(good, spec).flags)

    def test_a_missing_flank_is_not_held_against_a_candidate(self):
        """At a transcript end there is no flanking base to read."""
        spec = self._spec(Effector.LWA_CAS13A)
        spacer = "GCAUUGCUGUGGAGCUGAAGUCCAUCGA"
        result = qc_candidate(Candidate("g", spacer, "t1", 0, three_prime_flank=None), spec)
        self.assertEqual(result.pfs_preference, 1.0)
        self.assertEqual(result.warnings, ())

    def test_cas13d_has_no_pfs_constraint(self):
        spec = self._spec(Effector.RFX_CAS13D)
        cand = Candidate("g", "GCAUUGCUGUGGAGCUGAAGUC", "t1", 0, three_prime_flank="G")
        result = qc_candidate(cand, spec)
        self.assertNotIn(QCFlag.PFS_INCOMPATIBLE, result.flags)
        self.assertEqual(result.pfs_preference, 1.0)  # no preference either



class TestSpacerLengthIsConfigurableNotFixed(unittest.TestCase):
    """Tolerated-but-unpublished lengths are unvalidated, not incompatible.

    QC used to fail any length outside the ortholog's canonical tuple, which
    made FR2.1's configurable window un-configurable and, for a 22-24 nt mature
    miRNA against 28 nt LwaCas13a spacers, made the target undesignable rather
    than merely unusual. Out-of-range is still a hard failure -- the ortholog
    physically cannot use it.
    """

    def _spec_at(self, effector, length):
        return ResolvedDesignSpec(
            contract=TargetContract(
                task=make_spec().contract.task,
                targets=(TargetSpec(target_id="t1", sequence=SEQ_A,
                                    target_class=TargetClass.REQUIRED),),
            ),
            profile=EffectorAssayProfile(
                effector=effector, assay=AssayType.IN_VITRO_DETECTION,
                spacer_length=length,
            ),
            context=ContextScope(
                organism="Homo sapiens", genome_build="GRCh38",
                annotation_version="v44", declared_universes=FULL_UNIVERSES,
            ),
        )

    #: A composition-clean spacer, so these tests isolate the length rule.
    SPACER_28 = "GCAUUGCUGUGGAGCUGAAGUCCAUCGA"
    SPACER_24 = "GCAUUGCUGUGGAGCUGAAGUCCA"

    def test_a_canonical_length_is_clean(self):
        spec = self._spec_at(Effector.LWA_CAS13A, 28)
        result = qc_candidate(Candidate("g", self.SPACER_28, "t1", 0), spec)
        self.assertTrue(result.passed, result.flags)
        self.assertNotIn(QCFlag.SPACER_LENGTH_UNSUPPORTED, result.flags)
        self.assertNotIn(QCFlag.SPACER_LENGTH_NONCANONICAL, result.warnings)
        self.assertTrue(spec.profile.spacer_length_is_canonical)

    def test_a_tolerated_noncanonical_length_warns_but_passes(self):
        spec = self._spec_at(Effector.LWA_CAS13A, 24)
        result = qc_candidate(Candidate("g", self.SPACER_24, "t1", 0), spec)
        self.assertNotIn(QCFlag.SPACER_LENGTH_UNSUPPORTED, result.flags)
        self.assertIn(QCFlag.SPACER_LENGTH_NONCANONICAL, result.warnings)
        self.assertTrue(result.passed, result.flags)
        self.assertFalse(spec.profile.spacer_length_is_canonical)

    def test_a_length_outside_the_tolerated_range_is_rejected_at_the_spec(self):
        with self.assertRaises(Exception) as caught:
            self._spec_at(Effector.LWA_CAS13A, 40)
        self.assertIn("20-28", str(caught.exception))

    def test_qc_still_fails_a_candidate_whose_length_is_out_of_range(self):
        """The spec guards the configured length; QC guards each candidate."""
        spec = self._spec_at(Effector.LWA_CAS13A, 28)
        result = qc_candidate(Candidate("g", self.SPACER_28[:12], "t1", 0), spec)
        self.assertIn(QCFlag.SPACER_LENGTH_UNSUPPORTED, result.flags)
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()
