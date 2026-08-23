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

    def test_lwacas13a_rejects_a_g_at_the_three_prime_pfs(self):
        spec = self._spec(Effector.LWA_CAS13A)
        spacer = "GCAUUGCUGUGGAGCUGAAGUCCAUCGA"  # 28nt, the Cas13a length
        bad = Candidate("g", spacer, "t1", 0, three_prime_flank="G")
        good = Candidate("g", spacer, "t1", 0, three_prime_flank="A")
        self.assertIn(QCFlag.PFS_INCOMPATIBLE, qc_candidate(bad, spec).flags)
        self.assertNotIn(QCFlag.PFS_INCOMPATIBLE, qc_candidate(good, spec).flags)

    def test_cas13d_has_no_pfs_constraint(self):
        spec = self._spec(Effector.RFX_CAS13D)
        cand = Candidate("g", "GCAUUGCUGUGGAGCUGAAGUC", "t1", 0, three_prime_flank="G")
        self.assertNotIn(QCFlag.PFS_INCOMPATIBLE, qc_candidate(cand, spec).flags)


if __name__ == "__main__":
    unittest.main()
