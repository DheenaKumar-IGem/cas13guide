"""Stage E: what the gate is allowed to see."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import ReasonCode, TargetClass, Verdict
from cas13guide.pipeline.e_aggregate import assess_guides


def strong(**kw):
    return make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED, **kw)


class TestEvidenceVisibleToTheGate(unittest.TestCase):
    def test_gate_sees_evidence_gaps_from_non_reference_contexts(self):
        """An OOD flag in a variant context is still a gap for this guide.

        Passing only the reference context to the gate would let a guide that
        is unvalidated in a haplotype context report a clean PASS.
        """
        spec = make_spec()
        candidates = {"g1": make_candidate("g1")}
        qc = {"g1": clean_qc("g1")}
        evidence = {
            "g1": {
                "reference": (strong(),),
                "hap2": (strong(), make_evidence(0.0, 0.05, locus_id="LO", ood=True,
                                                 domain_ok=False)),
            }
        }
        [assessment] = assess_guides(
            candidates, qc, evidence, exhaustive_bound(), frozenset(), spec
        )
        self.assertIs(assessment.gate.verdict, Verdict.REVIEW)
        self.assertIn(ReasonCode.MODEL_OUT_OF_DOMAIN, assessment.gate.reasons)

    def test_reference_context_supplies_the_attached_evidence(self):
        spec = make_spec()
        reference_only = make_evidence(
            0.9, 0.95, locus_id="L_REF", target_id="t1", target_class=TargetClass.REQUIRED
        )
        variant_only = make_evidence(
            0.9, 0.95, locus_id="L_VAR", target_id="t1", target_class=TargetClass.REQUIRED
        )
        [assessment] = assess_guides(
            {"g1": make_candidate("g1")},
            {"g1": clean_qc("g1")},
            {"g1": {"hap2": (variant_only,), "reference": (reference_only,)}},
            exhaustive_bound(),
            frozenset(),
            spec,
        )
        self.assertEqual([e.locus_id for e in assessment.evidence], ["L_REF"])

    def test_empty_reference_is_not_replaced_by_a_variant_context(self):
        """An empty reference tuple is falsy but is still the reference.

        Falling through to a variant would attach evidence the metrics were
        never computed from, corrupting the off-target explorer.
        """
        spec = make_spec()
        variant_only = make_evidence(
            0.9, 0.95, locus_id="L_VAR", target_id="t1", target_class=TargetClass.REQUIRED
        )
        [assessment] = assess_guides(
            {"g1": make_candidate("g1")},
            {"g1": clean_qc("g1")},
            {"g1": {"hap2": (variant_only,), "reference": ()}},
            exhaustive_bound(),
            frozenset(),
            spec,
        )
        self.assertEqual(assessment.evidence, ())
        self.assertEqual(assessment.metrics.required_coverage, 0.0)
        self.assertIs(assessment.gate.verdict, Verdict.FAIL)


class TestStratumSelection(unittest.TestCase):
    def test_stage_d_keys_on_mismatch_depth_not_intent(self):
        from cas13guide.contract.enums import TargetClass
        from cas13guide.pipeline.d_pair_evidence import _stratum_for

        spec = make_spec()
        perfect = make_hit("g1", "L1", "t1", TargetClass.REQUIRED, n_mismatch=0)
        distant = make_hit("g1", "L2", "t1", TargetClass.REQUIRED, n_mismatch=3)
        self.assertIn("mm0", _stratum_for(spec, perfect))
        self.assertIn("mm3+", _stratum_for(spec, distant))
        # Same intent, different strata -- that is the whole point.
        self.assertNotEqual(_stratum_for(spec, perfect), _stratum_for(spec, distant))

    def test_same_duplex_different_intent_shares_a_stratum(self):
        from cas13guide.contract.enums import TargetClass
        from cas13guide.pipeline.d_pair_evidence import _stratum_for

        spec = make_spec()
        required = make_hit("g1", "L1", "t1", TargetClass.REQUIRED, n_mismatch=1)
        forbidden = make_hit("g1", "L2", "wt", TargetClass.FORBIDDEN, n_mismatch=1)
        self.assertEqual(_stratum_for(spec, required), _stratum_for(spec, forbidden))


if __name__ == "__main__":
    unittest.main()
