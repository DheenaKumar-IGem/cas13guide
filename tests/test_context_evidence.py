"""Expression as bounds (fix #6): 'not detected' is not 'absent'."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass, Universe
from cas13guide.contract.spec import HazardWeights
from cas13guide.services.context import (
    SATURATION_K,
    abundance_factor,
    expression_interval,
)


class TestAbundanceFactor(unittest.TestCase):
    def test_monotone_and_saturating(self):
        values = [abundance_factor(t) for t in (0, 1, 10, 100, 1000, 10000)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(abundance_factor(SATURATION_K), 0.5)
        self.assertLess(values[-1], 1.0)

    def test_rejects_negative_expression(self):
        with self.assertRaises(ValueError):
            abundance_factor(-1.0)


class TestExpressionInterval(unittest.TestCase):
    def test_unmeasured_is_maximal_ignorance_not_zero(self):
        """The load-bearing behaviour: an unmeasured locus is not a safe one.

        Reading it as zero would zero out its whole hazard term and make an
        unmeasured off-target invisible.
        """
        iv = expression_interval(None)
        self.assertTrue(iv.imputed)
        self.assertEqual(iv.lower, 0.0)
        self.assertGreater(iv.upper, 0.9)  # near-maximal at the UCB

    def test_measured_zero_is_genuinely_zero(self):
        iv = expression_interval(0.0)
        self.assertFalse(iv.imputed)
        self.assertEqual((iv.lower, iv.upper), (0.0, 0.0))

    def test_unmeasured_outranks_measured_low_in_hazard(self):
        unmeasured = expression_interval(None)
        measured_low = expression_interval(2.0)
        self.assertGreater(unmeasured.upper, measured_low.upper)

    def test_uncertainty_widens_the_interval(self):
        tight = expression_interval(50.0, relative_uncertainty=0.05)
        loose = expression_interval(50.0, relative_uncertainty=0.60)
        self.assertLess(tight.width, loose.width)
        self.assertGreater(loose.upper, tight.upper)

    def test_point_estimate_sits_inside_the_bounds(self):
        iv = expression_interval(120.0, relative_uncertainty=0.3)
        self.assertLessEqual(iv.lower, iv.point)
        self.assertLessEqual(iv.point, iv.upper)



class TestPartialHazardWeightsAreCompleted(unittest.TestCase):
    """A partial override means "override these, keep the rest".

    Pydantic *replaces* a ``default_factory`` dict rather than merging it, so
    softening one biotype silently dropped every other biotype to the 1.0
    fallback -- halving rRNA and mito hazard at the exact point the gate reads
    it -- while overriding one intent class made ``weight_for`` raise
    ``KeyError`` partway through stage D.
    """

    def test_a_partial_by_class_override_keeps_the_other_classes(self):
        weights = HazardWeights(by_class={TargetClass.FORBIDDEN: 10.0})
        self.assertEqual(weights.weight_for(TargetClass.FORBIDDEN, Universe.CODING), 10.0)
        # the untouched classes keep their defaults instead of raising
        self.assertEqual(weights.weight_for(TargetClass.AVOID, Universe.CODING), 1.0)
        self.assertEqual(weights.weight_for(TargetClass.TOLERATED, Universe.CODING), 0.1)

    def test_a_partial_by_biotype_override_does_not_deflate_the_risky_ones(self):
        weights = HazardWeights(by_biotype={Universe.VIRAL: 0.5})
        self.assertEqual(weights.weight_for(TargetClass.AVOID, Universe.VIRAL), 0.5)
        # rRNA and mito must keep their elevated priors, not fall back to 1.0
        self.assertEqual(weights.weight_for(TargetClass.AVOID, Universe.RRNA), 2.0)
        self.assertEqual(weights.weight_for(TargetClass.AVOID, Universe.MITO), 1.5)

    def test_an_unannotated_hit_still_carries_hazard(self):
        weights = HazardWeights(by_class={TargetClass.FORBIDDEN: 10.0})
        self.assertGreater(weights.weight_for(None, Universe.RRNA), 0.0)

    def test_defaults_are_unchanged(self):
        default = HazardWeights()
        self.assertEqual(default.weight_for(TargetClass.FORBIDDEN, Universe.RRNA), 10.0)
        self.assertEqual(default.weight_for(TargetClass.REQUIRED, Universe.CODING), 0.0)


if __name__ == "__main__":
    unittest.main()
