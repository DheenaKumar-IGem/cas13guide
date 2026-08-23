"""Expression as bounds (fix #6): 'not detected' is not 'absent'."""

import unittest

from helpers import *  # noqa: F401,F403

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


if __name__ == "__main__":
    unittest.main()
