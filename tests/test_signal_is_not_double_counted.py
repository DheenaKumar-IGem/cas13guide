"""An axis that is a rescaled copy of two other axes must not rank.

`predicted_signal` is computed from `activity_lower * accessibility`, and both
of those are already objectives. With the shipped placeholder constants the ODE
sits in its linear regime and the output is exactly
`k * activity_lower * accessibility` -- measured, the ratio varies by 0.159%
across the unit square, and the values occupy the bottom 1% of a declared [0,1]
range.

An objective like that cannot break a Pareto tie its own inputs did not already
break. What it can do is re-weight: at `DEFAULT_WEIGHTS["predicted_signal"] =
2.5`, the second-highest weight in the table, a user tuning `required_coverage`
and `accessibility` was moving them by more than the numbers they typed.

It is not inherently degenerate -- the ODE saturates once effective target
approaches the effector concentration, and there the response genuinely stops
being proportional. So the pipeline probes the model instead of assuming, and
the axis comes back on its own when constants that reach that regime arrive.
"""

import dataclasses
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.models.evidence import MetricVector
from cas13guide.services.kinetics import (
    PROVISIONAL_CONSTANTS,
    CollateralKineticsModel,
)

PROBES = ((0.8, 0.5), (0.4, 1.0), (0.5, 0.8), (1.0, 0.4), (0.2, 0.3), (0.9, 0.9))


class TestTheDegeneracyIsReal(unittest.TestCase):
    """Measure it rather than assert it, so the numbers stay honest."""

    def setUp(self):
        self.model = CollateralKineticsModel()

    def test_signal_is_a_constant_multiple_of_its_two_inputs(self):
        ratios = [
            self.model.predict(a, c).relative_signal / (a * c) for a, c in PROBES
        ]
        spread = (max(ratios) - min(ratios)) / max(ratios)
        self.assertLess(spread, 0.02, f"ratio spread {spread:.4%} is not constant")

    def test_equal_products_give_equal_signal(self):
        """The sharpest form: it is a function of the product alone."""
        same = [self.model.predict(a, c).relative_signal for a, c in
                ((0.8, 0.5), (0.4, 1.0), (0.5, 0.8), (1.0, 0.4))]
        for value in same[1:]:
            self.assertAlmostEqual(value, same[0], places=9)

    def test_the_model_reports_that_it_adds_nothing(self):
        self.assertFalse(self.model.adds_information())

    def test_saturating_constants_are_informative_again(self):
        """Self-correcting: the probe is about the regime, not about the code."""
        saturating = dataclasses.replace(PROVISIONAL_CONSTANTS, target_nm=400.0)
        self.assertTrue(CollateralKineticsModel(saturating).adds_information())


class TestItIsReportedButNotRanked(unittest.TestCase):
    def _vector(self, ranks):
        return MetricVector(
            required_coverage=0.8,
            desired_utility=0.0,
            worst_off_target=0.0,
            cumulative_burden=0.0,
            robustness=0.8,
            manufacturability=0.9,
            accessibility=0.6,
            predicted_signal=0.0048,
            predicted_signal_ranks=ranks,
        )

    def test_a_degenerate_signal_leaves_the_objective_set(self):
        self.assertNotIn("predicted_signal", self._vector(False).objectives())

    def test_an_informative_signal_stays_in_it(self):
        self.assertIn("predicted_signal", self._vector(True).objectives())

    def test_the_value_is_still_carried_either_way(self):
        """Dropping it from ranking must not drop it from the report."""
        for ranks in (True, False):
            with self.subTest(ranks=ranks):
                self.assertEqual(self._vector(ranks).predicted_signal, 0.0048)

    def test_the_default_is_to_rank(self):
        """Only a model that has been probed and found flat opts out."""
        self.assertTrue(MetricVector.predicted_signal_ranks)


class TestTheRunSaysWhy(unittest.TestCase):
    """A column that is present but not ranked needs explaining, or it misleads."""

    def _limitation_ids(self, **kw):
        from cas13guide.services.limitations import limitations_for_run
        from cas13guide.services.registry import (
            HeuristicCas13Model,
            ModelRegistry,
            NullAccessibilityModel,
        )

        registry = ModelRegistry(
            HeuristicCas13Model(), NullAccessibilityModel(),
            kinetics=CollateralKineticsModel(**kw),
        )
        spec = make_spec(task=Task.DETECTION, assay=AssayType.IN_VITRO_DETECTION)
        return {l.id for l in limitations_for_run(spec, registry=registry)}

    def test_a_degenerate_signal_is_announced(self):
        self.assertIn("signal-adds-no-information", self._limitation_ids())

    def test_an_informative_signal_is_not(self):
        """Both directions -- a note that always fires says nothing."""
        saturating = dataclasses.replace(PROVISIONAL_CONSTANTS, target_nm=400.0)
        self.assertNotIn(
            "signal-adds-no-information", self._limitation_ids(constants=saturating)
        )
