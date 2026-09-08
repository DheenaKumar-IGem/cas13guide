"""The fifth finding of round eleven, and the blind spot that hid it.

Round eleven's critic was asked two questions: what did the six dimensions not
look at, and is this whole approach converging. Its answer to the second is the
more useful one, and it is quoted in the README: this is a *consistency* method,
not a *ground-truth* one. It finds two descriptions of a run disagreeing. It is
structurally unable to find a formula that is internally coherent, documented as
exactly what it does, tested against its own output, and scientifically wrong.

It also named a blind spot the method *creates*: because the standard fix for a
cross-surface disagreement is to add a caveat, a flag or a probe, the repo
accumulates guards that nothing reads. Then it found one.
"""

import dataclasses
import unittest

from helpers import *  # noqa: F401,F403


class TestASaturatedAssayDoesNotRank(unittest.TestCase):
    """``SignalPrediction.saturated`` was computed and read by nothing.

    ``PROVISIONAL_CONSTANTS`` states the rule and names the check: "A saturating
    assay reports 1.0 for every candidate, which is not a weak ranking but no
    ranking at all ... ``SignalPrediction.saturated`` says when they have not
    been." Its only reader was a test.

    The gate that did run, ``adds_information``, measures the spread of
    ``relative_signal / (activity x accessibility)`` -- and that probe cannot
    tell two nonlinearities apart. An assay bending because the *activation*
    curve bends carries real ordering. One bending because the *reporter is
    exhausted* carries none: every well is as bright as it can get. Both look
    non-proportional.

    So supplying measured constants that exhaust the reporter switched the
    signal axis on. Measured with ``k_cat=4e-2`` and ``target_nm=400``: all 323
    guides at 499.9999 nM of a 500 nM reporter, a ``relative_signal`` spread of
    2.5e-08 that ``_normalize`` stretches to fill [0, 1] -- so an axis weighted
    2.5 contributed up to 22% of an 11-point score out of float noise, and 309
    of the 323 changed rank against the same run on shipped constants.
    """

    def _model(self, **overrides):
        from cas13guide.services.kinetics import (
            PROVISIONAL_CONSTANTS, CollateralKineticsModel,
        )

        constants = dataclasses.replace(
            PROVISIONAL_CONSTANTS, provisional=False,
            source="measured (test fixture)", **overrides)
        return CollateralKineticsModel(constants)

    def test_an_exhausted_reporter_does_not_earn_the_axis(self):
        """The finding. Every probe but the near-dead one saturates."""
        model = self._model(k_cat=4e-2, target_nm=400.0)
        saturated = sum(
            1 for a, x in model._PROBES if model.predict(a, x).saturated
        )
        self.assertGreaterEqual(saturated, 7, "fixture must exhaust the reporter")
        self.assertFalse(model.adds_information())

    def test_a_genuinely_bending_assay_still_earns_it(self):
        """The saving must not cost the case the probe exists for: an assay
        whose *activation* curve bends is informative and must keep ranking."""
        model = self._model(target_nm=400.0)
        saturated = sum(
            1 for a, x in model._PROBES if model.predict(a, x).saturated
        )
        self.assertEqual(saturated, 0, "fixture must not be reporter-limited")
        self.assertTrue(model.adds_information())

    def test_the_shipped_placeholders_still_do_not(self):
        """Unchanged: degenerate for the original reason, a rescaled copy of
        two axes already in the ranking."""
        from cas13guide.services.kinetics import CollateralKineticsModel

        self.assertFalse(CollateralKineticsModel().adds_information())

    def test_the_saturation_flag_is_read_by_the_package_now(self):
        """It was computed on every prediction and consulted only by a test."""
        import inspect

        from cas13guide.services import kinetics

        source = inspect.getsource(kinetics.CollateralKineticsModel.adds_information)
        self.assertIn("saturated", source)

    def test_a_saturated_run_leaves_the_axis_out_of_the_objectives(self):
        """End of the chain: `predicted_signal_ranks` is what puts the axis
        into `objectives()`, where a weight of 2.5 applies to it."""
        from cas13guide.metrics.aggregate import build_metric_vector
        from cas13guide.contract.enums import TargetClass

        model = self._model(k_cat=4e-2, target_nm=400.0)
        spec = make_spec(assay=AssayType.IN_VITRO_DETECTION, task=Task.DETECTION)
        evidence = (make_evidence(0.8, 0.9, target_id="t1",
                                  target_class=TargetClass.REQUIRED),)
        vector = build_metric_vector({"reference": evidence}, clean_qc(), spec,
                                     kinetics=model)

        self.assertFalse(vector.predicted_signal_ranks)
        self.assertNotIn("predicted_signal", vector.objectives())


class TestTheDeadValueSweepCanSeeProperties(unittest.TestCase):
    """The instrument could not have found the defect above.

    ``verify_nothing_is_computed_for_nobody.py`` walks dataclass field
    annotations. ``saturated`` is a ``@property``, so the sweep never saw it --
    and a property is exactly the shape a computed-but-unread *guard* takes,
    which is the class the critic says this method keeps producing.
    """

    def test_the_sweep_collects_properties_as_well_as_fields(self):
        import ast
        import pathlib

        instrument = (pathlib.Path(__file__).resolve().parent.parent
                      / "examples" / "verify_nothing_is_computed_for_nobody.py")
        tree = ast.parse(instrument.read_text(encoding="utf-8"))
        source = ast.unparse(tree)
        self.assertIn("property", source,
                      "the sweep still cannot see a computed-but-unread property")


if __name__ == "__main__":
    unittest.main()
