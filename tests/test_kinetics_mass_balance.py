"""The ODE must not create target it never had.

The activation step consumes a target molecule: one target activates one
effector complex and stays bound to it. So activated complex can never exceed
the target actually present. The original model held [T] constant -- the
standard pseudo-first-order simplification, valid when target is in large excess
over effector, which is exactly backwards for a diagnostic where the target IS
the analyte and the RNP is deliberately in excess. With this module's own
defaults (50 nM complex, 1 nM target) it produced 39.2 nM of activated complex
from 1 nM of target, and 82x over at 0.01 nM.

Nothing failed, because every test asked about ordering and ordering was never
wrong -- activation is monotone in [T]eff either way. That is what makes this
worth a test rather than a fix: the defect was invisible to every question the
suite knew how to ask, and it would have become a fabricated measurement the
moment real constants unlocked absolute_signal().
"""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.services.kinetics import CollateralKineticsModel

LEVELS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


class TestMassBalance(unittest.TestCase):
    def setUp(self):
        self.model = CollateralKineticsModel()
        self.constants = self.model.constants

    def _activated_nm(self, target_nm):
        fraction, _rate, _cleaved = self.model.integrate(target_nm)
        return fraction * self.constants.complex_nm

    def test_activated_complex_never_exceeds_the_target_available(self):
        """The invariant. One target activates at most one complex."""
        for target in LEVELS:
            with self.subTest(target_nm=target):
                self.assertLessEqual(
                    self._activated_nm(target),
                    target + 1e-9,
                    f"created activated complex from {target} nM of target",
                )

    def test_activated_complex_never_exceeds_the_effector_supplied(self):
        """The other cap: you cannot activate complex you did not add."""
        for target in LEVELS:
            with self.subTest(target_nm=target):
                self.assertLessEqual(
                    self._activated_nm(target), self.constants.complex_nm + 1e-9
                )

    def test_saturation_is_effector_limited_when_target_is_abundant(self):
        """Far above the effector concentration, the effector is the ceiling."""
        activated = self._activated_nm(1000.0)
        self.assertGreater(activated, 0.5 * self.constants.complex_nm)
        self.assertLessEqual(activated, self.constants.complex_nm + 1e-9)

    def test_the_defaults_do_not_satisfy_the_pseudo_first_order_assumption(self):
        """Why the constant-[T] form was wrong *here* specifically.

        Holding [T] constant is legitimate when [T] >> [C]tot. The module's own
        defaults have it the other way round by a factor of 50, so the
        simplification was never applicable to its own configuration.
        """
        self.assertLess(self.constants.target_nm, self.constants.complex_nm)


class TestRankingIsUnaffected(unittest.TestCase):
    """The fix must not disturb what relative_signal is for.

    Ordering by predicted signal has to stay monotone in effective target, or
    the fix would have traded a physical error for a ranking error.
    """

    def setUp(self):
        self.model = CollateralKineticsModel()

    def test_activation_is_monotone_in_effective_target(self):
        activated = [
            self.model.integrate(t)[0] for t in LEVELS
        ]
        for lower, higher in zip(activated, activated[1:]):
            self.assertLessEqual(lower, higher + 1e-12)

    def test_cleaved_reporter_is_monotone_in_effective_target(self):
        cleaved = [self.model.integrate(t)[2] for t in LEVELS]
        for lower, higher in zip(cleaved, cleaved[1:]):
            self.assertLessEqual(lower, higher + 1e-9)

    def test_zero_target_produces_no_signal(self):
        fraction, _rate, cleaved = self.model.integrate(0.0)
        self.assertEqual(fraction, 0.0)
        self.assertAlmostEqual(cleaved, 0.0, places=12)

    def test_absolute_signal_still_refuses_on_provisional_constants(self):
        """Fixing the physics must not be mistaken for unlocking the output."""
        from cas13guide.services.kinetics import ProvisionalConstantsError

        self.assertTrue(self.model.constants.provisional)
        with self.assertRaises(ProvisionalConstantsError):
            self.model.absolute_signal(1.0)


class TestConstantsCanActuallyBeSupplied(unittest.TestCase):
    """"Refuses until you supply constants" was unreachable by design.

    absolute_signal() unlocks when constants are measured, and the module
    documented KineticConstants.from_json as the way in -- but nothing plumbed a
    path through. workflow.py built CollateralKineticsModel() with no arguments,
    DesignRequest had no field for it, and the CLI had no flag. So the promise
    could not be kept by any caller short of writing Python against the
    internals. A documented escape hatch with no door is a hole, not a hatch.
    """

    def _measured_constants_file(self):
        import dataclasses
        import json
        import pathlib
        import tempfile

        from cas13guide.services.kinetics import PROVISIONAL_CONSTANTS

        data = dataclasses.asdict(PROVISIONAL_CONSTANTS)
        data["source"] = "measured on this instrument (test fixture)"
        data["provisional"] = False
        path = pathlib.Path(tempfile.mkdtemp()) / "kinetics.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_the_request_carries_a_constants_path(self):
        from cas13guide.services.workflow import DesignRequest

        self.assertTrue(hasattr(DesignRequest(), "kinetics_json"))

    def test_the_cli_exposes_a_flag_for_it(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "cas13guide", "design", "--help"],
            capture_output=True, text=True, timeout=120,
        )
        self.assertIn("--kinetics-constants", result.stdout)

    def test_measured_constants_unlock_the_absolute_path(self):
        from cas13guide.services.kinetics import CollateralKineticsModel, KineticConstants

        model = CollateralKineticsModel(
            KineticConstants.from_json(self._measured_constants_file())
        )
        self.assertFalse(model.constants.provisional)
        self.assertIn("measured", model.version)
        self.assertGreater(model.absolute_signal(0.9), 0.0)

    def test_a_run_given_real_constants_drops_the_provisional_warning(self):
        """The derived-limitations design earning its keep.

        A static caveat list would keep printing the placeholder warning after
        real constants arrived. This one is a function of the run, so it stops
        on its own.
        """
        from cas13guide.services.workflow import DesignRequest, run_request

        path = str(self._measured_constants_file())
        with_real = run_request(
            DesignRequest(preset="hsa-miR-21-5p", assay="in_vitro_detection",
                          kinetics_json=path)
        )
        self.assertNotIn(
            "provisional-kinetics", {l.id for l in with_real.limitations}
        )

    def test_a_run_without_them_still_carries_the_warning(self):
        """Both directions."""
        from cas13guide.services.workflow import DesignRequest, run_request

        without = run_request(
            DesignRequest(preset="hsa-miR-21-5p", assay="in_vitro_detection")
        )
        self.assertIn("provisional-kinetics", {l.id for l in without.limitations})

    def test_a_file_still_flagged_provisional_does_not_silently_unlock(self):
        """Supplying a file is not the same as measuring what is in it."""
        import dataclasses
        import json
        import pathlib
        import tempfile

        from cas13guide.services.kinetics import (
            PROVISIONAL_CONSTANTS,
            CollateralKineticsModel,
            KineticConstants,
            ProvisionalConstantsError,
        )

        data = dataclasses.asdict(PROVISIONAL_CONSTANTS)  # provisional stays True
        path = pathlib.Path(tempfile.mkdtemp()) / "still_provisional.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        model = CollateralKineticsModel(KineticConstants.from_json(path))
        with self.assertRaises(ProvisionalConstantsError):
            model.absolute_signal(0.9)
