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


class TestTheUnlockReachesASurfaceSomebodyReads(unittest.TestCase):
    """"Supply measured values and it unlocks" reached no output at all.

    ``absolute_signal()`` is a library method, and a grep for a call to it
    across ``src/`` returns nothing: the pipeline called ``predict()``, kept
    ``relative_signal`` and discarded ``cleaved_reporter_nm`` from the same
    object. So a user who did the work of measuring constants and passed
    ``--kinetics-constants`` got a ranking that looked identical and no figure
    in real units anywhere.

    It is reported and never ranked on: an axis that is a rescaling of another
    axis does not earn a vote, and ``predicted_signal`` is already there.
    """

    @staticmethod
    def _measured_constants_file():
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

    def _report(self, **extra):
        import pathlib
        import random
        import tempfile

        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(9)
        sequence = "".join(rng.choice("ACGU") for _ in range(300))
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "t.fa"
            path.write_text(">t1" + chr(10) + sequence + chr(10), encoding="utf-8")
            return run_request(
                DesignRequest(fasta_path=str(path), task="detection",
                              assay="in_vitro_detection", effector="LwaCas13a",
                              use_structure=False, use_signal=True, **extra)
            ).report

    def test_measured_constants_put_a_figure_in_real_units_on_the_row(self):
        rows = self._report(kinetics_json=str(self._measured_constants_file()))["ranked"]
        self.assertTrue(rows)
        values = [row["predicted_signal_nm"] for row in rows]
        self.assertTrue(all(value is not None for value in values))
        self.assertTrue(any(value > 0 for value in values))

    def test_placeholder_constants_leave_it_absent(self):
        """The refusal ``absolute_signal`` makes, kept on the reporting path:
        real units from placeholder rates are a fabricated measurement wearing
        a unit label."""
        rows = self._report()["ranked"]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(guide=row["guide_id"]):
                self.assertIsNone(row["predicted_signal_nm"])

    def test_the_unlock_adds_a_number_without_moving_the_ranking(self):
        """The fixture's constants are the shipped values with ``provisional``
        cleared, so nothing about the physics changed -- only whether the
        figure may be shown. The order must not move."""
        placeholders = self._report()["ranked"]
        measured = self._report(
            kinetics_json=str(self._measured_constants_file()))["ranked"]

        self.assertEqual(
            [row["guide_id"] for row in placeholders],
            [row["guide_id"] for row in measured],
        )
        self.assertEqual(
            [row["predicted_signal"] for row in placeholders],
            [row["predicted_signal"] for row in measured],
        )

    def test_it_is_not_an_objective(self):
        """Reported, never ranked on. If it entered the objective set it would
        re-weight ``predicted_signal`` under another name."""
        from cas13guide.models.evidence import MetricVector

        vector = MetricVector(0.5, 0.5, 0.1, 0.1, 0.5, 1.0,
                              predicted_signal=0.4, predicted_signal_nm=12.5)
        self.assertNotIn("predicted_signal_nm", vector.objectives())

    def test_the_csv_carries_the_column_either_way(self):
        """Blank when the figure was not earned, which is the point: an empty
        column says so, where a missing column says nothing."""
        import csv
        import io

        from cas13guide.services.fasta import RANKED_COLUMNS, ranked_to_csv

        self.assertIn("predicted_signal_nm", RANKED_COLUMNS)
        for extra, expect_value in (
            ({}, False),
            ({"kinetics_json": str(self._measured_constants_file())}, True),
        ):
            with self.subTest(measured=expect_value):
                text = ranked_to_csv(self._report(**extra))
                body = chr(10).join(
                    line for line in text.splitlines() if not line.startswith("#")
                )
                first = next(iter(csv.DictReader(io.StringIO(body))))
                self.assertEqual(bool(first["predicted_signal_nm"]), expect_value)
