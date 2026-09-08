"""The two limitations the suite never produced.

The same question that found three unexercised gate paths, asked of the other
thing this package says to users: 21 ``Limitation`` ids are declared and 19 are
produced somewhere in the suite. A warning nothing has ever emitted is a
sentence a user can receive that nobody has read in context.

Both survivors are conditional on state no fixture creates:

* ``panel-confidence-thin-calibration`` needs a panel confidence object whose
  members rest on a thin calibration window. Most runs supply no confidence
  model at all, so the branch was never reached -- and this is the warning that
  tells a reader their interval is wide because of missing data rather than
  because the guide is doubtful, which is precisely the distinction FR6.3
  exists to make;
* ``preset-sequence-unverified`` is the fallback for ``PRESETS_VERIFIED =
  False``, and the constant ships as ``True``. It cannot fire as shipped, which
  is not the same as being dead: it is what a contributor gets the moment they
  add a preset they have not checked, and it should work the first time.
"""

import unittest
from unittest import mock

from helpers import *  # noqa: F401,F403


def _limitations(**kwargs):
    from cas13guide.services.limitations import limitations_for_run

    spec = kwargs.pop("spec", None) or make_spec()
    return {limitation.id: limitation for limitation in limitations_for_run(spec, **kwargs)}


def _confidence(*, thin: bool):
    """A minimal panel confidence with one target and one member."""
    from cas13guide.services.panel_confidence import (
        GuideSuccess,
        PanelConfidence,
        TargetConfidence,
    )

    member = GuideSuccess(
        guide_id="g1", target_id="t1", prediction=0.8, trials=3 if thin else 400,
        successes=2 if thin else 320, posterior_a=2.5, posterior_b=1.5,
        credible_lower=0.2, credible_upper=0.95,
        stratum="mm0" if not thin else "none", thin=thin,
    )
    target = TargetConfidence(
        target_id="t1", members=(member,), lower=0.6, upper=0.6,
        credible_lower=0.2, credible_upper=0.95, thin=thin,
    )
    return PanelConfidence(version="panel-confidence/test", targets=(target,))


class TestTheThinCalibrationWarning(unittest.TestCase):
    """FR6.3's whole point is separating "wide because we lack data" from
    "wide because the guide is doubtful". This is the sentence that says so."""

    def test_it_fires_when_a_member_rests_on_a_thin_window(self):
        from cas13guide.services.limitations import Severity

        found = _limitations(panel_confidence=_confidence(thin=True))

        self.assertIn("panel-confidence-thin-calibration", found)
        limitation = found["panel-confidence-thin-calibration"]
        self.assertIs(limitation.severity, Severity.WARNING)
        self.assertIn("thin", limitation.text)

    def test_it_says_the_interval_is_about_data_not_doubt(self):
        """The distinction is the entire content of the warning, and it is what
        stops a reader reading a wide interval as a bad guide."""
        text = _limitations(
            panel_confidence=_confidence(thin=True)
        )["panel-confidence-thin-calibration"].text
        self.assertIn("how little data", text)
        self.assertIn("larger screen", text)

    def test_it_is_silent_when_the_calibration_is_not_thin(self):
        found = _limitations(panel_confidence=_confidence(thin=False))
        self.assertNotIn("panel-confidence-thin-calibration", found)

    def test_it_is_silent_when_no_confidence_was_computed(self):
        """No model is not a thin model: absent evidence must not be reported
        as evidence of thinness."""
        self.assertNotIn("panel-confidence-thin-calibration", _limitations())


class TestTheUnverifiedPresetWarning(unittest.TestCase):
    """Unreachable as shipped, and reachable the moment somebody adds a preset
    they have not checked -- which is exactly when it needs to work."""

    def _preset_spec(self):
        from cas13guide.contract.presets import MIRNA_PRESETS
        from cas13guide.contract.spec import TargetSpec
        from cas13guide.contract.enums import TargetClass

        preset_id = sorted(MIRNA_PRESETS)[0]
        preset = MIRNA_PRESETS[preset_id]
        return make_spec(targets=(
            TargetSpec(target_id=preset_id, sequence=preset.sequence,
                       target_class=TargetClass.REQUIRED, expression=100.0),
        )), preset_id

    def test_it_fires_when_the_presets_are_not_verified(self):
        from cas13guide.services.limitations import Severity

        spec, preset_id = self._preset_spec()
        with mock.patch("cas13guide.services.limitations.PRESETS_VERIFIED", False):
            found = _limitations(spec=spec)

        self.assertIn("preset-sequence-unverified", found)
        limitation = found["preset-sequence-unverified"]
        self.assertIs(limitation.severity, Severity.WARNING)
        self.assertIn(preset_id, limitation.text)

    def test_the_verified_and_unverified_branches_are_exclusive(self):
        """Two sentences about the same preset, one saying it was checked and
        one saying it was not, would be worse than either alone."""
        spec, _ = self._preset_spec()

        verified = _limitations(spec=spec)
        with mock.patch("cas13guide.services.limitations.PRESETS_VERIFIED", False):
            unverified = _limitations(spec=spec)

        self.assertNotIn("preset-sequence-unverified", verified)
        self.assertNotIn("preset-sequence-changed-upstream", unverified)

    def test_neither_fires_for_a_spec_that_uses_no_preset(self):
        with mock.patch("cas13guide.services.limitations.PRESETS_VERIFIED", False):
            found = _limitations()
        self.assertNotIn("preset-sequence-unverified", found)


if __name__ == "__main__":
    unittest.main()
