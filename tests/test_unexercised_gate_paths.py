"""The gate verdicts that 950 tests never produced.

Found by wrapping ``GateResult`` for a whole suite run and asking which of the
eighteen ``ReasonCode`` values ever came out: twelve did. Of the six that did
not, three are gate paths and three are orchestrator telemetry, and one of
those three had a test already.

The three gate paths matter most, because two of them sit behind knobs a user
can turn from the command line:

* ``TARGET_INACCESSIBLE`` fires only when the spec sets ``--min-accessibility``,
  which is exactly the flag somebody reaches for when structure matters to
  them. Until now, turning it on ran code no test had executed;
* ``DUPLEX_ENERGY_OUT_OF_RANGE`` fires only when the spec sets a duplex window,
  and it also picks which of two opposite explanations to print;
* ``BORDERLINE_OFF_TARGET`` needs no flag at all. It occupies the narrow band
  just under the hazard budget -- above it the guide fails outright -- and no
  fixture had happened to land a guide there.

These are cheap tests for paths a user reaches, which is the least a suite can
be asked for.
"""

import unittest

from helpers import *  # noqa: F401,F403


def _gate(spec, **metric_overrides):
    """Run the gate on a guide that is otherwise unremarkable.

    Coverage is set high and hazard low so nothing FAILs: the FAIL branch
    returns before the REVIEW checks run, and a fixture that trips it would
    test the wrong thing.
    """
    from cas13guide.models.evidence import MetricVector
    from cas13guide.pipeline.f_decide import gate

    fields = {
        "required_coverage": 1.0,
        "desired_utility": 1.0,
        "worst_off_target": 0.0,
        "cumulative_burden": 0.0,
        "robustness": 1.0,
        "manufacturability": 1.0,
    }
    fields.update(metric_overrides)
    evidence = (
        make_evidence(0.9, 1.0, target_id="t1", target_class=TargetClass.REQUIRED),
    )
    return gate(
        MetricVector(**fields),
        clean_qc(),
        evidence,
        exhaustive_bound(),
        frozenset(),
        spec,
    )


def _reasons(result):
    return {code.value for code in result.reasons}


class TestBorderlineOffTarget(unittest.TestCase):
    """The band just below the hazard budget, where a guide is not refused but
    should not be taken at face value either."""

    def test_a_guide_just_under_the_budget_is_flagged_for_review(self):
        from cas13guide.contract.enums import ReasonCode, ReviewTier, Verdict

        spec = make_spec(worst_off_target_max=1.0, borderline_band=0.10)
        result = _gate(spec, worst_off_target=0.95)

        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIn(ReasonCode.BORDERLINE_OFF_TARGET.value, _reasons(result))
        self.assertIs(result.review_tier, ReviewTier.R1)
        self.assertTrue(any("0.950" in d for d in result.detail))

    def test_a_guide_over_the_budget_fails_instead_of_being_flagged(self):
        """FAIL and REVIEW are not both available: the hard stop returns first,
        and a guide that is refused must not also appear as a soft caution."""
        from cas13guide.contract.enums import ReasonCode, Verdict

        result = _gate(make_spec(worst_off_target_max=1.0), worst_off_target=1.5)

        self.assertIs(result.verdict, Verdict.FAIL)
        self.assertNotIn(ReasonCode.BORDERLINE_OFF_TARGET.value, _reasons(result))

    def test_a_guide_well_under_the_budget_is_not_flagged(self):
        from cas13guide.contract.enums import ReasonCode

        result = _gate(make_spec(worst_off_target_max=1.0), worst_off_target=0.2)
        self.assertNotIn(ReasonCode.BORDERLINE_OFF_TARGET.value, _reasons(result))

    def test_the_band_width_is_what_decides_it(self):
        """0.85 is inside a 20% band and outside a 10% one. If the threshold
        ignored the configured width this would not move."""
        from cas13guide.contract.enums import ReasonCode

        narrow = _gate(make_spec(borderline_band=0.10), worst_off_target=0.85)
        wide = _gate(make_spec(borderline_band=0.20), worst_off_target=0.85)

        self.assertNotIn(ReasonCode.BORDERLINE_OFF_TARGET.value, _reasons(narrow))
        self.assertIn(ReasonCode.BORDERLINE_OFF_TARGET.value, _reasons(wide))


class TestTargetInaccessible(unittest.TestCase):
    """FR3.4, and the code behind ``--min-accessibility``."""

    def test_a_buried_site_is_flagged_once_a_floor_is_set(self):
        from cas13guide.contract.enums import ReasonCode, ReviewTier, Verdict

        result = _gate(make_spec(min_accessibility=0.5), accessibility=0.12)

        self.assertIs(result.verdict, Verdict.REVIEW)
        self.assertIn(ReasonCode.TARGET_INACCESSIBLE.value, _reasons(result))
        self.assertIs(result.review_tier, ReviewTier.R1)
        self.assertTrue(
            any("0.120" in d and "0.500" in d for d in result.detail),
            f"the detail should carry both numbers: {result.detail}",
        )

    def test_it_is_silent_when_no_floor_is_set(self):
        """"Off unless the spec sets a floor, because what counts as too
        structured is an assay decision." The default must not flag."""
        from cas13guide.contract.enums import ReasonCode

        result = _gate(make_spec(), accessibility=0.01)
        self.assertNotIn(ReasonCode.TARGET_INACCESSIBLE.value, _reasons(result))

    def test_an_open_site_is_not_flagged(self):
        from cas13guide.contract.enums import ReasonCode

        result = _gate(make_spec(min_accessibility=0.5), accessibility=0.9)
        self.assertNotIn(ReasonCode.TARGET_INACCESSIBLE.value, _reasons(result))

    def test_no_accessibility_model_is_not_an_inaccessible_target(self):
        """``None`` means nobody looked. Treating absent evidence as a bad
        number would flag every run made without a folding engine."""
        from cas13guide.contract.enums import ReasonCode

        result = _gate(make_spec(min_accessibility=0.5), accessibility=None)
        self.assertNotIn(ReasonCode.TARGET_INACCESSIBLE.value, _reasons(result))


class TestDuplexEnergyOutOfRange(unittest.TestCase):
    """FR5.2. Free energy has an optimum rather than a direction, so this path
    also has to choose which of two opposite explanations to give."""

    WINDOW = {"duplex_dg_per_nt_min": -0.50, "duplex_dg_per_nt_max": -0.20}

    def test_an_over_stable_duplex_is_flagged_as_over_stable(self):
        from cas13guide.contract.enums import ReasonCode

        result = _gate(make_spec(**self.WINDOW), duplex_dg_per_nt=-0.90)

        self.assertIn(ReasonCode.DUPLEX_ENERGY_OUT_OF_RANGE.value, _reasons(result))
        detail = " ".join(result.detail)
        self.assertIn("over-stable", detail)
        self.assertNotIn("too weak", detail)

    def test_a_weak_duplex_is_flagged_as_weak(self):
        from cas13guide.contract.enums import ReasonCode

        result = _gate(make_spec(**self.WINDOW), duplex_dg_per_nt=-0.05)

        self.assertIn(ReasonCode.DUPLEX_ENERGY_OUT_OF_RANGE.value, _reasons(result))
        detail = " ".join(result.detail)
        self.assertIn("too weak", detail)
        self.assertNotIn("over-stable", detail)

    def test_inside_the_window_is_not_flagged(self):
        from cas13guide.contract.enums import ReasonCode

        result = _gate(make_spec(**self.WINDOW), duplex_dg_per_nt=-0.35)
        self.assertNotIn(ReasonCode.DUPLEX_ENERGY_OUT_OF_RANGE.value, _reasons(result))

    def test_it_is_silent_when_the_spec_chose_no_window(self):
        """"the optimum is assay-specific and this package does not assert
        one" -- so an unconfigured run must not invent a verdict."""
        from cas13guide.contract.enums import ReasonCode

        for value in (-5.0, -0.001):
            with self.subTest(duplex=value):
                result = _gate(make_spec(), duplex_dg_per_nt=value)
                self.assertNotIn(
                    ReasonCode.DUPLEX_ENERGY_OUT_OF_RANGE.value, _reasons(result)
                )

    def test_one_sided_windows_work(self):
        """Both bounds are independently optional, and a half-open window is a
        reasonable thing for an assay to state."""
        from cas13guide.contract.enums import ReasonCode

        floor_only = _gate(make_spec(duplex_dg_per_nt_min=-0.5), duplex_dg_per_nt=-0.9)
        ceiling_only = _gate(make_spec(duplex_dg_per_nt_max=-0.2), duplex_dg_per_nt=-0.05)

        self.assertIn(ReasonCode.DUPLEX_ENERGY_OUT_OF_RANGE.value, _reasons(floor_only))
        self.assertIn(ReasonCode.DUPLEX_ENERGY_OUT_OF_RANGE.value, _reasons(ceiling_only))

    def test_no_duplex_model_is_not_an_out_of_range_duplex(self):
        from cas13guide.contract.enums import ReasonCode

        result = _gate(make_spec(**self.WINDOW), duplex_dg_per_nt=None)
        self.assertNotIn(ReasonCode.DUPLEX_ENERGY_OUT_OF_RANGE.value, _reasons(result))


if __name__ == "__main__":
    unittest.main()
