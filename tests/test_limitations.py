"""Limitations as data, derived from the run (NFR6).

The whole value is in the *derivation*. A static list has two failure modes and
this package would hit both: it keeps warning about placeholder constants after
real ones are supplied, and it stays silent about a heuristic index because
nobody added the entry. So most of these tests are of the form "this warning
appears exactly when the condition holds, and not otherwise".
"""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import AssayType, Effector, Task, Universe
from cas13guide.contract.spec import (
    ContextScope,
    EffectorAssayProfile,
    ResolvedDesignSpec,
    TargetContract,
    TargetSpec,
)
from cas13guide.services.limitations import (
    Limitation,
    Severity,
    as_lines,
    limitations_for_run,
    worst_severity,
)


def spec_for(assay=AssayType.IN_VITRO_DETECTION, effector=Effector.LWA_CAS13A,
             spacer_length=None, task=Task.DETECTION):
    return ResolvedDesignSpec(
        contract=TargetContract(
            task=task,
            targets=(TargetSpec(target_id="t1", sequence=SEQ_A,
                                target_class=TargetClass.REQUIRED),),
        ),
        profile=EffectorAssayProfile(effector=effector, assay=assay,
                                     spacer_length=spacer_length),
        context=ContextScope(organism="Homo sapiens", genome_build="GRCh38",
                             annotation_version="v44",
                             declared_universes=FULL_UNIVERSES),
    )


def ids(limitations):
    return {item.id for item in limitations}


class _Model:
    def __init__(self, version, **extra):
        self.version = version
        for key, value in extra.items():
            setattr(self, key, value)


class _Registry:
    def __init__(self, activity=None, accessibility=None, kinetics=None):
        self.activity = activity or _Model("model/ridge-logistic/v1")
        self.accessibility = accessibility or _Model(
            "accessibility/static-probing/v1", is_independent_evidence=True
        )
        self.kinetics = kinetics or _Model("kinetics/none/v1")


class _Constants:
    def __init__(self, provisional):
        self.provisional = provisional


class TestDerivationNotDeclaration(unittest.TestCase):
    """Each warning must track the condition it describes, in both directions."""

    def test_synthetic_calibration_is_flagged_only_when_marked(self):
        class Marked:
            synthetic = True

        self.assertIn("synthetic-calibration",
                      ids(limitations_for_run(spec_for(), calibration=Marked())))
        self.assertNotIn("synthetic-calibration",
                         ids(limitations_for_run(spec_for(), calibration=object())))

    def test_the_heuristic_model_is_flagged_and_a_fitted_one_is_not(self):
        heuristic = _Registry(activity=_Model("model/heuristic-cas13/v1"))
        fitted = _Registry(activity=_Model("model/ridge-logistic/v1"))
        self.assertIn("heuristic-activity-model",
                      ids(limitations_for_run(spec_for(), registry=heuristic)))
        self.assertNotIn("heuristic-activity-model",
                         ids(limitations_for_run(spec_for(), registry=fitted)))

    def test_missing_accessibility_and_predicted_accessibility_are_different(self):
        none = _Registry(accessibility=_Model("accessibility/none/v1"))
        fold = _Registry(accessibility=_Model("accessibility/vienna-rnaplfold/v1",
                                              is_independent_evidence=False))
        probing = _Registry(accessibility=_Model("accessibility/static-probing/v1",
                                                 is_independent_evidence=True))
        self.assertIn("no-accessibility", ids(limitations_for_run(spec_for(), registry=none)))
        self.assertIn("accessibility-is-predicted",
                      ids(limitations_for_run(spec_for(), registry=fold)))
        # a real probing track is independent evidence, so neither applies
        found = ids(limitations_for_run(spec_for(), registry=probing))
        self.assertNotIn("no-accessibility", found)
        self.assertNotIn("accessibility-is-predicted", found)

    def test_placeholder_constants_are_flagged_until_they_are_real(self):
        provisional = _Registry(kinetics=_Model("k", constants=_Constants(True)))
        measured = _Registry(kinetics=_Model("k", constants=_Constants(False)))
        self.assertIn("provisional-kinetics",
                      ids(limitations_for_run(spec_for(), registry=provisional)))
        self.assertNotIn("provisional-kinetics",
                         ids(limitations_for_run(spec_for(), registry=measured)))

    def test_a_non_canonical_spacer_length_is_flagged_and_a_canonical_one_is_not(self):
        self.assertIn(
            "non-canonical-spacer-length",
            ids(limitations_for_run(spec_for(spacer_length=24))),
        )
        self.assertNotIn(
            "non-canonical-spacer-length",
            ids(limitations_for_run(spec_for(spacer_length=28))),
        )


class TestAssayDependentWarnings(unittest.TestCase):
    """Collateral is the readout of one assay and a hazard in another."""

    def test_a_knockdown_assay_is_warned_about_collateral(self):
        found = ids(limitations_for_run(
            spec_for(assay=AssayType.CELL_CULTURE_KNOCKDOWN, effector=Effector.RFX_CAS13D,
                     task=Task.KNOCKDOWN)))
        self.assertIn("collateral-not-predicted", found)

    def test_a_detection_assay_is_not(self):
        """There collateral is the signal, and it *is* modelled."""
        found = ids(limitations_for_run(spec_for(assay=AssayType.IN_VITRO_DETECTION)))
        self.assertNotIn("collateral-not-predicted", found)

    def test_the_placeholder_warning_is_detection_only(self):
        """A knockdown run does not rank on signal, so the caveat is noise there."""
        registry = _Registry(kinetics=_Model("k", constants=_Constants(True)))
        knockdown = ids(limitations_for_run(
            spec_for(assay=AssayType.CELL_CULTURE_KNOCKDOWN,
                     effector=Effector.RFX_CAS13D, task=Task.KNOCKDOWN),
            registry=registry))
        self.assertNotIn("provisional-kinetics", knockdown)


class TestSearchScope(unittest.TestCase):
    def test_a_heuristic_bound_is_called_out(self):
        found = ids(limitations_for_run(spec_for(), bound=exhaustive_bound(4, exhaustive=False)))
        self.assertIn("search-not-exhaustive", found)

    def test_an_exhaustive_bound_is_not(self):
        found = ids(limitations_for_run(spec_for(), bound=exhaustive_bound(4)))
        self.assertNotIn("search-not-exhaustive", found)

    def test_scope_is_always_stated_when_a_search_ran(self):
        """Even an exhaustive search is only exhaustive over what was loaded."""
        found = limitations_for_run(spec_for(), bound=exhaustive_bound(4))
        scope = next(i for i in found if i.id == "search-scope-is-what-you-loaded")
        self.assertIs(scope.severity, Severity.CRITICAL)
        for universe in FULL_UNIVERSES:
            self.assertIn(universe.value, scope.text)

    def test_a_universe_gap_is_critical_and_names_the_universes(self):
        found = limitations_for_run(
            spec_for(), universe_gaps=frozenset({Universe.RRNA, Universe.MITO})
        )
        gap = next(i for i in found if i.id == "universe-gap")
        self.assertIs(gap.severity, Severity.CRITICAL)
        self.assertIn("rrna", gap.text)
        self.assertIn("mito", gap.text)

    def test_a_non_wobble_aware_search_is_called_out(self):
        from cas13guide.services.indexes import ExhaustivenessBound

        bound = ExhaustivenessBound(
            max_mismatches=4, wobble_aware=False, universes=FULL_UNIVERSES,
            exhaustive=False, method="blast", note="seeded",
        )
        self.assertIn("search-not-wobble-aware", ids(limitations_for_run(spec_for(), bound=bound)))


class TestPresentation(unittest.TestCase):
    def test_critical_limitations_come_first(self):
        found = limitations_for_run(
            spec_for(), registry=_Registry(activity=_Model("model/heuristic-cas13/v1")),
            bound=exhaustive_bound(4),
        )
        severities = [i.severity for i in found]
        self.assertEqual(severities, sorted(severities, key=lambda s: {
            Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}[s]))

    def test_the_order_is_deterministic(self):
        a = limitations_for_run(spec_for(), bound=exhaustive_bound(4))
        b = limitations_for_run(spec_for(), bound=exhaustive_bound(4))
        self.assertEqual([i.id for i in a], [i.id for i in b])

    def test_every_run_says_scores_are_predictions(self):
        self.assertIn("predictions-not-measurements", ids(limitations_for_run(spec_for())))

    def test_lines_carry_the_severity_and_the_source(self):
        item = Limitation("x", Severity.CRITICAL, "text here", source="Someone et al.")
        self.assertEqual(item.line(), "CRITICAL: text here [Someone et al.]")
        self.assertEqual(
            Limitation("y", Severity.INFO, "no source").line(), "INFO: no source"
        )

    def test_worst_severity_reports_the_most_serious(self):
        self.assertIsNone(worst_severity([]))
        self.assertIs(
            worst_severity([
                Limitation("a", Severity.INFO, "a"),
                Limitation("b", Severity.CRITICAL, "b"),
            ]),
            Severity.CRITICAL,
        )

    def test_as_lines_round_trips_every_limitation(self):
        found = limitations_for_run(spec_for(), bound=exhaustive_bound(4))
        self.assertEqual(len(as_lines(found)), len(found))


if __name__ == "__main__":
    unittest.main()
