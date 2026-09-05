"""Accessibility (FR3.x), duplex energy (FR5.x) and signal prediction (FR6.x).

The ViennaRNA-backed tests skip themselves when it is not installed, because
the package is designed to run without it -- the Null models report "no
evidence" rather than guessing, and that path has to keep working.
"""

import json
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import AssayType, Effector, TargetClass, Task, Universe
from cas13guide.contract.spec import (
    ContextScope,
    EffectorAssayProfile,
    ResolvedDesignSpec,
    TargetContract,
    TargetSpec,
    Thresholds,
)
from cas13guide.metrics.aggregate import build_metric_vector, predicted_signal
from cas13guide.services.kinetics import (
    PROVISIONAL_CONSTANTS,
    CollateralKineticsModel,
    KineticConstants,
    NullKineticsModel,
    ProvisionalConstantsError,
)
from cas13guide.services.registry import (
    HeuristicCas13Model,
    ModelRegistry,
    NullAccessibilityModel,
    NullDuplexEnergyModel,
)

try:  # pragma: no cover - environment dependent
    import RNA  # noqa: F401

    HAVE_VIENNA = True
except ImportError:  # pragma: no cover
    HAVE_VIENNA = False

needs_vienna = unittest.skipUnless(HAVE_VIENNA, "ViennaRNA not installed")

SPACER = "GCAUUGCUGUGGAGCUGAAGUC"


# ---------------------------------------------------------------------------
# FR3.x -- structural accessibility
# ---------------------------------------------------------------------------


@needs_vienna
class TestAccessibility(unittest.TestCase):
    def setUp(self):
        from cas13guide.services.accessibility import ViennaRNAAccessibilityModel

        self.model = ViennaRNAAccessibilityModel()

    def test_a_stem_is_less_accessible_than_an_unpaired_tail(self):
        """The whole point: buried sites must score below open ones."""
        sequence = "GGGGGGGGGGAAAACCCCCCCCCC" + "A" * 40
        stem = self.model.accessibility(sequence, 0, 10)
        tail = self.model.accessibility(sequence, 30, 20)
        self.assertLess(stem, 0.1)
        self.assertGreater(tail, 0.9)
        self.assertLess(stem, tail)

    def test_probabilities_stay_in_the_unit_interval(self):
        import random

        rng = random.Random(4)
        sequence = "".join(rng.choice("AUGC") for _ in range(400))
        profile = self.model.unpaired_profile(sequence)
        self.assertEqual(len(profile), len(sequence))
        for value in profile:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_a_window_outside_the_sequence_is_no_evidence_not_a_guess(self):
        sequence = "AUGCAUGCAUGCAUGCAUGCAU"
        self.assertIsNone(self.model.accessibility(sequence, 0, len(sequence) + 1))
        self.assertIsNone(self.model.accessibility(sequence, -1, 5))
        self.assertIsNone(self.model.accessibility(sequence, 0, 0))

    def test_profiles_are_cached_per_sequence(self):
        sequence = "AUGC" * 40
        first = self.model.unpaired_profile(sequence)
        second = self.model.unpaired_profile(sequence)
        self.assertIs(first, second)

    def test_a_fold_is_not_independent_evidence(self):
        """It is a prediction from the sequence the activity model already sees."""
        self.assertFalse(self.model.is_independent_evidence)

    def test_a_probing_track_is_independent_evidence(self):
        from cas13guide.services.accessibility import StaticAccessibilityModel

        sequence = "AUGCAUGCAUGCAUGCAUGCAU"
        probing = StaticAccessibilityModel({sequence: tuple([0.8] * len(sequence))})
        self.assertTrue(probing.is_independent_evidence)
        self.assertAlmostEqual(probing.accessibility(sequence, 0, 5), 0.8)
        self.assertIsNone(probing.accessibility("AUGC", 0, 4))  # unknown sequence


class TestAccessibilityIndependenceRule(unittest.TestCase):
    """A predicted fold must not tighten a calibrated bound."""

    def _evidence(self, accessibility_model):
        from cas13guide.pipeline.d_pair_evidence import evidence_for_hit
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.transcriptome import Locus, Transcriptome

        spec = make_spec()
        transcriptome = Transcriptome("GRCh38", "v44")
        transcriptome.add(Locus("L1", SEQ_A, Universe.CODING, expression=10.0))
        registry = ModelRegistry(HeuristicCas13Model(), accessibility_model)

        class _Expression:
            version = "test"

            def expression_tpm(self, locus_id):
                return 10.0

        return evidence_for_hit(
            make_hit("g1", "L1", "t1", TargetClass.REQUIRED),
            SPACER,
            spec,
            registry,
            CalibrationService(),
            transcriptome,
            _Expression(),
        )

    def test_a_dependent_model_is_recorded_but_flagged_as_predicted(self):
        class Predicted:
            version = "accessibility/fake-fold/v1"
            is_independent_evidence = False

            def accessibility(self, sequence, start, length):
                return 0.7

        evidence = self._evidence(Predicted())
        self.assertIsNotNone(evidence.accessibility)
        self.assertAlmostEqual(evidence.accessibility.point, 0.7)
        self.assertEqual(evidence.accessibility.stratum, "accessibility/predicted")

    def test_an_independent_model_is_recorded_as_measured(self):
        class Probed:
            version = "accessibility/fake-shape/v1"
            is_independent_evidence = True

            def accessibility(self, sequence, start, length):
                return 0.7

        evidence = self._evidence(Probed())
        self.assertEqual(evidence.accessibility.stratum, "accessibility")

    def test_no_model_means_no_accessibility_at_all(self):
        evidence = self._evidence(NullAccessibilityModel())
        self.assertIsNone(evidence.accessibility)


# ---------------------------------------------------------------------------
# FR5.x -- duplex free energy
# ---------------------------------------------------------------------------


@needs_vienna
class TestDuplexEnergy(unittest.TestCase):
    def setUp(self):
        from cas13guide.services.thermodynamics import duplex_free_energy

        self.dg = duplex_free_energy

    def test_a_perfect_duplex_is_more_stable_than_a_mismatched_one(self):
        from cas13guide.util.seq import revcomp

        perfect = self.dg(SPACER, revcomp(SPACER))
        site = list(revcomp(SPACER))
        for i in (3, 9, 15):
            site[i] = "C" if site[i] != "C" else "G"
        mismatched = self.dg(SPACER, "".join(site))
        self.assertLess(perfect.dg, mismatched.dg)

    def test_per_nucleotide_normalization_is_length_independent(self):
        from cas13guide.util.seq import revcomp

        short = self.dg(SPACER, revcomp(SPACER))
        self.assertAlmostEqual(short.dg_per_nt, short.dg / len(SPACER))

    def test_range_checking_reports_both_failure_directions(self):
        from cas13guide.util.seq import revcomp

        energy = self.dg(SPACER, revcomp(SPACER))
        self.assertTrue(energy.in_range(None, None))          # open range
        self.assertFalse(energy.in_range(energy.dg_per_nt + 0.1, None))
        self.assertIn("over-stable", energy.describe(energy.dg_per_nt + 0.1, None))
        self.assertFalse(energy.in_range(None, energy.dg_per_nt - 0.1))
        self.assertIn("weaker", energy.describe(None, energy.dg_per_nt - 0.1))

    def test_summarize_derives_a_range_from_data(self):
        from cas13guide.services.thermodynamics import summarize
        from cas13guide.util.seq import revcomp, windows

        pairs = [(revcomp(site), site) for _s, site in windows(SEQ_A, 22)]
        report = summarize(pairs)
        self.assertEqual(report.n, len(pairs))
        low, high = report.suggested_range()
        self.assertLess(low, high)
        self.assertLessEqual(report.min_per_nt, report.mean_per_nt)
        self.assertGreaterEqual(report.max_per_nt, report.mean_per_nt)

    def test_empty_sequences_are_rejected(self):
        with self.assertRaises(ValueError):
            self.dg("", "AUGC")


class TestEnergyIsNotAParetoObjective(unittest.TestCase):
    """It has an optimum, not a direction, so it must never be maximized."""

    def test_duplex_energy_is_absent_from_objectives(self):
        from cas13guide.models.evidence import MetricVector

        metrics = MetricVector(0.9, 0.0, 0.0, 0.0, 0.9, 1.0, duplex_dg_per_nt=-1.8)
        self.assertNotIn("duplex_dg_per_nt", metrics.objectives())

    def test_but_accessibility_is_present_because_it_is_monotone(self):
        from cas13guide.models.evidence import MetricVector

        metrics = MetricVector(0.9, 0.0, 0.0, 0.0, 0.9, 1.0, accessibility=0.6)
        self.assertEqual(metrics.objectives()["accessibility"], 0.6)

    def test_null_energy_model_reports_no_evidence(self):
        self.assertIsNone(NullDuplexEnergyModel().duplex_dg_per_nt(SPACER, SPACER))


# ---------------------------------------------------------------------------
# FR6.x -- collateral kinetics and predicted signal
# ---------------------------------------------------------------------------


class TestCollateralKinetics(unittest.TestCase):
    def setUp(self):
        self.model = CollateralKineticsModel()

    def test_signal_rises_monotonically_with_activity(self):
        values = [self.model.predict(i / 40).relative_signal for i in range(41)]
        for a, b in zip(values, values[1:]):
            self.assertGreaterEqual(b, a - 1e-12)
        self.assertGreater(values[-1], values[0])

    def test_signal_rises_monotonically_with_accessibility(self):
        values = [
            self.model.predict(0.6, accessibility=i / 40).relative_signal
            for i in range(41)
        ]
        for a, b in zip(values, values[1:]):
            self.assertGreaterEqual(b, a - 1e-12)

    def test_no_activity_means_no_signal(self):
        self.assertEqual(self.model.predict(0.0).relative_signal, 0.0)
        self.assertEqual(self.model.predict(0.9, accessibility=0.0).relative_signal, 0.0)

    def test_activity_is_read_at_the_lower_bound(self):
        """A bright signal off the wide end of an interval is a hope, not a
        prediction -- consistent with every other benefit in the package."""
        self.assertLess(
            self.model.effective_target_nm(0.2), self.model.effective_target_nm(0.8)
        )
        self.assertEqual(self.model.effective_target_nm(1.5), self.model.constants.target_nm)

    def test_the_default_constants_do_not_saturate_the_reporter(self):
        """A saturating assay ranks every candidate 1.0, which is no ranking."""
        best = self.model.predict(1.0, accessibility=1.0)
        self.assertFalse(best.saturated, best)
        self.assertLess(best.relative_signal, 0.99)

    def test_it_discriminates_across_the_plausible_range(self):
        values = {round(self.model.predict(i / 40).relative_signal, 6) for i in range(41)}
        self.assertGreater(len(values), 30)

    def test_peak_rate_is_reported_not_the_meaningless_t0_slope(self):
        """The system starts with zero activated complex, so its t=0 rate is
        identically zero for every candidate."""
        self.assertGreater(self.model.predict(0.8).cleavage_rate, 0.0)


class TestProvisionalConstantsBlockAbsoluteClaims(unittest.TestCase):
    """Relative ranking is invariant to the constants; absolute units are not."""

    def test_relative_signal_works_while_provisional(self):
        model = CollateralKineticsModel()
        self.assertTrue(model.constants.provisional)
        self.assertGreater(model.predict(0.8).relative_signal, 0.0)
        self.assertTrue(model.predict(0.8).provisional)

    def test_absolute_signal_refuses_while_provisional(self):
        model = CollateralKineticsModel()
        with self.assertRaises(ProvisionalConstantsError) as caught:
            model.absolute_signal(0.8)
        message = str(caught.exception)
        self.assertIn("provisional", message)
        self.assertIn("fabricated", message)

    def test_measured_constants_unlock_it(self):
        measured = KineticConstants(
            **{
                **{f: getattr(PROVISIONAL_CONSTANTS, f)
                   for f in PROVISIONAL_CONSTANTS.__dataclass_fields__},
                "source": "Feng et al. 2024",
                "provisional": False,
            }
        )
        model = CollateralKineticsModel(measured)
        self.assertGreater(model.absolute_signal(0.8), 0.0)
        self.assertIn("measured", model.version)

    def test_the_version_string_says_which_it_is(self):
        self.assertIn("provisional", CollateralKineticsModel().version)

    def test_constants_round_trip_through_json(self):
        """DR4: the constants are configuration, not code."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kinetics.json"
            PROVISIONAL_CONSTANTS.to_json(path)
            loaded = KineticConstants.from_json(path)
            self.assertEqual(loaded, PROVISIONAL_CONSTANTS)
            self.assertIn("k_cat", json.loads(path.read_text(encoding="utf-8")))


class TestSignalOnlyCountsWhereCollateralIsTheReadout(unittest.TestCase):
    """In a knockdown assay collateral is a hazard, not a benefit.

    Ranking guides by how much collateral cleavage they produce is right for a
    diagnostic and exactly inverted for a knockdown, so the objective is only
    emitted when ``collateral_is_hazard`` is False.
    """

    def _spec(self, assay):
        return ResolvedDesignSpec(
            contract=TargetContract(
                task=Task.DETECTION if assay is AssayType.IN_VITRO_DETECTION else Task.KNOCKDOWN,
                targets=(TargetSpec(target_id="t1", sequence=SEQ_A,
                                    target_class=TargetClass.REQUIRED),),
            ),
            profile=EffectorAssayProfile(effector=Effector.LWA_CAS13A, assay=assay),
            context=ContextScope(organism="Homo sapiens", genome_build="GRCh38",
                                 annotation_version="v44",
                                 declared_universes=FULL_UNIVERSES | {Universe.SYNTHETIC,
                                                                      Universe.VIRAL}),
        )

    def _evidence(self):
        return (make_evidence(0.8, 0.9, target_id="t1", target_class=TargetClass.REQUIRED),)

    def test_detection_assays_get_a_signal_prediction(self):
        spec = self._spec(AssayType.IN_VITRO_DETECTION)
        self.assertFalse(spec.profile.collateral_is_hazard)
        value = predicted_signal(self._evidence(), spec, CollateralKineticsModel())
        self.assertIsNotNone(value)
        self.assertGreater(value, 0.0)

    def test_the_signal_comes_from_the_intended_site_not_the_strongest_hit(self):
        """`_intended_prediction` skips evidence that is not the target.

        Nothing exercised that filter. Every fixture reaching this code had
        only intended evidence in it, so the `continue` never ran once in the
        whole suite and removing the filter changed no test -- a hole, found
        by branch coverage and confirmed by the sweep once the sweep could
        report one again.

        It matters because the filter is the only thing making this number
        mean what its docstring says. A guide with a strong off-target would
        otherwise report the collateral signal of the *off-target* as its own
        predicted fluorescence, which is a number about the wrong molecule.
        """
        spec = self._spec(AssayType.IN_VITRO_DETECTION)
        model = CollateralKineticsModel()
        intended = make_evidence(
            0.20, 0.30, target_id="t1", target_class=TargetClass.REQUIRED
        )
        # Unannotated, and far stronger than the intended site.
        off_target = make_evidence(0.95, 0.99, locus_id="L_OFF")

        self.assertFalse(off_target.is_intended, "fixture sanity")
        self.assertGreater(
            off_target.activity.lower, intended.activity.lower,
            "fixture does not discriminate: the off-target must be the one a "
            "max would pick",
        )

        alone = predicted_signal((intended,), spec, model)
        beside = predicted_signal((intended, off_target), spec, model)
        self.assertIsNotNone(alone)
        self.assertEqual(
            beside, alone,
            "an off-target changed the predicted signal, so the number "
            "describes the wrong site",
        )

    def test_the_absolute_figure_is_refused_on_placeholder_constants(self):
        """`predicted_signal_nm` had no caller and no test at all.

        Its sibling `predicted_signal` is public and tested; this one was
        reachable from nothing -- src, tests or examples -- because
        `build_metric_vector` inlines both readings to avoid running the
        collateral ODE twice. The dead-value sweep could not see it either:
        that walks fields, properties and methods, and a module-level function
        is none of those.

        The behaviour is worth having pinned here as well as through the
        report rows, because it is the one that refuses to fabricate: a
        fluorescence figure in real units derived from placeholder rate
        constants is a made-up measurement wearing a unit label, and nothing
        downstream can tell it from a measured one.
        """
        from cas13guide.metrics.aggregate import predicted_signal_nm
        from cas13guide.services.kinetics import KineticConstants

        spec = self._spec(AssayType.IN_VITRO_DETECTION)
        evidence = self._evidence()

        # The shipped default is provisional, and refuses.
        self.assertIsNone(
            predicted_signal_nm(evidence, spec, CollateralKineticsModel())
        )
        self.assertIsNone(predicted_signal_nm(evidence, spec, None))

        measured = KineticConstants(
            k_on=1.0, k_off=1e-4, k_cat=4e-3, k_m=200.0, complex_nm=50.0,
            target_nm=1.0, reporter_nm=500.0, assay_seconds=1800.0,
            source="measured", provisional=False,
        )
        value = predicted_signal_nm(
            evidence, spec, CollateralKineticsModel(measured)
        )
        self.assertIsNotNone(value, "measured constants must unlock the figure")
        self.assertGreater(value, 0.0)

    def test_the_absolute_figure_is_refused_where_collateral_is_a_hazard(self):
        """Same rule as the relative one: a knockdown assay gets neither."""
        from cas13guide.metrics.aggregate import predicted_signal_nm
        from cas13guide.services.kinetics import KineticConstants

        measured = KineticConstants(
            k_on=1.0, k_off=1e-4, k_cat=4e-3, k_m=200.0, complex_nm=50.0,
            target_nm=1.0, reporter_nm=500.0, assay_seconds=1800.0,
            source="measured", provisional=False,
        )
        spec = self._spec(AssayType.CELL_CULTURE_KNOCKDOWN)
        self.assertIsNone(
            predicted_signal_nm(
                self._evidence(), spec, CollateralKineticsModel(measured)
            )
        )

    def test_knockdown_assays_do_not(self):
        spec = self._spec(AssayType.CELL_CULTURE_KNOCKDOWN)
        self.assertTrue(spec.profile.collateral_is_hazard)
        self.assertIsNone(predicted_signal(self._evidence(), spec, CollateralKineticsModel()))

    def test_and_it_never_reaches_the_objective_set_there(self):
        spec = self._spec(AssayType.CELL_CULTURE_KNOCKDOWN)
        metrics = build_metric_vector(
            {"reference": self._evidence()}, clean_qc(), spec, CollateralKineticsModel()
        )
        self.assertIsNone(metrics.predicted_signal)
        self.assertNotIn("predicted_signal", metrics.objectives())

    def test_without_a_kinetics_model_there_is_no_signal_either(self):
        spec = self._spec(AssayType.IN_VITRO_DETECTION)
        self.assertIsNone(predicted_signal(self._evidence(), spec, None))
        self.assertIsNone(predicted_signal(self._evidence(), spec, NullKineticsModel()))


class TestAComputedFoldDoesNotVouchForTheModel(unittest.TestCase):
    """FR3.3's independence rule, tested at the call rather than the score.

    ``d_pair_evidence`` computes an accessibility for every intended hit and
    then passes it to ``predict()`` only when the source says it is
    independent evidence. A thermodynamic fold is not: it is a prediction
    from the same sequence the activity model already sees, so letting it in
    would have one computation vouch for another. The module says so in a
    comment beside the line that enforces it, and nothing enforced the
    comment.

    Deleting the guard -- ``model_accessibility = accessibility_value``,
    unconditionally -- left the whole suite green while changing every score
    in a run with ``--structure`` and reordering the ranking: rank 1 moved
    from one guide to another on a 400 nt target. ``[structure]`` is the
    install the README leads with, so that is the ordinary path.

    Tested by asking the model what it was handed, rather than by comparing
    scores. A score comparison would pass for the wrong reason the moment
    accessibility stopped ranking, and the claim here is about what reaches
    the prediction, not about what the number comes out as.
    """

    class _RecordingModel:
        """An activity model that remembers the accessibility it was given."""

        version = "recording/v1"

        def __init__(self):
            from cas13guide.services.registry import HeuristicCas13Model

            self._real = HeuristicCas13Model()
            self.domain = self._real.domain
            self.seen = []

        def predict(self, spacer, duplex, accessibility):
            self.seen.append(accessibility)
            return self._real.predict(spacer, duplex, accessibility)

    class _FoldSource:
        """A thermodynamic fold: a real number, and not independent."""

        version = "accessibility/fake-fold/v1"
        is_independent_evidence = False

        def accessibility(self, sequence, start, length):
            return 0.9

    class _ProbingSource(_FoldSource):
        """A probing experiment: the same number, independently obtained."""

        version = "accessibility/fake-probe/v1"
        is_independent_evidence = True

    class _NoExpression:
        def expression_tpm(self, locus_id):
            return None

    def _evidence_with(self, source, target_class=None):
        from cas13guide.contract.enums import Universe
        from cas13guide.pipeline.d_pair_evidence import evidence_for_hit
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.registry import ModelRegistry
        from cas13guide.services.transcriptome import Locus, Transcriptome

        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        # REQUIRED, or the accessibility is never computed for this hit.
        hit = make_hit("g1", "L1", target_id="t1",
                       target_class=target_class or TargetClass.REQUIRED)
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("L1", hit.target_site, Universe.CODING, expression=10.0))
        model = self._RecordingModel()
        evidence = evidence_for_hit(
            hit, spacer, make_spec(), ModelRegistry(model, source),
            CalibrationService(), world, self._NoExpression(),
        )
        return model, evidence

    def test_a_fold_is_computed_and_withheld(self):
        model, evidence = self._evidence_with(self._FoldSource())
        self.assertTrue(model.seen, "the model was never asked")
        self.assertEqual(
            set(model.seen), {None},
            "a non-independent accessibility reached predict(): one "
            "computation is vouching for another",
        )
        self.assertIsNotNone(
            evidence.accessibility,
            "and it should still be recorded -- withheld from the "
            "prediction, not thrown away",
        )

    def test_a_desired_target_is_aimed_at_too(self):
        """The accessibility is computed for sites the design aims at.

        REQUIRED and DESIRED are both aimed at; only the rest are skipped, and
        skipping them is the 8.7x saving the comment above the branch
        describes. Narrowing it to REQUIRED alone leaves a DESIRED target with
        no accessibility at all -- not a slower run, a different one -- and
        the whole suite stayed green.
        """
        from cas13guide.contract.enums import TargetClass

        for target_class in (TargetClass.REQUIRED, TargetClass.DESIRED):
            with self.subTest(target_class=target_class.value):
                _model, evidence = self._evidence_with(
                    self._ProbingSource(), target_class=target_class)
                self.assertIsNotNone(
                    evidence.accessibility,
                    f"a {target_class.value} site is one the design aims at",
                )

    def test_a_probing_experiment_is_passed_through(self):
        """Or the test above would pass with accessibility disabled entirely."""
        model, _evidence = self._evidence_with(self._ProbingSource())
        self.assertTrue(model.seen, "the model was never asked")
        self.assertEqual(
            set(model.seen), {0.9},
            "independent evidence is exactly what predict() is for",
        )


if __name__ == "__main__":
    unittest.main()
