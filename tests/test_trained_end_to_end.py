"""A fitted model enters through the same seam -- and fixes the ranking."""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass, Task, Universe
from cas13guide.contract.spec import (
    ContextScope,
    EffectorAssayProfile,
    ResolvedDesignSpec,
    TargetContract,
    TargetSpec,
)
from cas13guide.contract.enums import AssayType, Effector
from cas13guide.pipeline.orchestrator import DesignContext, run_design
from cas13guide.services.calibration import CalibrationService
from cas13guide.services.registry import (
    ActivityModel,
    HeuristicCas13Model,
    ModelRegistry,
    NullAccessibilityModel,
)
from cas13guide.services.training import train_and_calibrate
from cas13guide.services.transcriptome import Locus, Transcriptome
from cas13guide.simulate import simulate_screen

UNIVERSES = frozenset(
    {Universe.CODING, Universe.NCRNA, Universe.PRE_MRNA, Universe.RRNA, Universe.MITO}
)


def screen(n_transcripts=40, transcript_length=220, step=4):
    # 40 transcripts -> >= 8 calibration groups, which is what the calibration
    # service requires before it will present a bound as certified.
    return simulate_screen(
        seed=21, step=step, signal_scale=10, transcript_effect_sd=0.8,
        noise_sd=0.2, n_transcripts=n_transcripts, transcript_length=transcript_length,
    )[0]


def world_and_spec(examples):
    sequence = "".join(e.target_site for e in examples[:6])
    world = Transcriptome("GRCh38", "v44")
    world.add(Locus("TGT1", sequence, Universe.CODING, gene="TGT", expression=120.0))
    world.add(Locus("RRNA1", "AUGC" * 100, Universe.RRNA, gene="RN18S", expression=9000.0))
    spec = ResolvedDesignSpec(
        contract=TargetContract(
            task=Task.KNOCKDOWN,
            targets=(
                TargetSpec(
                    target_id="t_tgt", sequence=sequence,
                    target_class=TargetClass.REQUIRED, expression=120.0,
                ),
            ),
        ),
        profile=EffectorAssayProfile(
            effector=Effector.RFX_CAS13D, assay=AssayType.CELL_CULTURE_KNOCKDOWN
        ),
        context=ContextScope(
            organism="Homo sapiens", genome_build="GRCh38",
            annotation_version="v44", declared_universes=UNIVERSES,
        ),
    )
    return world, spec, {"TGT1": "t_tgt"}


class TestTrainedModelSatisfiesTheSeam(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = screen()
        cls.result = train_and_calibrate(cls.examples, alpha=1.0, seed=5)

    def test_trained_model_satisfies_the_activity_model_protocol(self):
        model = self.result.model
        self.assertIsInstance(model, ActivityModel)
        self.assertTrue(model.version)
        self.assertTrue(model.domain.effectors)

    def test_it_drops_into_a_registry_unchanged(self):
        registry = ModelRegistry(self.result.model, NullAccessibilityModel())
        self.assertIn("activity", registry.versions())
        self.assertEqual(registry.versions()["activity"], self.result.model.version)


class TestRankingBecomesInformative(unittest.TestCase):
    """The README's 'known limitation', pinned as a regression test.

    The heuristic saturates, so every perfect-match guide gets the same lower
    bound and the ranking carries no information. A fitted model discriminates
    between guides -- and nothing downstream changes to accommodate it.
    """

    @classmethod
    def setUpClass(cls):
        cls.examples = screen()
        cls.result = train_and_calibrate(cls.examples, alpha=1.0, seed=5)
        cls.world, cls.spec, cls.mapping = world_and_spec(cls.examples)

    def _run(self, model, calibration):
        contexts = [
            DesignContext.auto("reference", self.world, UNIVERSES, self.mapping)
        ]
        return run_design(
            self.spec, contexts, ModelRegistry(model, NullAccessibilityModel()),
            calibration, panel_size=3,
        )

    def _heuristic_calibration(self):
        model = HeuristicCas13Model()
        cs = CalibrationService()
        for e in self.examples:
            cs.add_observation(
                e.stratum(), model.predict(e.spacer, e.duplex(), e.accessibility),
                e.observed_activity,
            )
        return model, cs

    def test_heuristic_barely_discriminates_between_guides(self):
        """The saturating model compresses every guide into a narrow band."""
        model, calibration = self._heuristic_calibration()
        summary = self._run(model, calibration).metric_summary
        self.assertGreater(summary.n_assessed, 50)
        self.assertLess(summary.coverage_spread, 0.25)
        # Measured against the trained model below, which is far finer.
        self.assertLess(summary.distinct_coverage_values, summary.n_assessed / 4)

    def test_trained_model_discriminates_far_more_finely(self):
        """The claim, measured across *every* assessed guide.

        Comparing only ranked entries would be misleading: the ranked list
        holds gate survivors, so a model that fails everything shows zero
        distinct scores for reasons that have nothing to do with whether it
        discriminated.
        """
        trained = self._run(self.result.model, self.result.calibration).metric_summary
        model, calibration = self._heuristic_calibration()
        heuristic = self._run(model, calibration).metric_summary

        self.assertTrue(trained.discriminated)
        self.assertGreater(
            trained.distinct_coverage_values, 4 * heuristic.distinct_coverage_values
        )
        self.assertGreater(trained.coverage_spread, 2 * heuristic.coverage_spread)
        self.assertGreater(
            trained.distinct_coverage_values, 0.8 * trained.n_assessed
        )

    def test_trained_model_produces_certifiable_guides(self):
        report = self._run(self.result.model, self.result.calibration)
        self.assertGreater(report.verdict_counts["PASS"], 0)
        self.assertTrue(report.ranked)

    def test_ranking_is_ordered_by_score(self):
        report = self._run(self.result.model, self.result.calibration)
        scores = [r.score for r in report.ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual([r.rank for r in report.ranked], list(range(1, len(scores) + 1)))

    def test_gate_still_governs_the_trained_model(self):
        """A better model must not be able to bypass the safety gate."""
        report = self._run(self.result.model, self.result.calibration)
        for entry in report.to_dict()["ranked"]:
            self.assertNotEqual(entry["verdict"], "FAIL")


class TestThinCalibrationRefusesToCertify(unittest.TestCase):
    """Too little calibration data must widen bounds, not quietly narrow them.

    With a small screen the conformal interval for perfect duplexes is ~0.50
    wide, so the lower bound falls under the coverage floor and the gate FAILs
    everything. That is the architecture behaving correctly: it declines to
    certify guides whose evidence base cannot support the claim, rather than
    reporting a confident number it has not earned.
    """

    @classmethod
    def setUpClass(cls):
        cls.examples = screen(n_transcripts=12, transcript_length=160, step=6)
        cls.result = train_and_calibrate(cls.examples, alpha=1.0, seed=5)
        cls.world, cls.spec, cls.mapping = world_and_spec(cls.examples)

    def test_thin_calibration_produces_wide_intervals(self):
        key = "RfxCas13d|cell_culture_knockdown|mm0"
        interval = self.result.calibration.interval(0.9, key, alpha=0.10)
        self.assertGreater(interval.width, 0.4)

    def test_and_therefore_nothing_is_certified(self):
        contexts = [DesignContext.auto("reference", self.world, UNIVERSES, self.mapping)]
        report = run_design(
            self.spec, contexts,
            ModelRegistry(self.result.model, NullAccessibilityModel()),
            self.result.calibration, panel_size=3,
        )
        self.assertEqual(report.verdict_counts["PASS"], 0)
        self.assertTrue(report.shortfalls)

    def test_more_calibration_data_narrows_the_interval(self):
        """The same model, better calibrated, earns tighter bounds."""
        rich = train_and_calibrate(screen(), alpha=1.0, seed=5)
        key = "RfxCas13d|cell_culture_knockdown|mm0"
        thin_width = self.result.calibration.interval(0.9, key, alpha=0.10).width
        rich_width = rich.calibration.interval(0.9, key, alpha=0.10).width
        self.assertLess(rich_width, thin_width)


if __name__ == "__main__":
    unittest.main()
