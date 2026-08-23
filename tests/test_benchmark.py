"""Regression harness and the wet-lab -> calibration return path."""

import random
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.services.benchmark import (
    BenchmarkCase,
    RegressionError,
    assert_no_regression,
    record_wetlab_result,
    run_benchmark,
    spearman,
)
from cas13guide.services.calibration import CalibrationService
from cas13guide.services.registry import (
    HeuristicCas13Model,
    ModelRegistry,
    NullAccessibilityModel,
)
from cas13guide.util.duplex import profile_duplex
from cas13guide.util.seq import revcomp

SPACER = "GCAUUGCUGUGGAGCUGAAGUC"
STRATUM = "RfxCas13d|cell_culture_knockdown|mm1-2"


MODEL = HeuristicCas13Model()
NOISE = 0.12


def cases(n=60, seed=4, noise=NOISE):
    """Held-out cases drawn from the same process the calibration saw.

    ``observed`` is the model's prediction plus screen noise -- i.e. an
    in-distribution held-out set. Generating labels independently of the model
    instead simulates distribution shift, which the coverage gate is supposed
    to catch (see ``test_regression_gate_catches_distribution_shift``).
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        site = list(revcomp(SPACER))
        for _ in range(rng.randint(0, 4)):
            i = rng.randrange(len(site))
            site[i] = rng.choice([b for b in "AUGC" if b != site[i]])
        site_str = "".join(site)
        pred = MODEL.predict(SPACER, profile_duplex(SPACER, site_str), None)
        observed = min(1.0, max(0.0, pred + rng.gauss(0, noise)))
        out.append(
            BenchmarkCase(
                spacer=SPACER,
                target_site=site_str,
                observed_activity=observed,
                stratum=STRATUM,
            )
        )
    return out


def shifted_cases(n=60, seed=4):
    """Labels unrelated to the model: the distribution-shift scenario."""
    rng = random.Random(seed)
    return [
        BenchmarkCase(
            spacer=c.spacer,
            target_site=c.target_site,
            observed_activity=rng.random(),
            stratum=STRATUM,
        )
        for c in cases(n, seed)
    ]


class TestSpearman(unittest.TestCase):
    def test_monotone_relationships(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_handles_ties_without_blowing_up(self):
        self.assertIsInstance(spearman([1, 1, 1, 1], [1, 2, 3, 4]), float)


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.registry = ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel())
        self.calibration = CalibrationService()
        rng = random.Random(8)
        for case in cases(n=200, seed=99):
            pred = MODEL.predict(
                case.spacer, profile_duplex(case.spacer, case.target_site), None
            )
            self.calibration.add_observation(STRATUM, pred, case.observed_activity)

    def test_report_is_populated(self):
        report = run_benchmark(self.registry, self.calibration, cases())
        self.assertEqual(report.n, 60)
        self.assertTrue(report.coverage)
        self.assertIn("spearman", report.summary())

    def test_regression_gate_passes_when_unchanged(self):
        report = run_benchmark(self.registry, self.calibration, cases())
        assert_no_regression(report, report)  # must not raise

    def test_regression_gate_catches_distribution_shift(self):
        """Coverage, not just skill, is a CI gate.

        When held-out labels stop matching the process the calibration was
        built on, the conformal intervals silently stop covering. That is
        exactly the failure the harness exists to make loud.
        """
        report = run_benchmark(self.registry, self.calibration, shifted_cases())
        self.assertFalse(report.coverage_ok)
        with self.assertRaises(RegressionError):
            assert_no_regression(report, report)

    def test_regression_gate_catches_a_skill_drop(self):
        baseline = run_benchmark(self.registry, self.calibration, cases())
        degraded = type(baseline)(
            n=baseline.n,
            spearman=baseline.spearman - 0.4,
            mae=baseline.mae,
            coverage=baseline.coverage,
        )
        with self.assertRaises(RegressionError):
            assert_no_regression(baseline, degraded)

    def test_regression_gate_catches_broken_coverage(self):
        baseline = run_benchmark(self.registry, self.calibration, cases())
        broken_cov = tuple(
            type(c)(stratum=c.stratum, n=c.n, nominal=0.90, empirical=0.40)
            for c in baseline.coverage
        )
        degraded = type(baseline)(
            n=baseline.n, spearman=baseline.spearman, mae=baseline.mae, coverage=broken_cov
        )
        with self.assertRaises(RegressionError):
            assert_no_regression(baseline, degraded)


class TestWetLabFeedback(unittest.TestCase):
    def test_feedback_moves_a_stratum_from_ood_to_in_domain(self):
        """Fix #9: assay results tighten the bounds the whole system reads."""
        calibration = CalibrationService(min_stratum_n=20)
        before = calibration.interval(0.5, STRATUM, alpha=0.10)
        self.assertTrue(before.ood)
        self.assertEqual((before.lower, before.upper), (0.0, 1.0))

        rng = random.Random(21)
        for _ in range(40):
            truth = rng.random()
            record_wetlab_result(
                calibration, STRATUM, min(1.0, max(0.0, truth + rng.gauss(0, 0.08))), truth
            )

        after = calibration.interval(0.5, STRATUM, alpha=0.10)
        self.assertFalse(after.ood)
        self.assertLess(after.width, before.width)


if __name__ == "__main__":
    unittest.main()
