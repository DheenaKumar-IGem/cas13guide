"""Conformal calibration: coverage must hold, and ignorance must stay wide."""

import random
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.services.calibration import CalibrationService


def synthetic(n, noise, rng):
    out = []
    for _ in range(n):
        truth = rng.random()
        pred = min(1.0, max(0.0, truth + rng.gauss(0, noise)))
        out.append((pred, truth))
    return out


class TestConformalCoverage(unittest.TestCase):
    def test_empirical_coverage_meets_nominal(self):
        rng = random.Random(42)
        cs = CalibrationService()
        key = CalibrationService.stratum_key("RfxCas13d", "cell_culture_knockdown", "mm0")
        data = synthetic(600, 0.12, rng)
        cs.add_many(key, data[:300])
        report = cs.validate_coverage(key, data[300:], alpha=0.10)
        self.assertTrue(report.ok, report)
        self.assertGreaterEqual(report.empirical, 0.85)

    def test_wider_alpha_gives_narrower_interval(self):
        rng = random.Random(1)
        cs = CalibrationService()
        key = CalibrationService.stratum_key("RfxCas13d", "cell_culture_knockdown", "mm0")
        cs.add_many(key, synthetic(300, 0.15, rng))
        tight = cs.interval(0.5, key, alpha=0.30)
        loose = cs.interval(0.5, key, alpha=0.01)
        self.assertLess(tight.width, loose.width)

    def test_noisier_model_gives_wider_interval(self):
        rng = random.Random(5)
        cs = CalibrationService()
        quiet = CalibrationService.stratum_key("RfxCas13d", "a", "mm0")
        noisy = CalibrationService.stratum_key("RfxCas13d", "b", "mm0")
        cs.add_many(quiet, synthetic(400, 0.05, rng))
        cs.add_many(noisy, synthetic(400, 0.25, rng))
        self.assertLess(
            cs.interval(0.5, quiet, 0.1).width, cs.interval(0.5, noisy, 0.1).width
        )


class TestStratification(unittest.TestCase):
    """Strata are error regimes, not intent labels."""

    def test_mismatch_regimes_bucket_by_difficulty(self):
        self.assertEqual(CalibrationService.mismatch_regime(0), "mm0")
        self.assertEqual(CalibrationService.mismatch_regime(1), "mm1-2")
        self.assertEqual(CalibrationService.mismatch_regime(2), "mm1-2")
        self.assertEqual(CalibrationService.mismatch_regime(3), "mm3+")
        self.assertEqual(CalibrationService.mismatch_regime(9), "mm3+")

    def test_a_hard_duplex_and_an_easy_one_get_different_bounds(self):
        """Which is the point: one pooled stratum would average them together."""
        rng = random.Random(31)
        cs = CalibrationService()
        easy = CalibrationService.stratum_key("RfxCas13d", "a", "mm0")
        hard = CalibrationService.stratum_key("RfxCas13d", "a", "mm3+")
        cs.add_many(easy, synthetic(300, 0.05, rng))
        cs.add_many(hard, synthetic(300, 0.30, rng))
        self.assertLess(cs.interval(0.5, easy, 0.1).width, cs.interval(0.5, hard, 0.1).width)

    def test_intent_does_not_enter_the_key(self):
        key = CalibrationService.stratum_key("RfxCas13d", "cell_culture_knockdown", "mm0")
        for intent in ("required", "forbidden", "tolerated"):
            self.assertNotIn(intent, key)


class TestUncertifiableLevels(unittest.TestCase):
    def test_alpha_below_one_over_n_plus_one_is_flagged(self):
        """Coverage 1-alpha is unreachable when alpha < 1/(n+1).

        The interval is still returned -- it is the best available -- but it
        must be flagged, or the caller believes a guarantee that the stratum
        cannot supply.
        """
        rng = random.Random(11)
        cs = CalibrationService(min_stratum_n=20)
        key = CalibrationService.stratum_key("RfxCas13d", "cell_culture_knockdown", "mm0")
        cs.add_many(key, synthetic(20, 0.12, rng))  # alpha must be >= 1/21 = 0.0476
        iv = cs.interval(0.5, key, alpha=0.02)
        self.assertTrue(iv.ood)
        self.assertIn("uncertified", iv.stratum)

    def test_achievable_level_on_the_same_stratum_is_not_flagged(self):
        rng = random.Random(11)
        cs = CalibrationService(min_stratum_n=20)
        key = CalibrationService.stratum_key("RfxCas13d", "cell_culture_knockdown", "mm0")
        cs.add_many(key, synthetic(20, 0.12, rng))
        iv = cs.interval(0.5, key, alpha=0.10)
        self.assertFalse(iv.ood)
        self.assertNotIn("uncertified", iv.stratum)

    def test_finite_sample_correction_is_present(self):
        """Dropping the (n+1) term silently loses coverage at small n."""
        rng = random.Random(2)
        cs = CalibrationService(min_stratum_n=5)
        key = CalibrationService.stratum_key("E", "A", "mm0")
        data = synthetic(30, 0.15, rng)
        cs.add_many(key, data)
        stratum = cs._resolve(key)
        q, certified = stratum.quantile(0.10)
        self.assertTrue(certified)
        scores = sorted(stratum.residuals())
        # ceil(31 * 0.9) = 28 -> the 28th smallest, strictly above the naive
        # ceil(30 * 0.9) = 27th.
        self.assertEqual(q, scores[27])


class TestOutOfDomain(unittest.TestCase):
    def test_no_stratum_yields_maximal_ignorance_flagged_ood(self):
        cs = CalibrationService()
        iv = cs.interval(0.9, "Unknown|assay|required", alpha=0.10)
        self.assertTrue(iv.ood)
        self.assertEqual((iv.lower, iv.upper), (0.0, 1.0))

    def test_a_low_prediction_is_never_evidence_of_safety(self):
        """The design's core asymmetry, enforced at the interval level."""
        cs = CalibrationService()
        iv = cs.interval(0.01, "Unknown|assay|mm3+", alpha=0.10)
        self.assertEqual(iv.upper, 1.0)  # UCB stays maximal despite a low point

    def test_fallback_stratum_is_used_but_flagged(self):
        rng = random.Random(9)
        cs = CalibrationService()
        cs.add_many("RfxCas13d|*|*", synthetic(200, 0.1, rng))
        iv = cs.interval(0.5, "RfxCas13d|novel_assay|mm0", alpha=0.10)
        self.assertTrue(iv.ood)  # not the exact stratum
        self.assertLess(iv.width, 1.0)  # but still informative
        self.assertEqual(iv.stratum, "RfxCas13d|*|*")

    def test_thin_stratum_is_not_trusted(self):
        rng = random.Random(3)
        cs = CalibrationService(min_stratum_n=20)
        key = CalibrationService.stratum_key("RfxCas13d", "assay", "mm0")
        cs.add_many(key, synthetic(5, 0.1, rng))
        iv = cs.interval(0.5, key, alpha=0.10)
        self.assertTrue(iv.ood)
        self.assertEqual((iv.lower, iv.upper), (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
