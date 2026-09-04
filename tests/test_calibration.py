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


def synthetic_groups(n, group_count=12):
    """Group labels for `synthetic` rows, standing in for distinct transcripts.

    Calibration data with no stated provenance can no longer certify a bound,
    so a fixture that means "many independent units" has to say which.
    """
    return [f"T{i % group_count:02d}" for i in range(n)]


def add_synthetic(service, key, n, noise, rng, group_count=12):
    """`add_many` with per-row group labels."""
    for (pred, truth), group in zip(
        synthetic(n, noise, rng), synthetic_groups(n, group_count)
    ):
        service.add_observation(key, pred, truth, group=group)


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
        # grouped: this test is about achievable alpha, not about provenance
        add_synthetic(cs, key, 20, 0.12, rng)  # alpha must be >= 1/21 = 0.0476
        iv = cs.interval(0.5, key, alpha=0.02)
        self.assertTrue(iv.ood)
        self.assertIn("uncertified", iv.stratum)

    def test_achievable_level_on_the_same_stratum_is_not_flagged(self):
        rng = random.Random(11)
        cs = CalibrationService(min_stratum_n=20)
        key = CalibrationService.stratum_key("RfxCas13d", "cell_culture_knockdown", "mm0")
        add_synthetic(cs, key, 20, 0.12, rng)
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
        add_synthetic(cs, "RfxCas13d|*|*", 200, 0.1, rng)
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



class TestMissingCalibrationCannotScoreAsCoverage(unittest.TestCase):
    """No calibration data must fail the CI gate, not sail through it.

    ``interval`` falls back to ``Interval.unknown(0.0, 1.0)`` when nothing in
    the chain resolves, and every observed activity lies in ``[0, 1]``, so that
    fallback covers *by construction*. ``validate_coverage`` therefore reported
    a perfect 1.000 for a stratum with no data at all and ``ok`` said True --
    deleting calibration data made the benchmark greener than having it, which
    is the one direction a coverage check must never move.
    """

    KEY = "RfxCas13d|cell_culture_knockdown|mm0"

    def test_an_empty_service_reports_vacuous_coverage_and_fails(self):
        rng = random.Random(3)
        service = CalibrationService()
        report = service.validate_coverage(self.KEY, synthetic(50, 0.1, rng), 0.10)
        self.assertEqual(report.empirical, 1.0)   # it does "cover" everything
        self.assertTrue(report.uncalibrated)
        self.assertEqual(report.n_uncalibrated, report.n)
        self.assertFalse(report.ok, "vacuous coverage must not pass the gate")

    def test_the_same_pairs_pass_once_the_stratum_is_calibrated(self):
        rng = random.Random(4)
        service = CalibrationService()
        service.add_many(self.KEY, synthetic(200, 0.1, rng))
        report = service.validate_coverage(self.KEY, synthetic(100, 0.1, rng), 0.10)
        self.assertFalse(report.uncalibrated)
        self.assertEqual(report.n_uncalibrated, 0)
        self.assertTrue(report.ok, report)

    def test_a_fallback_stratum_is_still_a_real_measurement(self):
        """Resolving to a broader stratum is ood, but it is not vacuous."""
        rng = random.Random(5)
        service = CalibrationService()
        service.add_many("RfxCas13d|cell_culture_knockdown|*", synthetic(200, 0.1, rng))
        report = service.validate_coverage(self.KEY, synthetic(80, 0.1, rng), 0.10)
        self.assertFalse(report.uncalibrated)
        self.assertTrue(service.interval(0.5, self.KEY, 0.10).ood)


class TestStratumKeysAreValidatedWhereTheyAreBuilt(unittest.TestCase):
    """A malformed key used to be stored happily and explode on every read."""

    def test_a_label_containing_the_separator_is_rejected_at_construction(self):
        with self.assertRaises(ValueError) as caught:
            CalibrationService.stratum_key("RfxCas13d", "HEK293|rep1", "mm0")
        self.assertIn("|", str(caught.exception))

    def test_a_well_formed_key_still_round_trips(self):
        key = CalibrationService.stratum_key("RfxCas13d", "cell_culture_knockdown", "mm0")
        self.assertEqual(key, "RfxCas13d|cell_culture_knockdown|mm0")
        self.assertEqual(len(CalibrationService._fallback_chain(key)), 4)

    def test_reading_a_malformed_key_names_the_key_and_the_format(self):
        service = CalibrationService()
        service.add_observation("screen2024", 0.5, 0.5)
        with self.assertRaises(ValueError) as caught:
            service.interval(0.5, "screen2024", 0.10)
        message = str(caught.exception)
        self.assertIn("screen2024", message)
        self.assertIn("effector|assay|regime", message)


class TestResidualCacheIsTransparent(unittest.TestCase):
    """The sorted-residual cache must never go stale after an add."""

    def test_adding_an_observation_changes_the_quantile(self):
        rng = random.Random(9)
        service = CalibrationService()
        key = "RfxCas13d|cell_culture_knockdown|mm0"
        service.add_many(key, synthetic(60, 0.05, rng))
        before = service.interval(0.5, key, 0.10).width
        # A wildly wrong pair must widen the interval, not be cached away.
        for _ in range(40):
            service.add_observation(key, 0.0, 1.0)
        after = service.interval(0.5, key, 0.10).width
        self.assertGreater(after, before)

    def test_cached_quantile_equals_a_freshly_sorted_one(self):
        import math

        rng = random.Random(11)
        service = CalibrationService()
        key = "RfxCas13d|cell_culture_knockdown|mm0"
        pairs = synthetic(137, 0.2, rng)
        service.add_many(key, pairs)
        stratum = service._strata[key]
        stratum.quantile(0.10)                      # populate the cache
        stratum.quantile(0.10)                      # read it back
        scores = sorted(abs(o - p) for p, o in pairs)
        rank = math.ceil((len(pairs) + 1) * 0.90)
        self.assertEqual(stratum.quantile(0.10), (scores[rank - 1], True))
        self.assertEqual(stratum.sorted_residuals(), scores)



class TestTheStratumSizeMinimumIsTheDefaultOne(unittest.TestCase):
    """`MIN_STRATUM_N` is 20, and nothing exercised the default.

    Two tests construct a service with an explicit `min_stratum_n` -- 10 in
    one, 20 in the other -- so the *parameter* was covered and the module
    constant behind it was not. Every production path builds
    `CalibrationService()` with no arguments (cli, demo, training, workflow),
    so the default is the only value a real run ever uses, and lowering it to
    15 left the whole suite green.

    Stated in rows rather than against the constant: an assertion written in
    terms of `MIN_STRATUM_N` holds whatever `MIN_STRATUM_N` becomes.
    """

    KEY = "RfxCas13d|cell_culture_knockdown|mm0"

    def _service_with(self, rows: int) -> CalibrationService:
        service = CalibrationService()
        for i in range(rows):
            service.add_observation(self.KEY, 0.5, 0.5, group=f"T{i % 12:02d}")
        return service

    def test_nineteen_rows_do_not_certify(self):
        self.assertIsNone(self._service_with(19).resolve(self.KEY))

    def test_twenty_do(self):
        stratum = self._service_with(20).resolve(self.KEY)
        self.assertIsNotNone(stratum)
        self.assertEqual(stratum.n, 20)


if __name__ == "__main__":
    unittest.main()
