"""Eighth review pass: the acceptance gate, and three stale claims.

This pass turned the instruments on the project's own quality machinery rather
than on the product: the docstrings audited claim by claim, the four service
modules nothing had pointed at, and a sweep for tests that cannot fail.

The find that matters is a gate printing PASS beside a number below the
threshold it prints in the same line -- in the criterion annotated "bounds mean
what they claim".
"""

import unittest

from helpers import *  # noqa: F401,F403


class TestTheCoverageGatePrintsTheThresholdItApplies(unittest.TestCase):
    """``[PASS] conformal coverage 0.889 (need 0.9)``.

    The verdict came from ``coverage_ok``, which applies ``empirical >= nominal
    - 0.05``; the line printed ``nominal``. For any worst-stratum coverage in
    [0.85, 0.90) the report was self-contradictory on its face and the dataset
    was accepted. Every other criterion is an exact ``>=`` against the number it
    displays, so a reader had no reason to suspect a hidden slack on this one --
    and it is the criterion annotated "bounds mean what they claim".

    Reproduced on a shipped-default ``train`` run over a simulated screen.
    """

    def test_the_slack_is_named_rather_than_a_literal(self):
        from cas13guide.services.calibration import COVERAGE_SLACK

        self.assertGreater(COVERAGE_SLACK, 0.0)
        self.assertLess(COVERAGE_SLACK, 0.5)

    def test_the_gate_uses_the_named_slack(self):
        from cas13guide.services.calibration import COVERAGE_SLACK, CoverageReport

        nominal = 0.9
        just_above = CoverageReport(
            stratum="a|b|mm0", n=100, nominal=nominal,
            empirical=nominal - COVERAGE_SLACK + 1e-9,
        )
        just_below = CoverageReport(
            stratum="a|b|mm0", n=100, nominal=nominal,
            empirical=nominal - COVERAGE_SLACK - 1e-9,
        )
        self.assertTrue(just_above.ok)
        self.assertFalse(just_below.ok)

    def test_a_passing_criterion_is_recomputable_from_its_own_line(self):
        """The property that was broken: value and threshold on one line must
        agree with the verdict beside them."""
        from cas13guide.services.acceptance import evaluate_dataset
        from cas13guide.simulate import simulate_screen

        examples, _truth = simulate_screen(n_transcripts=40, seed=12)
        report = evaluate_dataset(examples, seed=0)
        coverage = [c for c in report.criteria if c.name == "conformal coverage"]
        self.assertTrue(coverage, "the criterion must be present to be a test")
        for criterion in coverage:
            with self.subTest(measured=criterion.measured):
                if criterion.note.startswith("regime(s)"):
                    continue          # the uncalibrated branch, reported by cause
                self.assertEqual(
                    criterion.passed, criterion.measured >= criterion.threshold,
                    f"{criterion.measured} vs {criterion.threshold} contradicts "
                    f"passed={criterion.passed}",
                )

    def test_the_note_says_where_the_floor_comes_from(self):
        from cas13guide.services.acceptance import evaluate_dataset
        from cas13guide.simulate import simulate_screen

        examples, _truth = simulate_screen(n_transcripts=40, seed=12)
        report = evaluate_dataset(examples, seed=0)
        for criterion in report.criteria:
            if criterion.name == "conformal coverage" and not criterion.note.startswith(
                "regime(s)"
            ):
                self.assertIn("slack", criterion.note)


class TestVacuousCoverageIsNotPrintedAsAMeasurement(unittest.TestCase):
    """A stratum with no calibration data scores exactly 1.000 by construction.

    ``acceptance.py`` already refuses to print that number and says why in its
    own comment. ``BenchmarkReport.summary()`` printed it anyway -- and
    ``AcceptanceReport.summary()`` embeds that line, so one block carried both
    the fix and the thing it was fixing, two lines apart.
    """

    def _report(self, n_uncalibrated):
        from cas13guide.services.benchmark import BenchmarkReport
        from cas13guide.services.calibration import CoverageReport

        return BenchmarkReport(
            n=10, spearman=0.5, mae=0.1,
            coverage=[
                CoverageReport(stratum="A|B|mm0", n=30, nominal=0.9, empirical=0.94),
                CoverageReport(
                    stratum="A|B|mm3+", n=7, nominal=0.9, empirical=1.0,
                    n_uncalibrated=n_uncalibrated,
                ),
            ],
        )

    def test_an_uncalibrated_stratum_reads_as_vacuous(self):
        summary = self._report(n_uncalibrated=7).summary()
        self.assertIn("vacuous", summary)
        self.assertNotIn("mm3+:1.000", summary)

    def test_a_measured_stratum_still_prints_its_number(self):
        summary = self._report(n_uncalibrated=0).summary()
        self.assertIn("mm0:0.940", summary)
        self.assertIn("mm3+:1.000", summary)
        self.assertNotIn("vacuous", summary)


class TestTheClosedVocabularyDescribesTheCode(unittest.TestCase):
    """``SYNTHESIS_CONSTRAINT`` was documented as "no QC check raises it".

    ``b_qc`` raises it, into the *hard* list, so it fails a guide -- three
    things the comment denied. The comment predates the commit that wired the
    flag up and was never updated, leaving the closed-vocabulary module, which
    is where a maintainer reads what a code means, declaring a live hard-FAIL
    reason dead and free to repurpose.
    """

    def test_the_flag_is_emitted_hard(self):
        from cas13guide.contract.enums import QCFlag
        from cas13guide.models.evidence import Candidate
        from cas13guide.pipeline.b_qc import qc_candidate

        candidate = Candidate(
            guide_id="g1", spacer="GCAUUGCUGNNGGAGCUGAAGUC"[:22],
            source_target_id="t1", start=0,
        )
        result = qc_candidate(candidate, make_spec())
        self.assertIn(QCFlag.SYNTHESIS_CONSTRAINT, result.flags)
        self.assertNotIn(QCFlag.SYNTHESIS_CONSTRAINT, result.warnings)
        self.assertFalse(result.passed)

    def test_no_flag_documented_as_unemitted_is_emitted(self):
        """Pins the class of staleness, not this one instance of it."""
        import re
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "cas13guide" / "contract" / "enums.py"
        ).read_text(encoding="utf-8")
        claimed_dead = set(re.findall(r"no QC check raises it.*?\n\s+([A-Z_]+) =", source, re.S))
        emitted = set(
            re.findall(
                r"(?:flags|warnings)\.append\(QCFlag\.([A-Z_]+)\)",
                (
                    Path(__file__).resolve().parents[1]
                    / "src" / "cas13guide" / "pipeline" / "b_qc.py"
                ).read_text(encoding="utf-8"),
            )
        )
        self.assertEqual(claimed_dead & emitted, set())


class TestTheDefaultTestReadsAnInstance(unittest.TestCase):
    """``assertTrue(MetricVector.predicted_signal_ranks)`` on a slots dataclass.

    The class attribute is a ``member_descriptor``, whose ``bool()`` is
    unconditionally True and has no relationship to the declared default -- so
    the test named for that default passed whichever way the default was set.
    """

    def test_the_class_attribute_is_a_descriptor(self):
        from cas13guide.models.evidence import MetricVector

        self.assertIsInstance(
            MetricVector.predicted_signal_ranks, type(MetricVector.__dict__["metric_version"])
            if "metric_version" in MetricVector.__dict__ else type(MetricVector.predicted_signal_ranks)
        )
        # The point, stated plainly: truthy regardless of the default.
        self.assertTrue(bool(MetricVector.predicted_signal_ranks))

    def test_the_instance_carries_the_real_default(self):
        from cas13guide.models.evidence import MetricVector

        self.assertTrue(MetricVector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0).predicted_signal_ranks)

    def test_the_flag_can_actually_be_turned_off(self):
        """Which is what makes an instance-level assertion meaningful."""
        from cas13guide.models.evidence import MetricVector

        vector = MetricVector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, predicted_signal_ranks=False)
        self.assertFalse(vector.predicted_signal_ranks)


if __name__ == "__main__":
    unittest.main()


class TestStatedFiguresMatchWhatTheCodeDoes(unittest.TestCase):
    """Two figures in comments that the repository itself contradicted.

    Found by the ninth pass, which audited the domain constants and the cited
    claims against each other rather than against outside literature -- the
    only kind of domain check this repository can actually settle.
    """

    def test_the_hard_pfs_cost_is_a_quarter_not_a_third(self):
        """Excluding one base of four deletes a quarter of the windows.

        ``effectors.py`` said "a third" in its module docstring while the
        entry a hundred lines below and the test that pins it both said a
        quarter. Measured on uniform random RNA it is 0.2502.
        """
        import random
        from pathlib import Path

        from cas13guide.contract.effectors import EFFECTORS
        from cas13guide.contract.enums import Effector

        preferred = EFFECTORS[Effector.LWA_CAS13A].pfs.three_prime_preferred
        self.assertEqual(len(preferred), 3, "the claim assumes one excluded base")

        rng = random.Random(4)
        sequence = "".join(rng.choice("ACGU") for _ in range(60_000))
        spacer_length = 28
        flanks = [
            sequence[start + spacer_length]
            for start in range(len(sequence) - spacer_length)
        ]
        excluded = sum(1 for base in flanks if base not in preferred) / len(flanks)
        self.assertAlmostEqual(excluded, 0.25, places=1)

        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "cas13guide" / "contract" / "effectors.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("deleted a third of every candidate set", source)

    def test_the_preset_length_claim_counts_correctly(self):
        """"every miRNA preset resolves to 22 or 23 nt" is four of the five."""
        from pathlib import Path

        from cas13guide.contract.enums import Effector
        from cas13guide.contract.presets import (
            MIRNA_PRESETS,
            largest_designable_spacer_length,
        )

        resolved = {
            preset_id: largest_designable_spacer_length(preset, Effector.LWA_CAS13A)
            for preset_id, preset in MIRNA_PRESETS.items()
        }
        matching = [pid for pid, length in resolved.items() if length in (22, 23)]
        self.assertEqual(len(matching), 4)
        self.assertEqual(resolved["hsa-miR-21-5p"], 24)

        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "cas13guide" / "services" / "registry.py"
        ).read_text(encoding="utf-8")
        self.assertIn("four of the five miRNA presets", source)
