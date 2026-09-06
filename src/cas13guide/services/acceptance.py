"""Decide whether a screen is good enough to build a model on.

The package's models have only ever been validated on simulated data. This is
the gate a *real* screen has to clear before its model is allowed anywhere near
a design run, so that "we trained on real data" is a checked claim rather than
a hopeful one.

Each criterion is reported separately with its measured value, because the
useful output when a dataset fails is *which* property failed, not a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .calibration import COVERAGE_SLACK
from .training import (
    TrainingExample,
    TrainingResult,
    compare_splitting,
    groups_needed_for_calibration,
    train_and_calibrate,
)


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str
    passed: bool
    measured: float
    threshold: float
    note: str = ""

    def line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"  [{mark}] {self.name:<28} {self.measured:>8.3f} (need {self.threshold:g}) {self.note}"


@dataclass(slots=True)
class AcceptanceReport:
    criteria: list[Criterion] = field(default_factory=list)
    split_bias: str = ""
    result: TrainingResult | None = None

    @property
    def accepted(self) -> bool:
        return bool(self.criteria) and all(c.passed for c in self.criteria)

    def summary(self) -> str:
        verdict = "ACCEPTED" if self.accepted else "REJECTED"
        lines = [f"dataset {verdict}"]
        lines.extend(c.line() for c in self.criteria)
        if self.split_bias:
            lines.append(f"  split-bias probe: {self.split_bias}")
        if self.result is not None:
            lines.append(f"  {self.result.summary()}")
        return "\n".join(lines)


def evaluate_dataset(
    examples: list[TrainingExample],
    alpha: float = 1.0,
    seed: int = 0,
    min_rows: int = 300,
    min_groups: int | None = None,
    min_calibration_groups: int = 8,
    min_cv_spearman: float = 0.20,
    min_holdout_spearman: float = 0.15,
    calibration_alpha: float = 0.10,
    probe_split_bias: bool = True,
) -> AcceptanceReport:
    """Run the full protocol on a screen and report each acceptance criterion.

    The skill thresholds are deliberately low. The question this gate answers
    is "is there learnable signal here and are the bounds honest", not "is this
    model good" -- a model can be weak and still be *safe*, because the
    calibration widens to match and the gate refuses to certify what the
    evidence cannot support.
    """
    report = AcceptanceReport()
    groups = {e.group for e in examples}
    if min_groups is None:
        # Not 6. Calibration groups are a subset of all groups and the split
        # takes a fifth of them, so a screen needs 38 transcripts before it can
        # hold `min_calibration_groups` of them -- measured at 720 rows: 6
        # groups gives 1 calibration transcript, 12 gives 2, 24 gives 5, 48
        # gives 10. The old default admitted screens that the criterion five
        # rows below could not accept, and told their author the number was 6.
        #
        # Derived from the splitter's own fraction so the two cannot drift.
        min_groups = groups_needed_for_calibration(min_calibration_groups)

    report.criteria.append(
        Criterion("rows", len(examples) >= min_rows, float(len(examples)), float(min_rows))
    )
    report.criteria.append(
        Criterion(
            "transcript groups",
            len(groups) >= min_groups,
            float(len(groups)),
            float(min_groups),
            note=("conformal calibration is measured on a held-out fifth of "
                  f"the transcripts, and needs {min_calibration_groups} of "
                  "them"),
        )
    )
    if not all(c.passed for c in report.criteria):
        return report

    result = train_and_calibrate(
        examples, alpha=alpha, seed=seed, calibration_alpha=calibration_alpha
    )
    report.result = result

    report.criteria.append(
        Criterion(
            "grouped CV spearman",
            result.cv.spearman_mean >= min_cv_spearman,
            result.cv.spearman_mean,
            min_cv_spearman,
            note="learnable signal across transcripts",
        )
    )
    report.criteria.append(
        Criterion(
            "held-out spearman",
            result.benchmark.spearman >= min_holdout_spearman,
            result.benchmark.spearman,
            min_holdout_spearman,
            note="generalizes to unseen transcripts",
        )
    )

    calibration_groups = len(result.split_groups["calibrate"])
    report.criteria.append(
        Criterion(
            "calibration transcripts",
            calibration_groups >= min_calibration_groups,
            float(calibration_groups),
            float(min_calibration_groups),
            note="too few makes conformal coverage unreliable, not just loose",
        )
    )

    nominal = 1.0 - calibration_alpha
    uncalibrated = [c for c in result.benchmark.coverage if c.uncalibrated]
    if uncalibrated:
        # Report the *cause*, not the number. An uncalibrated stratum scores a
        # perfect 1.000 by construction -- every observation falls inside the
        # [0, 1] fallback interval -- so printing `min(empirical)` next to a
        # FAIL mark showed a passing-looking figure with no hint of why the
        # criterion failed.
        names = ", ".join(sorted(c.stratum.rsplit("|", 1)[-1] for c in uncalibrated))
        report.criteria.append(
            Criterion(
                "conformal coverage",
                False,
                0.0,
                nominal,
                note=(
                    f"regime(s) {names} had no calibration stratum at all, so their "
                    "coverage is vacuous rather than measured"
                ),
            )
        )
    else:
        worst_coverage = min((c.empirical for c in result.benchmark.coverage), default=0.0)
        # The floor the gate applies, not the nominal level it aims at. Every
        # other criterion here is an exact `>=` against the number it prints,
        # so a reader had no reason to suspect a hidden 0.05 slack on this one
        # -- and the line rendered "[PASS] conformal coverage 0.889 (need 0.9)",
        # self-contradictory on its face, in the criterion annotated "bounds
        # mean what they claim". Reproduced on a shipped-default `train` run.
        floor = nominal - COVERAGE_SLACK
        report.criteria.append(
            Criterion(
                "conformal coverage",
                result.benchmark.coverage_ok,
                worst_coverage,
                floor,
                note=(
                    f"bounds mean what they claim; floor is nominal {nominal:.2f} "
                    f"less {COVERAGE_SLACK:.2f} for finite-sample slack under "
                    "grouped splitting"
                ),
            )
        )

    if probe_split_bias:
        # Ten extra model fits (five grouped folds, five ungrouped). Worth it
        # for a real dataset, skippable when the caller only wants the gate.
        comparison = compare_splitting(examples, alpha=alpha, seed=seed)
        report.split_bias = comparison.interpretation
    return report
