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

from .training import (
    TrainingExample,
    TrainingResult,
    compare_splitting,
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
    min_groups: int = 6,
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

    report.criteria.append(
        Criterion("rows", len(examples) >= min_rows, float(len(examples)), float(min_rows))
    )
    report.criteria.append(
        Criterion(
            "transcript groups",
            len(groups) >= min_groups,
            float(len(groups)),
            float(min_groups),
            note="grouped splitting needs several transcripts",
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

    worst_coverage = min((c.empirical for c in result.benchmark.coverage), default=0.0)
    nominal = 1.0 - calibration_alpha
    report.criteria.append(
        Criterion(
            "conformal coverage",
            result.benchmark.coverage_ok,
            worst_coverage,
            nominal,
            note="bounds mean what they claim",
        )
    )

    if probe_split_bias:
        # Ten extra model fits (five grouped folds, five ungrouped). Worth it
        # for a real dataset, skippable when the caller only wants the gate.
        comparison = compare_splitting(examples, alpha=alpha, seed=seed)
        report.split_bias = comparison.interpretation
    return report
