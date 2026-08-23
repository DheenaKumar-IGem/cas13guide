"""Benchmark / regression harness (fix #8) and the wet-lab return path (fix #9).

The provenance ledger records *that* a model version changed. It cannot tell
you whether the change made things worse. This harness re-scores held-out
tiling-screen data on every registry or calibration change and fails loudly
when either predictive skill or conformal coverage regresses.

It also owns :func:`record_wetlab_result`, which routes measured assay
outcomes into the calibration service rather than back into the design spec.
That is the highest-value destination for feedback: real outcomes tighten the
very bounds the whole architecture rests on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..util.duplex import profile_duplex
from .calibration import CalibrationService, CoverageReport
from .registry import ModelRegistry


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One held-out observation from a tiling screen."""

    spacer: str
    target_site: str
    observed_activity: float
    stratum: str
    accessibility: float | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    n: int
    spearman: float
    mae: float
    coverage: tuple[CoverageReport, ...]

    @property
    def coverage_ok(self) -> bool:
        return all(c.ok for c in self.coverage)

    def summary(self) -> str:
        cov = ", ".join(f"{c.stratum}:{c.empirical:.3f}" for c in self.coverage)
        return f"n={self.n} spearman={self.spearman:.3f} mae={self.mae:.3f} coverage[{cov}]"


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else 0.0


def run_benchmark(
    registry: ModelRegistry,
    calibration: CalibrationService,
    cases: list[BenchmarkCase],
    alpha: float = 0.10,
) -> BenchmarkReport:
    predictions: list[float] = []
    observations: list[float] = []
    by_stratum: dict[str, list[tuple[float, float]]] = {}

    for case in cases:
        duplex = profile_duplex(case.spacer, case.target_site)
        pred = registry.activity.predict(case.spacer, duplex, case.accessibility)
        predictions.append(pred)
        observations.append(case.observed_activity)
        by_stratum.setdefault(case.stratum, []).append((pred, case.observed_activity))

    mae = (
        sum(abs(p - o) for p, o in zip(predictions, observations)) / len(cases) if cases else 0.0
    )
    coverage = tuple(
        calibration.validate_coverage(stratum, pairs, alpha)
        for stratum, pairs in sorted(by_stratum.items())
    )
    return BenchmarkReport(
        n=len(cases),
        spearman=spearman(predictions, observations),
        mae=mae,
        coverage=coverage,
    )


class RegressionError(AssertionError):
    """Raised in CI when a registry change degrades skill or coverage."""


def assert_no_regression(
    baseline: BenchmarkReport,
    current: BenchmarkReport,
    max_spearman_drop: float = 0.05,
    max_mae_rise: float = 0.03,
) -> None:
    """CI gate. Any model or calibration bump must clear this."""
    problems: list[str] = []
    if current.spearman < baseline.spearman - max_spearman_drop:
        problems.append(f"spearman {baseline.spearman:.3f} -> {current.spearman:.3f}")
    if current.mae > baseline.mae + max_mae_rise:
        problems.append(f"mae {baseline.mae:.3f} -> {current.mae:.3f}")
    if not current.coverage_ok:
        broken = [c.stratum for c in current.coverage if not c.ok]
        problems.append(f"conformal coverage below nominal in: {broken}")
    if problems:
        raise RegressionError("; ".join(problems))


def record_wetlab_result(
    calibration: CalibrationService,
    stratum: str,
    predicted: float,
    observed: float,
) -> None:
    """Route a measured assay outcome to the calibration service (fix #9).

    Feedback used to loop back into the design spec, which is the lowest-value
    destination -- it changes what you asked for, not how well you can answer.
    Sent here instead, every experiment narrows the intervals that the gate,
    the metrics and the ranking all read.
    """
    calibration.add_observation(stratum, predicted, observed)
