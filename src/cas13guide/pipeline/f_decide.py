"""Stage F -- the safety gate, then Pareto, then preferences. In that order.

The ordering *is* the design. A guide that fails the gate cannot be rescued by
any weighting a user supplies, and the Pareto frontier is computed before any
scalar collapse so trade-offs survive to the point where a human sees them.

REVIEW is graded (fix #12). With honest OOD flagging, incomplete search and
unknown expression, an ungraded REVIEW bucket swallows most real designs and
users learn to click through it. R1/R2/R3 separate "borderline number" from
"evidence gap" from "nobody validated this", and :func:`review_rate` makes the
bucket's size a number the system can be tuned against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..contract.enums import (
    SEVERITY,
    ReasonCode,
    ReviewTier,
    TargetClass,
    Verdict,
)
from ..contract.spec import ResolvedDesignSpec
from ..models.evidence import GateResult, GuideAssessment, MetricVector, PairEvidence, QCResult
from ..services.indexes import ExhaustivenessBound


def gate(
    metrics: MetricVector,
    qc: QCResult,
    evidence: tuple[PairEvidence, ...],
    bound: ExhaustivenessBound,
    universe_gaps: frozenset,
    spec: ResolvedDesignSpec,
) -> GateResult:
    """PASS / REVIEW / FAIL with machine-readable reasons."""
    t = spec.thresholds
    fail: list[ReasonCode] = []
    detail: list[str] = []

    # ---------------- FAIL: hard stops ----------------
    if not qc.passed:
        fail.append(ReasonCode.DETERMINISTIC_INCOMPATIBLE)
        detail.append("QC flags: " + ",".join(f.value for f in qc.flags))

    if metrics.required_coverage < t.required_coverage_min:
        fail.append(ReasonCode.REQUIRED_TARGET_MISSED)
        detail.append(
            f"required_coverage {metrics.required_coverage:.3f} < {t.required_coverage_min:.3f}"
        )

    forbidden = [
        e
        for e in evidence
        if e.target_class is TargetClass.FORBIDDEN
        and e.activity.upper > t.forbidden_activity_max
    ]
    if forbidden:
        fail.append(ReasonCode.CREDIBLE_FORBIDDEN_HIT)
        worst = max(e.activity.upper for e in forbidden)
        detail.append(f"forbidden hit UCB(activity) {worst:.3f} > {t.forbidden_activity_max:.3f}")

    # Universe adequacy is a gate condition, not documentation (fix #3).
    if universe_gaps:
        fail.append(ReasonCode.UNIVERSE_INADEQUATE)
        detail.append(
            "unsearched universes required by assay: "
            + ",".join(sorted(u.value for u in universe_gaps))
        )

    if (
        metrics.discrimination_margin is not None
        and metrics.discrimination_margin < t.discrimination_margin_min
    ):
        fail.append(ReasonCode.DISCRIMINATION_MARGIN_INSUFFICIENT)
        detail.append(
            f"discrimination margin {metrics.discrimination_margin:.2f} "
            f"< tau {t.discrimination_margin_min:.2f}"
        )

    if metrics.worst_off_target > t.worst_off_target_max:
        fail.append(ReasonCode.HAZARD_BUDGET_EXCEEDED)
        detail.append(
            f"worst_off_target {metrics.worst_off_target:.3f} > {t.worst_off_target_max:.3f}"
        )
    elif metrics.cumulative_burden > t.cumulative_burden_max:
        fail.append(ReasonCode.HAZARD_BUDGET_EXCEEDED)
        detail.append(
            f"cumulative_burden {metrics.cumulative_burden:.3f} > {t.cumulative_burden_max:.3f}"
        )

    if fail:
        return GateResult(verdict=Verdict.FAIL, reasons=tuple(fail), detail=tuple(detail))

    # ---------------- REVIEW: graded ----------------
    review: list[tuple[ReviewTier, ReasonCode, str]] = []

    model_gaps = sum(1 for e in evidence if not e.domain_ok)
    calibration_gaps = sum(1 for e in evidence if e.domain_ok and e.activity.ood)
    if model_gaps:
        review.append(
            (
                ReviewTier.R3,
                ReasonCode.MODEL_OUT_OF_DOMAIN,
                f"{model_gaps} pair(s) outside the activity model's declared domain",
            )
        )
    if calibration_gaps:
        review.append(
            (
                ReviewTier.R3,
                ReasonCode.CALIBRATION_UNSUPPORTED,
                f"{calibration_gaps} pair(s) with no in-domain calibration stratum",
            )
        )

    if not bound.covers(t.required_search_mismatches):
        review.append(
            (
                ReviewTier.R2,
                ReasonCode.SEARCH_NOT_EXHAUSTIVE,
                f"search bound: {bound.describe()}"
                + (f" ({bound.note})" if bound.note else ""),
            )
        )

    if any(e.expression_unknown for e in evidence):
        n = sum(1 for e in evidence if e.expression_unknown)
        review.append(
            (ReviewTier.R2, ReasonCode.EXPRESSION_UNKNOWN, f"{n} locus/loci with no expression evidence")
        )

    if metrics.required_coverage < t.required_coverage_min + t.borderline_band:
        review.append(
            (
                ReviewTier.R1,
                ReasonCode.BORDERLINE_COVERAGE,
                f"required_coverage {metrics.required_coverage:.3f} within borderline band",
            )
        )

    if metrics.worst_off_target > t.worst_off_target_max * (1 - t.borderline_band):
        review.append(
            (
                ReviewTier.R1,
                ReasonCode.BORDERLINE_OFF_TARGET,
                f"worst_off_target {metrics.worst_off_target:.3f} within borderline band",
            )
        )

    if review:
        tier = max((r[0] for r in review), key=lambda x: SEVERITY[x.value])
        return GateResult(
            verdict=Verdict.REVIEW,
            reasons=tuple(r[1] for r in review),
            review_tier=tier,
            detail=tuple(r[2] for r in review),
        )

    return GateResult(
        verdict=Verdict.PASS,
        reasons=(ReasonCode.CONSTRAINTS_SATISFIED,),
        detail=("all constraints satisfied with adequate evidence",),
    )


# ---------------------------------------------------------------------------
# Pareto, then and only then preferences
# ---------------------------------------------------------------------------


def dominates(a: MetricVector, b: MetricVector) -> bool:
    """True when ``a`` is at least as good everywhere and better somewhere."""
    oa, ob = a.objectives(), b.objectives()
    keys = oa.keys() & ob.keys()
    if not keys:
        return False
    better_somewhere = False
    for k in keys:
        if oa[k] < ob[k]:
            return False
        if oa[k] > ob[k]:
            better_somewhere = True
    return better_somewhere


def pareto_front(assessments: list[GuideAssessment]) -> list[GuideAssessment]:
    """Non-dominated set. Computed *before* any scalar ranking."""
    front: list[GuideAssessment] = []
    for cand in assessments:
        if any(dominates(other.metrics, cand.metrics) for other in assessments if other is not cand):
            continue
        front.append(cand)
    return front


DEFAULT_WEIGHTS: dict[str, float] = {
    "required_coverage": 3.0,
    "desired_utility": 1.0,
    "worst_off_target": 2.0,
    "cumulative_burden": 1.0,
    "robustness": 1.5,
    "manufacturability": 0.5,
}


@dataclass(frozen=True, slots=True)
class RankedGuide:
    assessment: GuideAssessment
    rank: int
    score: float
    rationale: tuple[str, ...]
    #: d(score)/d(weight) per objective -- how much the rank depends on the
    #: user's weighting rather than on the evidence.
    sensitivity: dict[str, float]


def _normalize(value: float, lo: float, hi: float) -> float:
    """Min-max normalize to ``[0, 1]``, treating infinities as the extremes."""
    if math.isinf(value):
        return 1.0 if value > 0 else 0.0
    if hi == lo:
        return 0.5
    return (value - lo) / (hi - lo)


def rank_by_preference(
    assessments: list[GuideAssessment], weights: dict[str, float] | None = None
) -> list[RankedGuide]:
    """Scalarize -- but only over guides that already cleared the gate.

    Contributions are min-max normalized across the surviving set so one
    metric's units cannot dominate, and each guide carries the per-objective
    contribution back out as its sensitivity report.
    """
    # Defence in depth: the gate runs first and a failed guide must never be
    # rankable, whatever the caller passes in.
    assessments = [a for a in assessments if a.gate.verdict is not Verdict.FAIL]
    if not assessments:
        return []
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    objective_sets = [a.metrics.objectives() for a in assessments]
    keys = sorted(set().union(*(set(o) for o in objective_sets)))
    # Ranges are computed over *finite* values only. discrimination_margin is
    # legitimately +inf for a guide with no credible off-allele hit -- i.e. the
    # ideal guide -- and folding inf into a min-max range yields NaN for that
    # guide and 0.0 for everyone else, which makes the sort order depend on
    # input order and ranks the best guide last.
    ranges: dict[str, tuple[float, float]] = {}
    for k in keys:
        finite = [o[k] for o in objective_sets if k in o and math.isfinite(o[k])]
        ranges[k] = (min(finite), max(finite)) if finite else (0.0, 0.0)

    scored: list[tuple[float, GuideAssessment, dict[str, float], list[str]]] = []
    for assessment, objectives in zip(assessments, objective_sets):
        contributions: dict[str, float] = {}
        rationale: list[str] = []
        total = 0.0
        for k in keys:
            if k not in objectives:
                continue
            lo, hi = ranges[k]
            norm = _normalize(objectives[k], lo, hi)
            w = weights.get(k, 0.0)
            contributions[k] = w * norm
            total += w * norm
            rationale.append(f"{k}={objectives[k]:+.3f} (norm {norm:.2f} x w{w:g})")
        scored.append((total, assessment, contributions, rationale))

    scored.sort(key=lambda row: row[0], reverse=True)
    return [
        RankedGuide(
            assessment=assessment,
            rank=i + 1,
            score=score,
            rationale=tuple(rationale),
            sensitivity=contributions,
        )
        for i, (score, assessment, contributions, rationale) in enumerate(scored)
    ]


def review_rate(assessments: list[GuideAssessment]) -> float:
    """Fraction of assessed guides sitting in REVIEW.

    Exposed so the gate's usefulness is measurable: a REVIEW rate near 1.0
    means the gate has degraded into noise users will click through.
    """
    if not assessments:
        return 0.0
    n = sum(1 for a in assessments if a.gate.verdict is Verdict.REVIEW)
    return n / len(assessments)
