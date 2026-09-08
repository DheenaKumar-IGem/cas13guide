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
    Universe,
    Verdict,
)
from ..contract.spec import ResolvedDesignSpec
from ..metrics.bounds import benefit_lower
from ..models.evidence import GateResult, GuideAssessment, MetricVector, PairEvidence, QCResult
from ..services.indexes import ExhaustivenessBound


def gate(
    metrics: MetricVector,
    qc: QCResult,
    evidence: tuple[PairEvidence, ...],
    bound: ExhaustivenessBound,
    universe_gaps: frozenset[Universe],
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
        # Attribute the cause. Coverage is read at a lower bound, so *missing
        # evidence* and *demonstrated absence of activity* both drive it to
        # zero and produce the same verdict -- but they are completely
        # different findings, and reporting an unvalidated design as
        # "required target missed" tells the user their guide is bad when in
        # fact nothing has been measured. The verdict stays FAIL either way:
        # an uncertifiable design must not certify.
        intended = [e for e in evidence if e.is_intended]
        if intended and all(not e.domain_ok for e in intended):
            fail.append(ReasonCode.MODEL_OUT_OF_DOMAIN)
            detail.append(
                "every intended-target pair is outside the activity model's "
                "declared domain, so the bound is maximally wide and coverage "
                "cannot be demonstrated -- this is missing validation, not a "
                "demonstrated miss"
            )
        elif intended and all(e.activity.ood for e in intended):
            fail.append(ReasonCode.CALIBRATION_UNSUPPORTED)
            detail.append(
                "every intended-target pair fell back to an out-of-domain "
                "calibration stratum, so the bound is too wide to clear the "
                "floor -- more calibration data in this regime, not a "
                "different guide, is what closes this"
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

    # Universe adequacy is a gate condition, not documentation (fix #3) -- but
    # it covers two findings that deserve different verdicts. A universe the
    # run *declared* and did not search is a defect in the run: the index was
    # narrower than the corpus it was built from, and the off-target table is
    # clean for a reason that has nothing to do with this guide. A universe the
    # assay is exposed to that the corpus simply holds no sequence in is a
    # statement about the evidence supplied -- the same statement for every
    # candidate here -- and no verdict on this guide can lift it.
    # What separates the two is whether the sequences were there to search. An
    # index that cannot see its corpus reports None and does not get to have
    # the run accused of skipping anything.
    # Two ways a gap is the *run's* fault rather than the corpus's. Either the
    # run declared a universe and searched less than it declared -- provable
    # from the declaration alone, and the hole fix #3 was written for -- or the
    # corpus held sequences in a universe the assay needs and the search did
    # not look at them. An index that cannot see its corpus reports None and
    # does not get to have the run accused of the second.
    available = bound.corpus_universes
    if available is not None:
        # The precise test, and it subsumes the declaration: a universe the
        # corpus holds and the search did not cover was skipped, whatever was
        # declared. Declaring more than the corpus holds is not a skip -- there
        # was nothing there to skip -- it is an overclaim, and it is the
        # unrepresented arm below that says so.
        skipped = frozenset(universe_gaps & available)
    else:
        # No corpus to consult, so the declaration is the only evidence: a
        # universe the run said it would search and did not. Still only ever a
        # classification of the gaps the caller reported -- a caller that says
        # there is no gap is not overruled by arithmetic on the declaration.
        skipped = frozenset(
            universe_gaps & (spec.context.declared_universes - bound.universes)
        )
    if skipped:
        fail.append(ReasonCode.UNIVERSE_INADEQUATE)
        detail.append(
            "available and not searched: "
            + ",".join(sorted(u.value for u in skipped))
        )

    unrepresented = frozenset(universe_gaps - skipped)

    if (
        metrics.discrimination_margin is not None
        and metrics.discrimination_margin < t.discrimination_margin_min
    ):
        fail.append(ReasonCode.DISCRIMINATION_MARGIN_INSUFFICIENT)
        detail.append(
            f"discrimination margin {metrics.discrimination_margin:.2f} "
            f"< tau {t.discrimination_margin_min:.2f}"
        )

    # Both hazard budgets are reported when both are blown. They share one
    # reason code -- the verdict is the same either way -- but the *detail* is
    # what a reviewer acts on, and an ``elif`` here hid the burden overrun
    # behind the worst-single-hit overrun, which is the more obvious of the two
    # and the less informative about why the guide is unusable.
    hazard_detail: list[str] = []
    if metrics.worst_off_target > t.worst_off_target_max:
        hazard_detail.append(
            f"worst_off_target {metrics.worst_off_target:.3f} > {t.worst_off_target_max:.3f}"
        )
    if metrics.cumulative_burden > t.cumulative_burden_max:
        hazard_detail.append(
            f"cumulative_burden {metrics.cumulative_burden:.3f} > {t.cumulative_burden_max:.3f}"
        )
    if hazard_detail:
        fail.append(ReasonCode.HAZARD_BUDGET_EXCEEDED)
        detail.extend(hazard_detail)

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

    # Distinct loci, not pair rows. PairEvidence is guide x locus x context, so
    # summing rows counted one locus once per context and once per binding site
    # within it -- and the string says "locus/loci", unlike the two R3 details
    # above it, which say "pair(s)" and do mean rows.
    # Off-target loci only. An intended locus's abundance reaches no metric,
    # no threshold and no verdict: `PairEvidence.expression` has exactly two
    # readers, `pair_hazard` and the off-target explorer rows, and both are
    # behind `not e.is_intended`. Counting the target itself downgraded every
    # guide for a gap that changes no number the run produces.
    #
    # It made the certified surface unreachable from the documented command.
    # `design --fasta t1.fa` gave PASS 0 / REVIEW 272, the sole reason on every
    # survivor being this one, so there was no panel, no controls and no FR6.3
    # interval -- a 100% review rate, which is the degeneracy the R1/R2/R3
    # grading in this module's docstring exists to prevent. The same run with
    # `--expression 120` gave PASS 272 under identical synthetic calibration.
    # So did `--expression 0`, which asserts the target is not expressed at
    # all: any value cleared the flag, because nothing downstream read it.
    #
    # This also makes the reason mean what the report already says it means --
    # `report_html.py`'s glossary glosses R2 EXPRESSION_UNKNOWN as "an
    # off-target whose expression is unknown", which was not the case that
    # fired.
    unknown_loci = {
        e.locus_id for e in evidence if e.expression_unknown and not e.is_intended
    }
    if unknown_loci:
        review.append(
            (ReviewTier.R2, ReasonCode.EXPRESSION_UNKNOWN,
             f"{len(unknown_loci)} locus/loci with no expression evidence")
        )

    # Robustness is the minimum required coverage across every evaluated
    # context; with only the reference it *is* required_coverage, so this
    # cannot fire unless the user supplied --variants. Against the same floor
    # that FAILs a guide on the reference, because a guide is not more
    # acceptable for missing its target in an isoform than in the reference --
    # it is only less certain which one the user cares about, and that is a
    # judgement for a human, which is what REVIEW means.
    if unrepresented:
        review.append(
            (
                ReviewTier.R1,
                ReasonCode.UNIVERSE_UNREPRESENTED,
                "no sequence supplied in "
                + ",".join(sorted(u.value for u in unrepresented))
                + ", which this assay exposes the guide to",
            )
        )

    if metrics.robustness < t.required_coverage_min:
        review.append(
            (
                ReviewTier.R1,
                ReasonCode.VARIANT_COVERAGE_LOST,
                f"robustness {metrics.robustness:.3f} < "
                f"{t.required_coverage_min:.3f} in some supplied context",
            )
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

    # FR3.4 -- a site buried in secondary structure. R1, not R2: this is a
    # borderline *number* from a model we have, not a gap in the evidence. Off
    # unless the spec sets a floor, because what counts as too structured is an
    # assay decision.
    if (
        t.min_accessibility > 0.0
        and metrics.accessibility is not None
        and metrics.accessibility < t.min_accessibility
    ):
        review.append(
            (
                ReviewTier.R1,
                ReasonCode.TARGET_INACCESSIBLE,
                f"intended site mean unpaired probability {metrics.accessibility:.3f} "
                f"< {t.min_accessibility:.3f}; the target region is highly structured",
            )
        )

    # FR5.2 -- duplex free energy outside the optimal window. R1: a borderline
    # number, not an evidence gap. Silent unless the spec chose a window, since
    # the optimum is assay-specific and this package does not assert one.
    if metrics.duplex_dg_per_nt is not None and (
        t.duplex_dg_per_nt_min is not None or t.duplex_dg_per_nt_max is not None
    ):
        low, high = t.duplex_dg_per_nt_min, t.duplex_dg_per_nt_max
        value = metrics.duplex_dg_per_nt
        if (low is not None and value < low) or (high is not None and value > high):
            direction = (
                "over-stable: tolerates mismatches it should reject, costing "
                "discrimination"
                if low is not None and value < low
                else "too weak: the guide may not stay bound, costing signal"
            )
            review.append(
                (
                    ReviewTier.R1,
                    ReasonCode.DUPLEX_ENERGY_OUT_OF_RANGE,
                    f"duplex {value:.2f} kcal/mol/nt outside "
                    f"[{low}, {high}] -- {direction}",
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


def _dominates_objectives(oa: dict[str, float], ob: dict[str, float]) -> bool:
    """``dominates`` on already-materialized objective dicts."""
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


def dominates(a: MetricVector, b: MetricVector) -> bool:
    """True when ``a`` is at least as good everywhere and better somewhere."""
    return _dominates_objectives(a.objectives(), b.objectives())


def pareto_front(assessments: list[GuideAssessment]) -> list[GuideAssessment]:
    """Non-dominated set. Computed *before* any scalar ranking.

    Still the O(n^2) pairwise comparison -- the frontier has to be exact and n
    is bounded by ``budget.max_candidates`` -- but each guide's objective dict
    is built once instead of once per comparison. At 1500 passing guides that
    is 2n dict builds rather than 2n^2, which took the pass from ~1.4 s to
    ~0.2 s; the returned set is unchanged.
    """
    objectives = [a.metrics.objectives() for a in assessments]
    front: list[GuideAssessment] = []
    for i, cand in enumerate(assessments):
        oc = objectives[i]
        if any(
            _dominates_objectives(oo, oc)
            for j, oo in enumerate(objectives)
            if j != i
        ):
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
    # The soft PFS bias. Weighted below coverage and hazard on purpose: it is a
    # real efficacy signal, but it is a base-composition prior, and it must not
    # be able to outrank evidence about whether the guide hits its target.
    "pfs_preference": 1.0,
    # Structural accessibility. Only present when an accessibility model was
    # configured, so this weight is inert on a run without one.
    "accessibility": 1.5,
    # Predicted diagnostic signal. Present only for assays where collateral
    # cleavage is the readout, and weighted high there because it is the thing
    # the assay actually measures.
    "predicted_signal": 2.5,
    # Only present in the objective set for allele/SNV tasks, so this weight is
    # inert for every other task. Omitting it did not make the margin a pure
    # constraint -- the gate already enforces tau -- it made the one metric the
    # task exists for contribute *nothing* to the order among guides that
    # cleared tau, while still appearing in the rationale line as "x w0".
    "discrimination_margin": 2.0,
}


@dataclass(frozen=True, slots=True)
class RankedGuide:
    assessment: GuideAssessment
    rank: int
    score: float
    rationale: tuple[str, ...]
    #: What each objective contributed to ``score``: ``weight x normalised
    #: value``, so the values sum to ``score`` exactly. That is what answers
    #: "is this guide here because of the evidence or because of how I weighted
    #: it" -- an axis contributing most of the total is a ranking resting on a
    #: preference.
    #:
    #: Described until now as ``d(score)/d(weight)``, which is a different
    #: number: the derivative in the weight is the *normalised value* alone,
    #: i.e. this divided by the weight. Anyone wanting the derivative can take
    #: it; the comment simply did not describe what the field holds.
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


def distinct_site_ids(
    ranked: list[RankedGuide], min_spacing: int
) -> frozenset[str]:
    """Guide ids that are the best-ranked representative of a distinct site.

    A ranked list is not a panel. Activity varies smoothly along a transcript,
    so the top entries are usually the same window shifted by a base or two:
    they share almost all of their target, the same local structure and the same
    accessibility, and they succeed or fail together. Measured on a 400 nt random
    target with a 28 nt spacer, the top ten ranked guides covered **two**
    distinct sites -- ranks 1, 2 and 3 differed by one nucleotide each. Ordering
    that list buys one experiment ten times while looking, on screen, like ten
    independent candidates, and it makes "test several candidates" impossible to
    follow from the list the tool hands you.

    This marks rather than filters. Nothing is dropped and no rank changes:
    ranks stay numbered against the full set, so a caller that shows only
    distinct sites still displays #1, #4, #9 and a reader can see that guides
    were collapsed rather than never generated.

    Sites are compared on the coordinates a guide actually **binds**, per
    target, and two guides are the same site if they overlap on *any* target
    they share. Keying on the candidate's ``source_target_id`` instead -- the
    record it happened to be tiled from -- is not the same question, and gets
    the wrong answer as soon as a run holds more than one target record: a
    spacer tiled from the second record carries that record's coordinate, so it
    was never compared against anything from the first, and two guides whose
    sites on the REQUIRED target sit one nucleotide apart were both listed as
    distinct. That is the failure this function exists to prevent, reappearing
    through the door the multi-record path opened.

    Comparing binding sites is also what ``optimizer._profile`` does when it
    enforces panel spacing, so the ranked list and the panel selector now
    answer the same question rather than two similar ones.
    """
    seen: dict[str, list[int]] = {}
    distinct: set[str] = set()
    for entry in ranked:
        # One rule, on the assessment, read by this and by the panel
        # selector. It used to be written out here and again in
        # ``optimizer._profile``, with the two versions disagreeing.
        sites = entry.assessment.binding_sites()

        overlaps = any(
            abs(start - other) < min_spacing
            for target_id, start in sites.items()
            for other in seen.get(target_id, ())
        )
        if not overlaps:
            for target_id, start in sites.items():
                seen.setdefault(target_id, []).append(start)
            distinct.add(entry.assessment.guide_id)
    return frozenset(distinct)


def review_rate(assessments: list[GuideAssessment]) -> float:
    """Fraction of assessed guides sitting in REVIEW.

    Exposed so the gate's usefulness is measurable: a REVIEW rate near 1.0
    means the gate has degraded into noise users will click through.
    """
    if not assessments:
        return 0.0
    n = sum(1 for a in assessments if a.gate.verdict is Verdict.REVIEW)
    return n / len(assessments)
