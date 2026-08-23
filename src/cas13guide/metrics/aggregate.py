"""The six named metrics (design revision v2, section 03).

Aggregation rules, and why each is shaped the way it is:

* **Required coverage** -- ``min`` over required targets of the *best* site's
  ``LCB(activity)``. The worst intended target sets the score, so an average
  can never hide a miss.
* **Desired utility** -- weighted benefit over DESIRED targets only. Tolerated
  targets earn no reward (they merely avoid punishment).
* **Worst off-target** -- ``max`` of ``UCB(act) x UCB(expr) x hazard``.
* **Cumulative burden** -- the same product summed, so many weak hits cannot
  vanish under a single-max view.
* **Robustness** -- worst-case required coverage across evaluated contexts.
* **Discrimination margin** -- ``min LCB(on)/UCB(off)`` for allele tasks, the
  ratio constraint that two independent thresholds cannot express (fix #2).

Everything here is a pure function of evidence; nothing decides anything. The
gate does that, in :mod:`cas13guide.pipeline.f_decide`.
"""

from __future__ import annotations

import math
from collections import defaultdict

from ..contract.enums import TargetClass
from ..contract.spec import ResolvedDesignSpec
from ..models.evidence import MetricVector, PairEvidence, QCResult
from .bounds import benefit_lower, hazard_product

#: Returned when a required target has no supporting evidence at all.
NO_EVIDENCE = 0.0


def _best_lcb_per_target(
    evidence: tuple[PairEvidence, ...], target_class: TargetClass
) -> dict[str, float]:
    """Best (highest LCB) site per named target of the given class.

    ``benefit_lower`` rather than ``.lower`` for the same reason ``_hazard``
    calls ``hazard_product``: which end of the interval a *benefit* reads is a
    property of the bound algebra, and it should have exactly one definition.
    """
    best: dict[str, float] = defaultdict(lambda: NO_EVIDENCE)
    for e in evidence:
        if e.target_class is target_class and e.hit.target_id is not None:
            best[e.hit.target_id] = max(
                best[e.hit.target_id], benefit_lower(e.activity)
            )
    return dict(best)


def required_coverage(evidence: tuple[PairEvidence, ...], spec: ResolvedDesignSpec) -> float:
    required = spec.contract.by_class(TargetClass.REQUIRED)
    if not required:
        return 1.0
    best = _best_lcb_per_target(evidence, TargetClass.REQUIRED)
    return min(best.get(t.target_id, NO_EVIDENCE) for t in required)


def desired_utility(evidence: tuple[PairEvidence, ...], spec: ResolvedDesignSpec) -> float:
    desired = spec.contract.by_class(TargetClass.DESIRED)
    if not desired:
        return 0.0
    best = _best_lcb_per_target(evidence, TargetClass.DESIRED)
    return sum(t.weight * best.get(t.target_id, NO_EVIDENCE) for t in desired)


def pair_hazard(e: PairEvidence) -> float:
    """``UCB(activity) x UCB(expression) x hazard_weight`` for one pair.

    Delegated to :func:`~cas13guide.metrics.bounds.hazard_product` rather than
    multiplying the three factors here. ``metrics/bounds.py`` documents itself
    as the one place the LCB/UCB asymmetry lives, and it was not: this function
    re-implemented the product by hand, so an edit there -- to the non-negative
    check, or to which end of an interval a hazard reads -- would silently not
    reach the number the gate actually tests.
    """
    return hazard_product(e.activity, e.expression, scale=e.hazard_weight)


#: Kept as the old private name so nothing that imported it breaks.
_hazard = pair_hazard


def off_target_hazards(evidence: tuple[PairEvidence, ...]) -> list[tuple[PairEvidence, float]]:
    return [(e, pair_hazard(e)) for e in evidence if not e.is_intended]


def worst_off_target(evidence: tuple[PairEvidence, ...]) -> float:
    hazards = off_target_hazards(evidence)
    return max((h for _, h in hazards), default=0.0)


def cumulative_burden(evidence: tuple[PairEvidence, ...]) -> float:
    return sum(h for _, h in off_target_hazards(evidence))


def discrimination_margin(
    evidence: tuple[PairEvidence, ...], spec: ResolvedDesignSpec
) -> float | None:
    """Ratio of intended-allele benefit to off-allele hazard (fix #2).

    For an allele task the FORBIDDEN target differs from the REQUIRED one by a
    single nucleotide, so thresholding ``LCB(on)`` and ``UCB(off)``
    independently makes every guide fail. What matters is their *ratio*.

    Returns ``None`` when the task does not call for discrimination, ``inf``
    when the off-allele has no credible activity at all.

    The ``on`` term is the min over the contract's REQUIRED targets, exactly as
    :func:`required_coverage` computes it -- *not* over the targets that happen
    to appear in ``evidence``. Taking the min over observed targets meant a
    required allele with no evidence at all was dropped from the minimum rather
    than scoring ``NO_EVIDENCE``, so a context missing a required allele
    reported a *higher* margin than one covering it (6.00 vs 5.67 in the case
    that surfaced this), and ``build_metric_vector``'s cross-context ``min``
    then hid the gap entirely. A missing required target is the worst case for
    this ratio, not an absent one.
    """
    if not spec.requires_discrimination:
        return None

    required = spec.contract.by_class(TargetClass.REQUIRED)
    if not required:
        return 0.0

    on_best = _best_lcb_per_target(evidence, TargetClass.REQUIRED)
    on = min(on_best.get(t.target_id, NO_EVIDENCE) for t in required)

    off_upper = [
        e.activity.upper
        for e in evidence
        if e.target_class is TargetClass.FORBIDDEN and e.hit.target_id is not None
    ]
    if not off_upper:
        return math.inf
    worst_off = max(off_upper)
    if worst_off <= 0.0:
        return math.inf
    return on / worst_off


def manufacturability(qc: QCResult) -> float:
    """Synthesis/cloning friendliness in ``[0, 1]``; QC failures already gate."""
    score = qc.manufacturability
    return min(max(score, 0.0), 1.0)


def build_metric_vector(
    evidence_by_context: dict[str, tuple[PairEvidence, ...]],
    qc: QCResult,
    spec: ResolvedDesignSpec,
) -> MetricVector:
    """Aggregate every context into one metric vector.

    The reference context supplies the headline numbers; robustness is the
    worst required coverage seen across *all* evaluated contexts (variants,
    haplotypes, isoform sets), which is what a guide actually has to survive.
    """
    if not evidence_by_context:
        raise ValueError("no contexts evaluated")

    reference = evidence_by_context.get("reference")
    if reference is None:
        reference = next(iter(evidence_by_context.values()))

    per_context_coverage = {
        name: required_coverage(ev, spec) for name, ev in evidence_by_context.items()
    }
    # Hazard is read at its worst across contexts too -- never averaged.
    worst = max(worst_off_target(ev) for ev in evidence_by_context.values())
    burden = max(cumulative_burden(ev) for ev in evidence_by_context.values())
    margins = [
        m
        for m in (discrimination_margin(ev, spec) for ev in evidence_by_context.values())
        if m is not None
    ]

    return MetricVector(
        required_coverage=required_coverage(reference, spec),
        desired_utility=desired_utility(reference, spec),
        worst_off_target=worst,
        cumulative_burden=burden,
        robustness=min(per_context_coverage.values()),
        manufacturability=manufacturability(qc),
        discrimination_margin=min(margins) if margins else None,
        pfs_preference=qc.pfs_preference,
    )
