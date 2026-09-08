"""Guide-set optimizer -- named, not just described (fix #11).

Panel design maximizes

    f(S) = sum_t  w_t * (1 - prod_{g in S} (1 - p_gt))

the expected number of required targets covered by at least one guide, where
``p_gt`` is guide ``g``'s calibrated *lower* bound on target ``t``. That
function is monotone and submodular, so greedy selection carries a real
approximation guarantee:

* under a pure cardinality constraint (panel size ``k``): ``1 - 1/e``. This one
  is real -- Nemhauser, Wolsey and Fisher, for monotone submodular maximization
  under a cardinality constraint;
* with spacing or diversity switched on: **nothing**. This used to claim
  ``1/2`` on the grounds that the feasible family was "matroid-like". It is
  not. "No two members within ``min_spacing``, and no two sharing an off-target
  locus" is the independent-set family of a conflict graph, which has no
  exchange property: one pan-variant guide in a conserved window is feasible,
  the three variant-specific guides packed around it are feasible together, and
  the first cannot be extended into the second. Greedy takes the pan guide for
  the larger first-round gain and forfeits all three.

  Measured on exactly that instance with the orchestrator's own defaults,
  greedy reached **0.404** of the brute-forced optimum against a claimed floor
  of 0.5, and 0.06 with a weaker hub and a larger panel. A conflict graph is a
  ``p``-independence system, where greedy gives ``1/(p + 1)`` -- and ``p`` here
  is however many guides one guide can block, which is unbounded. So there is
  no constant to report, and the code now says so instead of inventing one.

Constraints applied during selection:

* **spacing** -- two guides tiling the same target must not sit within
  ``min_spacing`` nt of each other, so a panel cannot be three shifted copies
  of one site;
* **diversity** -- a new guide must not share more than ``max_shared_off``
  off-target loci with any already-selected guide, so two members cannot be
  brought down by the same off-target locus.

What this does **not** buy is independence. The diversity constraint is about
shared *off-targets*; it says nothing about whether two guides succeed or fail
together on the *intended* target, which is what ``coverage`` below multiplies
as though it did. Members still share a transcript, a single activity model
whose errors are correlated, and -- above ``min_spacing`` -- often overlapping
sites. ``Panel.coverage`` is a noisy-OR of per-guide lower bounds and therefore
reads as more confident than the evidence supports; it is a ranking aid, not a
probability that the panel works. See ``Panel.coverage`` for the same caveat at
the point of use.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contract.enums import TargetClass
from ..contract.spec import ResolvedDesignSpec
from ..metrics.bounds import benefit_lower
from ..models.evidence import GuideAssessment


@dataclass(frozen=True, slots=True)
class Panel:
    """A selected panel, plus what its numbers do and do not mean.

    ``coverage`` is a noisy-OR over per-guide conformal lower bounds: for each
    required target, ``1 - prod(1 - lcb)``. That formula assumes the members
    fail independently, and they do not. They share a transcript, they share one
    activity model whose errors are correlated, and the spacing constraint only
    stops them sharing a *site*. So read it as a ranking aid -- higher is a
    better panel than lower -- and not as the probability that the panel
    detects the target, which would be an overstatement in the confident
    direction.
    """

    guides: tuple[GuideAssessment, ...]
    coverage: dict[str, float]
    objective: float
    guarantee: str
    dropped: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.guides)


@dataclass(slots=True)
class _Profile:
    """Precomputed per-guide facts the selector needs."""

    coverage: dict[str, float] = field(default_factory=dict)
    off_loci: frozenset[str] = frozenset()
    sites: dict[str, int] = field(default_factory=dict)


def _profile(assessment: GuideAssessment) -> _Profile:
    """Coverage is a REQUIRED-only question; spacing is not.

    ``coverage`` feeds the objective, which is the expected number of
    *required* targets covered, so it reads REQUIRED evidence only. ``sites``
    feeds the spacing constraint, which asks whether two guides land on top of
    each other -- a question that does not care why a target was wanted. The
    two used to be built in one pass under one condition, which quietly gave
    the spacing constraint the coverage rule: DESIRED overlaps and any overlap
    at a zero lower bound were invisible to it. See
    :meth:`GuideAssessment.binding_sites`, which both this and
    ``f_decide.distinct_site_ids`` now read.
    """
    coverage: dict[str, float] = {}
    off: set[str] = set()
    for e in assessment.evidence:
        if e.target_class is TargetClass.REQUIRED and e.hit.target_id:
            tid = e.hit.target_id
            if benefit_lower(e.activity) > coverage.get(tid, 0.0):
                coverage[tid] = benefit_lower(e.activity)
        elif not e.is_intended:
            off.add(e.locus_id)
    return _Profile(coverage=coverage, off_loci=frozenset(off),
                    sites=assessment.binding_sites())


def _objective(profiles: list[_Profile], weights: dict[str, float]) -> float:
    total = 0.0
    for target_id, w in weights.items():
        miss = 1.0
        for p in profiles:
            miss *= 1.0 - p.coverage.get(target_id, 0.0)
        total += w * (1.0 - miss)
    return total


def _compatible(
    candidate: _Profile,
    chosen: list[_Profile],
    min_spacing: int,
    max_shared_off: int,
) -> bool:
    for other in chosen:
        for target_id, start in candidate.sites.items():
            if target_id in other.sites and abs(start - other.sites[target_id]) < min_spacing:
                return False
        if len(candidate.off_loci & other.off_loci) > max_shared_off:
            return False
    return True


def select_panel(
    assessments: list[GuideAssessment],
    spec: ResolvedDesignSpec,
    size: int = 3,
    min_spacing: int | None = None,
    max_shared_off: int = 0,
) -> Panel:
    """Greedy submodular selection under spacing and diversity constraints.

    ``min_spacing`` defaults to ``spec.spacer_length`` -- one full spacer, so
    two members cannot share a target base. It used to default to a bare ``15``,
    shorter than every ortholog's spacer (22, 28 and 30 nt), so a panel could
    contain two guides that ``g_output`` then marked as the *same site*. Two
    answers from one run, disagreeing, with the panel being the half a user
    actually orders. Pass an explicit value only to widen it.
    """
    if min_spacing is None:
        min_spacing = spec.spacer_length
    required = spec.contract.by_class(TargetClass.REQUIRED)
    weights = {t.target_id: max(t.weight, 1.0) for t in required}

    pool = [(a, _profile(a)) for a in assessments]
    chosen: list[GuideAssessment] = []
    chosen_profiles: list[_Profile] = []
    dropped: list[str] = []
    # Membership by identity, not by value. ``GuideAssessment`` is a frozen
    # dataclass holding a tuple of ``PairEvidence``, each of which holds a
    # ``SearchHit`` and two ``Interval``s, so ``assessment in chosen`` was a
    # deep structural comparison over the whole evidence tree for every
    # candidate on every round -- and it is also the wrong question: two
    # distinct guides that happened to compare equal would collapse.
    chosen_ids: set[int] = set()

    while len(chosen) < size:
        best_gain, best = 0.0, None
        current = _objective(chosen_profiles, weights)
        for assessment, prof in pool:
            if id(assessment) in chosen_ids:
                continue
            if not _compatible(prof, chosen_profiles, min_spacing, max_shared_off):
                continue
            gain = _objective(chosen_profiles + [prof], weights) - current
            if gain > best_gain:
                best_gain, best = gain, (assessment, prof)
        if best is None:
            break
        chosen.append(best[0])
        chosen_ids.add(id(best[0]))
        chosen_profiles.append(best[1])

    if len(chosen) < size:
        dropped.append(
            f"panel stopped at {len(chosen)}/{size}: no remaining guide added "
            "coverage without violating spacing/diversity constraints"
        )

    # Whether the diversity constraint *can* bind, not how many candidates
    # there are. The two have nothing to do with each other: `max_shared_off`
    # is a count of shared off-target loci, and comparing it to the pool size
    # meant a large `max_shared_off` declared the run unconstrained while
    # `_compatible` was still rejecting pairs -- printing the (1 - 1/e) bound
    # for a feasible family that is an independence system, which is the
    # claim fcb1bd8 removed from the other branch.
    widest_off = max((len(p.off_loci) for _a, p in pool), default=0)
    constrained = min_spacing > 0 or max_shared_off < widest_off
    guarantee = (
        "greedy heuristic -- NO approximation guarantee: spacing/diversity "
        "feasibility is an independence system, not a matroid, so the panel can "
        "be arbitrarily worse than the best feasible one"
        if constrained
        else "greedy >= (1 - 1/e) of optimum (cardinality-constrained submodular maximization)"
    )

    coverage: dict[str, float] = {}
    for target_id in weights:
        miss = 1.0
        for p in chosen_profiles:
            miss *= 1.0 - p.coverage.get(target_id, 0.0)
        coverage[target_id] = 1.0 - miss

    return Panel(
        guides=tuple(chosen),
        coverage=coverage,
        objective=_objective(chosen_profiles, weights),
        guarantee=guarantee,
        dropped=tuple(dropped),
    )
