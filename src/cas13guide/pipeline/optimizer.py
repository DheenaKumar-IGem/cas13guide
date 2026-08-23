"""Guide-set optimizer -- named, not just described (fix #11).

Panel design maximizes

    f(S) = sum_t  w_t * (1 - prod_{g in S} (1 - p_gt))

the expected number of required targets covered by at least one guide, where
``p_gt`` is guide ``g``'s calibrated *lower* bound on target ``t``. That
function is monotone and submodular, so greedy selection carries a real
approximation guarantee:

* under a pure cardinality constraint (panel size ``k``): ``1 - 1/e``;
* with the spacing/diversity independence constraints switched on the problem
  becomes maximization over a matroid-like family, where greedy guarantees
  ``1/2``. The looser bound is reported honestly rather than quietly claimed
  as ``1 - 1/e``.

Constraints applied during selection:

* **spacing** -- two guides tiling the same target must not sit within
  ``min_spacing`` nt of each other, so a panel cannot be three shifted copies
  of one site;
* **diversity** -- a new guide must not share more than ``max_shared_off``
  off-target loci with any already-selected guide, so panel members fail
  independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contract.enums import TargetClass
from ..contract.spec import ResolvedDesignSpec
from ..metrics.bounds import benefit_lower
from ..models.evidence import GuideAssessment


@dataclass(frozen=True, slots=True)
class Panel:
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
    coverage: dict[str, float] = {}
    sites: dict[str, int] = {}
    off: set[str] = set()
    for e in assessment.evidence:
        if e.target_class is TargetClass.REQUIRED and e.hit.target_id:
            tid = e.hit.target_id
            if benefit_lower(e.activity) > coverage.get(tid, 0.0):
                coverage[tid] = benefit_lower(e.activity)
                sites[tid] = e.hit.start
        elif not e.is_intended:
            off.add(e.locus_id)
    return _Profile(coverage=coverage, off_loci=frozenset(off), sites=sites)


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
    min_spacing: int = 15,
    max_shared_off: int = 0,
) -> Panel:
    """Greedy submodular selection under spacing and diversity constraints."""
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

    constrained = min_spacing > 0 or max_shared_off < len(assessments)
    guarantee = (
        "greedy >= 1/2 of optimum (matroid-constrained submodular maximization)"
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
