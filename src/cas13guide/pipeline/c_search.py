"""Stage C -- sensitivity-bounded search.

Runs the index over the declared universes, annotates each hit against the
contract, and reports two things the gate depends on:

* the :class:`ExhaustivenessBound` the index could actually prove, and
* the **universe adequacy** verdict (fix #3) -- which required universes for
  this assay were never searched. A missing universe does not yield a warning
  and a slightly worse score; it yields a suspiciously *clean* off-target
  report, which is why it is a gate condition.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..contract.enums import Universe
from ..contract.spec import ResolvedDesignSpec
from ..models.evidence import Candidate, SearchHit
from ..services.indexes import ExhaustivenessBound, SearchIndex


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    hits_by_guide: dict[str, tuple[SearchHit, ...]]
    bound: ExhaustivenessBound
    universe_gaps: frozenset[Universe]

    @property
    def universe_adequate(self) -> bool:
        return not self.universe_gaps


def annotate_hit(hit: SearchHit, spec: ResolvedDesignSpec, locus_to_target: dict[str, str]) -> SearchHit:
    """Attach contract identity to a raw hit, when the locus is named."""
    target_id = locus_to_target.get(hit.locus_id)
    if target_id is None:
        return hit
    target = spec.contract.get(target_id)
    if target is None:
        return hit
    return dataclasses.replace(hit, target_id=target_id, target_class=target.target_class)


def run_search(
    candidates: tuple[Candidate, ...],
    spec: ResolvedDesignSpec,
    index: SearchIndex,
    locus_to_target: dict[str, str],
    max_mismatches: int | None = None,
) -> SearchOutcome:
    envelope = (
        max_mismatches
        if max_mismatches is not None
        else spec.thresholds.required_search_mismatches
    )
    hits_by_guide: dict[str, tuple[SearchHit, ...]] = {}
    bound: ExhaustivenessBound | None = None

    for cand in candidates:
        result = index.search(cand.guide_id, cand.spacer, envelope)
        bound = result.bound
        hits_by_guide[cand.guide_id] = tuple(
            annotate_hit(h, spec, locus_to_target) for h in result.hits
        )

    if bound is None:  # no candidates survived to be searched
        bound = ExhaustivenessBound(
            max_mismatches=envelope,
            wobble_aware=True,
            universes=spec.context.declared_universes,
            exhaustive=False,
            method="index/not-run",
            note="no candidates reached the search stage",
        )

    return SearchOutcome(
        hits_by_guide=hits_by_guide,
        bound=bound,
        universe_gaps=spec.context.missing_universes(spec.profile.assay),
    )
