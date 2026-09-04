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

from ..contract.enums import ASSAY_REQUIRED_UNIVERSES, Universe
from ..contract.spec import ResolvedDesignSpec
from ..models.evidence import Candidate, SearchHit
from ..services.indexes import ExhaustivenessBound, SearchIndex, is_weaker


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    hits_by_guide: dict[str, tuple[SearchHit, ...]]
    bound: ExhaustivenessBound
    universe_gaps: frozenset[Universe]
    #: Loci whose hit is sequence-identical to a site in one of this run's own
    #: targets, yet carries no contract identity. Empty for an index built from
    #: the run's Transcriptome, because `locus_to_target` names those loci.
    #: Non-empty means a search reached a corpus this run did not assemble --
    #: `--blast-db` -- where the target's own transcript is present under an
    #: accession no key can match, and is therefore scored as an off-target.
    self_identical_loci: frozenset[str] = frozenset()


def universe_gaps(
    spec: ResolvedDesignSpec, bound: ExhaustivenessBound
) -> frozenset[Universe]:
    """Universes this assay is exposed to that the search did not actually cover.

    Keyed on what the index *proved* it searched (``bound.universes``), not on
    what the spec declared. Those are different things, and trusting the
    declaration was a hole straight through the universe-adequacy gate: an
    index built over a narrower universe set than the spec declares returns a
    perfectly clean off-target report, and because ``missing_universes`` only
    ever compared ``ASSAY_REQUIRED`` against ``declared_universes`` the gate saw
    no gap, raised no REVIEW and recorded no shortfall. In a reproduction, the
    same spec and transcriptome went from 42 FAIL to 42 PASS with zero
    off-target rows purely by narrowing the index -- rRNA hits at 5000 TPM
    simply ceased to exist.

    The declared set is unioned in so that declaring a universe and then not
    searching it is also a gap; when the index is built from
    ``declared_universes`` (as every caller in this package does) the result is
    identical to the old computation.
    """
    required = ASSAY_REQUIRED_UNIVERSES[spec.profile.assay]
    return frozenset((required | spec.context.declared_universes) - bound.universes)


def self_identical_loci(
    hits_by_guide: dict[str, tuple[SearchHit, ...]], spec: ResolvedDesignSpec
) -> frozenset[str]:
    """Unannotated hits whose site is a site in one of this run's targets.

    `annotate_hit` names a hit by looking its locus up in ``locus_to_target``,
    which is keyed on the ids of the loci this run assembled. A BLAST subject
    id is an accession from someone else's database, so the target's own
    transcript -- which a transcriptome database contains -- comes back as an
    unannotated perfect hit and is scored as the strongest off-target evidence
    there is, for the site the guide was designed from.

    This does not fix it by reclassifying. A site identical to the target's is
    either the target or a perfect-identity paralog, and nothing here can tell
    them apart; attributing it to the target would hide a genuine off-target,
    which is the worse error for a tool whose subject is specificity. The
    hazard stays counted and the run reports what it could not distinguish.
    """
    sites = [t.sequence for t in spec.contract.targets if t.sequence]
    if not sites:
        return frozenset()
    found: set[str] = set()
    for hits in hits_by_guide.values():
        for hit in hits:
            if hit.target_id is not None or hit.duplex.n_mismatch:
                continue
            if any(hit.target_site in sequence for sequence in sites):
                found.add(hit.locus_id)
    return frozenset(found)


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
        # Keep the *weakest* bound rather than the last one written. Both
        # shipped indexes return a query-independent bound, so today every
        # iteration writes an equivalent value -- but the SearchIndex Protocol
        # constrains only the return type, and an index that degrades to a
        # heuristic for some spacers would have its exhaustive=False bound
        # overwritten by the next candidate's exhaustive=True one, and the gate
        # would then pass a guide whose search was never exhaustive.
        if bound is None or is_weaker(result.bound, bound):
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
        universe_gaps=universe_gaps(spec, bound),
        self_identical_loci=self_identical_loci(hits_by_guide, spec),
    )
