"""Stage A -- candidate design.

Tile the intended transcripts, mark junction-spanning and unique-region
guides, validate coordinates, and deduplicate identical spacers. No scoring
happens here; the stage only proposes.
"""

from __future__ import annotations

import hashlib

from ..contract.enums import TargetClass
from ..contract.spec import ResolvedDesignSpec, TargetSpec
from ..models.evidence import Candidate
from ..util.seq import revcomp, windows


def guide_id_for(spacer: str) -> str:
    """Stable, content-addressed guide id so reruns line up."""
    return "g_" + hashlib.sha1(spacer.encode()).hexdigest()[:10]


def _junction_spanning(start: int, length: int, junctions: tuple[int, ...]) -> bool:
    return any(start < j < start + length for j in junctions)


def tile_target(
    target: TargetSpec,
    spacer_length: int,
    step: int = 1,
    junctions: tuple[int, ...] = (),
) -> list[Candidate]:
    """Every window of ``spacer_length`` on one target, as spacers.

    The spacer is the reverse complement of the target site: Cas13 binds its
    target as an antiparallel RNA:RNA duplex.
    """
    out: list[Candidate] = []
    seq = target.sequence
    for start, site in windows(seq, spacer_length, step):
        spacer = revcomp(site)
        end = start + spacer_length
        out.append(
            Candidate(
                guide_id=guide_id_for(spacer),
                spacer=spacer,
                source_target_id=target.target_id,
                start=start,
                five_prime_flank=seq[start - 1] if start > 0 else None,
                three_prime_flank=seq[end] if end < len(seq) else None,
                spans_junction=_junction_spanning(start, spacer_length, junctions),
            )
        )
    return out


def design_candidates(
    spec: ResolvedDesignSpec,
    step: int = 1,
    junctions_by_target: dict[str, tuple[int, ...]] | None = None,
) -> tuple[Candidate, ...]:
    """Tile every intended (REQUIRED or DESIRED) target and deduplicate.

    Deduplication is by spacer sequence: the same spacer reached from two
    targets is one guide with two intended sites, not two guides.
    """
    junctions_by_target = junctions_by_target or {}
    seen: dict[str, Candidate] = {}
    for target in spec.contract.targets:
        if target.target_class not in (TargetClass.REQUIRED, TargetClass.DESIRED):
            continue
        if len(target.sequence) < spec.spacer_length:
            continue
        for cand in tile_target(
            target,
            spec.spacer_length,
            step=step,
            junctions=junctions_by_target.get(target.target_id, ()),
        ):
            seen.setdefault(cand.guide_id, cand)
            if len(seen) >= spec.budget.max_candidates:
                return tuple(seen.values())
    return tuple(seen.values())
