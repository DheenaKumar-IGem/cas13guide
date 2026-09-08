"""Stage A -- candidate design (FR2.1, FR2.2).

Tile the intended transcripts with a sliding window (FR2.1), tag each candidate
with the flanking bases that make up its protospacer flanking site (FR2.2), mark
junction-spanning guides, validate coordinates, and deduplicate identical
spacers. No scoring happens here; the stage only proposes.

The PFS bases are recorded rather than judged: whether a given flank is a hard
incompatibility or a soft preference is an ortholog property, and it is decided
in stage B against ``contract/effectors.py``.
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

    The candidate budget is shared out **per intended target**, not spent in
    contract order. Spending it in order meant the first target consumed
    whatever it needed and later ones got the remainder -- down to nothing, at
    which point the function returned and a REQUIRED target was never tiled at
    all. That is reachable at the shipped default: ``max_candidates`` is 5000
    and an ordinary 3 kb transcript yields ~2979 windows at a 22 nt spacer, so
    three REQUIRED transcripts already starve the third to zero. The run then
    reported "no feasible panel covers required target(s): t3", which reads as
    "t3 has no usable guides" when the truth is that nobody looked -- and
    reordering the contract's targets silently changed which one it was.

    Within a target the share is spent by *widening the step* rather than by
    truncating, so a target that cannot be tiled densely is still spanned end
    to end. A sparse tiling of the whole transcript is a worse answer than a
    dense one; half a transcript at full density is a wrong one.
    """
    junctions_by_target = junctions_by_target or {}
    intended = [
        target
        for target in spec.contract.targets
        if target.target_class in (TargetClass.REQUIRED, TargetClass.DESIRED)
        and len(target.sequence) >= spec.spacer_length
    ]
    if not intended:
        return ()

    budget = spec.budget.max_candidates
    share, remainder = divmod(budget, len(intended))
    seen: dict[str, Candidate] = {}
    for position, target in enumerate(intended):
        # The remainder goes to the earliest targets, so the split is
        # deterministic rather than dependent on iteration luck.
        allowance = max(1, share + (1 if position < remainder else 0))
        offered = len(range(0, max(0, len(target.sequence) - spec.spacer_length) + 1, step))

        # Which of this target's windows to keep, when it offers more than its
        # share. Evenly spaced *by index* rather than by widening the step to an
        # integer: an integer step overshoots badly just above 1, where
        # ceil(2973 / 2500) = 2 keeps 1487 windows against an allowance of
        # 2500 -- 40% of the share unspent, and the target tiled at half the
        # density the budget allowed, for nothing. Even spacing keeps both ends
        # and uses the whole allowance, so the transcript is still spanned.
        keep: set[int] | None = None
        if offered > allowance:
            if allowance == 1:
                keep = {0}
            else:
                keep = {
                    round(i * (offered - 1) / (allowance - 1))
                    for i in range(allowance)
                }

        for index, cand in enumerate(
            tile_target(
                target,
                spec.spacer_length,
                step=step,
                junctions=junctions_by_target.get(target.target_id, ()),
            )
        ):
            # `junctions` can add windows beyond the plain stride, so index them
            # past the end of the plan rather than dropping them: a
            # junction-spanning candidate is deliberately generated and is not
            # the thing the budget is trying to thin out.
            if keep is not None and index < offered and index not in keep:
                continue
            seen.setdefault(cand.guide_id, cand)
    return tuple(seen.values())
