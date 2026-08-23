"""Sensitivity-bounded search with a *provable* exhaustiveness claim (fix #4).

The original design reported a soft "recall/completeness status". That is a
promise; downstream code and reviewers need a claim they can act on. Every
index here returns an :class:`ExhaustivenessBound` stating exactly what was
guaranteed -- e.g. "exhaustive to <=4 mismatches, wobble-aware, over
{coding, ncrna}" -- and anything past that bound is labelled heuristic.

Two implementations share the :class:`SearchIndex` seam:

* :class:`ExhaustiveScanIndex` -- reference; provably complete by construction.
* :class:`SeedHeuristicIndex`  -- fast, and honest that it may miss hits whose
  mismatches are spread across every seed. Its bound reports
  ``exhaustive=False``, which the gate turns into an R2 REVIEW.

A production bidirectional FM-index plugs in behind the same Protocol and can
claim ``exhaustive=True`` for its supported envelope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..contract.enums import Universe
from ..models.evidence import SearchHit
from ..util.duplex import WOBBLE_SUBSTITUTIONS, profile_duplex
from ..util.seq import revcomp
from .transcriptome import Transcriptome


@dataclass(frozen=True, slots=True)
class ExhaustivenessBound:
    """What the search actually guarantees."""

    max_mismatches: int
    wobble_aware: bool
    universes: frozenset[Universe]
    exhaustive: bool
    method: str
    note: str = ""

    def covers(self, required_mismatches: int) -> bool:
        return self.exhaustive and self.max_mismatches >= required_mismatches

    def describe(self) -> str:
        kind = "exhaustive" if self.exhaustive else "heuristic"
        uni = ",".join(sorted(u.value for u in self.universes))
        return f"{kind} to <={self.max_mismatches}mm (wobble-aware={self.wobble_aware}) over {{{uni}}}"


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    bound: ExhaustivenessBound


@runtime_checkable
class SearchIndex(Protocol):
    version: str

    def search(self, guide_id: str, spacer: str, max_mismatches: int) -> SearchResult: ...


def _scan_locus(expected: str, sequence: str, max_mismatches: int):
    """Yield ``(start, window, n_mismatch)`` for windows within the envelope.

    Wobble substitutions (C->U, A->G in target space) still pair, so they do
    not count against the mismatch budget.
    """
    length = len(expected)
    for start in range(len(sequence) - length + 1):
        window = sequence[start : start + length]
        mismatches = 0
        for exp_b, obs_b in zip(expected, window):
            if exp_b == obs_b:
                continue
            if (exp_b, obs_b) in WOBBLE_SUBSTITUTIONS:
                continue
            mismatches += 1
            if mismatches > max_mismatches:
                break
        else:
            yield start, window, mismatches


class ExhaustiveScanIndex:
    """Full scan over the declared universes. Complete by construction."""

    version = "index/exhaustive-scan/v1"

    def __init__(self, transcriptome: Transcriptome, universes: frozenset[Universe]) -> None:
        self.transcriptome = transcriptome
        self.universes = universes
        self._loci = transcriptome.in_universes(universes)

    def search(self, guide_id: str, spacer: str, max_mismatches: int) -> SearchResult:
        expected = revcomp(spacer)
        hits: list[SearchHit] = []
        for locus in self._loci:
            for start, window, _n in _scan_locus(expected, locus.sequence, max_mismatches):
                flank5 = locus.sequence[start - 1] if start > 0 else None
                end = start + len(expected)
                flank3 = locus.sequence[end] if end < len(locus.sequence) else None
                hits.append(
                    SearchHit(
                        guide_id=guide_id,
                        locus_id=locus.locus_id,
                        target_site=window,
                        start=start,
                        duplex=profile_duplex(spacer, window),
                        biotype=locus.biotype,
                        gene=locus.gene,
                        transcript=locus.transcript,
                        five_prime_flank=flank5,
                        three_prime_flank=flank3,
                    )
                )
        bound = ExhaustivenessBound(
            max_mismatches=max_mismatches,
            wobble_aware=True,
            universes=self.universes,
            exhaustive=True,
            method=self.version,
        )
        return SearchResult(hits=tuple(hits), bound=bound)


class SeedHeuristicIndex:
    """Seed-anchored search: fast, and explicitly *not* exhaustive."""

    version = "index/seed-heuristic/v1"

    def __init__(
        self,
        transcriptome: Transcriptome,
        universes: frozenset[Universe],
        seed_length: int = 11,
    ) -> None:
        self.transcriptome = transcriptome
        self.universes = universes
        self.seed_length = seed_length
        self._loci = transcriptome.in_universes(universes)

    def search(self, guide_id: str, spacer: str, max_mismatches: int) -> SearchResult:
        expected = revcomp(spacer)
        k = self.seed_length
        # (seed, offset) pairs, not a set of seeds: when the 5' and 3' seeds
        # are identical a set collapses them and silently drops 3' anchoring.
        seeds = [(expected[:k], 0), (expected[-k:], len(expected) - k)]
        hits: list[SearchHit] = []
        for locus in self._loci:
            seq = locus.sequence
            starts: set[int] = set()
            for seed, offset in seeds:
                pos = seq.find(seed)
                while pos != -1:
                    starts.add(pos - offset)
                    pos = seq.find(seed, pos + 1)
            for start in sorted(s for s in starts if 0 <= s <= len(seq) - len(expected)):
                window = seq[start : start + len(expected)]
                profile = profile_duplex(spacer, window)
                if profile.n_mismatch > max_mismatches:
                    continue
                flank5 = seq[start - 1] if start > 0 else None
                end = start + len(expected)
                flank3 = seq[end] if end < len(seq) else None
                hits.append(
                    SearchHit(
                        guide_id=guide_id,
                        locus_id=locus.locus_id,
                        target_site=window,
                        start=start,
                        duplex=profile,
                        biotype=locus.biotype,
                        gene=locus.gene,
                        transcript=locus.transcript,
                        five_prime_flank=flank5,
                        three_prime_flank=flank3,
                    )
                )
        bound = ExhaustivenessBound(
            max_mismatches=max_mismatches,
            wobble_aware=True,
            universes=self.universes,
            exhaustive=False,
            method=self.version,
            note=(
                f"requires an exact {k}nt terminal seed; hits with mismatches "
                "in both seeds are not guaranteed to be found"
            ),
        )
        return SearchResult(hits=tuple(hits), bound=bound)
