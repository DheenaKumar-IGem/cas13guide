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

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

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
    #: Universes the corpus behind this index holds a sequence in, or ``None``
    #: when the index cannot know -- a remote BLAST database, say. It is the
    #: only thing that separates "the search skipped sequences it had" from
    #: "there were none": both leave the same hole in the off-target table and
    #: only one is the run's fault. An index that cannot prove the sequences
    #: were there does not get to have the run accused of skipping them.
    corpus_universes: frozenset[Universe] | None = None

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


def is_weaker(candidate: ExhaustivenessBound, current: ExhaustivenessBound) -> bool:
    """Order two bounds by how much they actually guarantee.

    A non-exhaustive bound is always weaker than an exhaustive one; between
    two of the same kind, the shallower mismatch depth wins. Comparing only
    ``exhaustive`` would ignore an index that honestly reports a shallower
    supported depth, and ``SEARCH_NOT_EXHAUSTIVE`` would never fire for it.

    Lives here rather than in the orchestrator because two stages fold bounds
    -- ``c_search`` across candidates, the orchestrator across contexts -- and
    "weakest wins" has to mean the same thing in both.
    """
    if candidate.exhaustive != current.exhaustive:
        return not candidate.exhaustive
    return candidate.max_mismatches < current.max_mismatches


@runtime_checkable
class SearchIndex(Protocol):
    version: str

    def search(self, guide_id: str, spacer: str, max_mismatches: int) -> SearchResult: ...


#: Positions between subsequent compactions once a live set exists.
_PRUNE_EVERY = 4

#: Placed between loci in the concatenated corpus. Not a base, so it pairs with
#: nothing and every window overlapping it is charged a mismatch. This is the
#: first of two independent boundary defenses; see
#: ``ExhaustiveScanIndex._within_locus`` for the exact one.
_CORPUS_SEPARATOR = "#"


def _prune_after(max_mismatches: int) -> int:
    """Positions scanned across *every* window before the first compaction.

    Compaction trades a contiguous slice over the whole corpus for a gather
    over the survivors, and a gather costs several times more per element, so
    it only pays once the survivors are a small fraction of the corpus. How
    long that takes is governed by the mismatch budget, not by a constant: with
    a four-letter background and one wobble partner per position, the fraction
    of windows still within budget after ``j`` positions is ~100% until roughly
    ``j = 2k``, then falls away quickly. Measured over 1 Mb of random RNA with
    a 22-mer:

    ====  =====================  =====================
    k     survival after 6       survival after 2k+4
    ====  =====================  =====================
    2     34%                    14%  (j=8)
    4     89%                    19%  (j=12)
    6     100%                   ~20% (j=16)
    ====  =====================  =====================

    A fixed ``6`` -- the value this started with, on the mistaken belief that
    almost nothing survives six positions -- therefore compacted while ~89% of
    windows were still live at the default ``k=4``, paying for a full-corpus
    gather that removed a tenth of the work. ``2k + 4`` compacts near the 20%
    line and measured ~1.8x faster at ``k=4``; the hit set is unchanged either
    way, because ``cost`` only ever increases, so a window dropped at any
    compaction had already exceeded the budget and could never re-enter it.
    """
    return 2 * max_mismatches + 4


@lru_cache(maxsize=None)
def _allowed_bases(expected_base: str) -> tuple[str, ...]:
    """Target bases that pair with the spacer base behind ``expected_base``.

    Exactly :data:`~cas13guide.util.duplex.WOBBLE_SUBSTITUTIONS` re-expressed
    as a lookup, so the wobble rule still has one definition.
    """
    allowed = {expected_base}
    allowed.update(obs for exp, obs in WOBBLE_SUBSTITUTIONS if exp == expected_base)
    return tuple(sorted(allowed))


def _encode(sequence: str) -> np.ndarray:
    """One byte per base, for vectorized comparison. One byte per base of RAM."""
    return np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)


def _scan_array(
    encoded: np.ndarray, expected: str, max_mismatches: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, costs)`` for every window within the envelope, starts ascending.

    The core of the scan. Windows are compared one position at a time across
    the whole array at once, and the surviving set is compacted every few
    positions -- see :func:`_prune_after` for why the compaction point depends
    on the budget.
    """
    length = len(expected)
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int16))
    if length == 0:
        return empty

    width = encoded.shape[0] - length + 1
    if width <= 0:
        return empty

    prune_after = _prune_after(max_mismatches)
    #: Ascending starts still within budget; ``None`` means "all of them".
    live: np.ndarray | None = None
    cost = np.zeros(width, dtype=np.int16)

    for offset, expected_base in enumerate(expected):
        allowed = _allowed_bases(expected_base)
        column = encoded[offset : offset + width] if live is None else encoded[live + offset]
        paired = column == ord(allowed[0])
        for base in allowed[1:]:
            paired |= column == ord(base)
        cost += ~paired

        compact = (offset + 1 == prune_after) or (
            live is not None and (offset + 1 - prune_after) % _PRUNE_EVERY == 0
        )
        if compact and offset + 1 < length:
            keep = cost <= max_mismatches
            live = np.flatnonzero(keep) if live is None else live[keep]
            cost = cost[keep]
            if live.size == 0:
                return empty

    if live is None:
        starts = np.flatnonzero(cost <= max_mismatches)
        return starts, cost[starts]
    keep = cost <= max_mismatches
    return live[keep], cost[keep]


def _scan_encoded(
    encoded: np.ndarray, sequence: str, expected: str, max_mismatches: int
):
    """The scan over one pre-encoded locus. See :func:`_scan_locus`."""
    length = len(expected)
    if length == 0:
        # Degenerate, but preserve the original generator's behaviour exactly.
        for start in range(len(sequence) + 1):
            yield start, "", 0
        return

    starts, costs = _scan_array(encoded, expected, max_mismatches)
    for start, mismatches in zip(starts.tolist(), costs.tolist()):
        yield start, sequence[start : start + length], mismatches


def _scan_locus(expected: str, sequence: str, max_mismatches: int):
    """Yield ``(start, window, n_mismatch)`` for windows within the envelope.

    Wobble substitutions (C->U, A->G in target space) still pair, so they do
    not count against the mismatch budget.

    Vectorized over the locus rather than looped per character. The scan is the
    single hot path of the whole package -- it was 77% of the bundled demo's
    runtime -- and the per-character version cost ~1.8 s per 22-mer query per
    megabase at ``k=4``. Windows are compared a position at a time across the
    whole corpus at once, and the surviving set is compacted every few
    positions, which is what makes it fast: the live set collapses quickly once
    the budget starts binding (see :func:`_prune_after`).

    NFR1 gives a 60 s budget for a <=5 kb transcript; the vectorized scan puts a
    full design run on that scale at well under a second, so the budget is not
    the constraint it once was.

    The hit set is *identical* to the per-character scan, including start
    order and the reported mismatch count -- ``tests/test_search.py`` pins the
    equivalence against a brute-force reference. That matters more than the
    speed: this index is the one that gets to claim ``exhaustive=True``.
    """
    yield from _scan_encoded(_encode(sequence), sequence, expected, max_mismatches)


class ExhaustiveScanIndex:
    """Full scan over every locus admitted by the filter.

    Complete with respect to the corpus, which is not the same as complete
    with respect to the universes the filter names: a universe the corpus
    holds no sequence in is reported as absent rather than as searched.
    """

    version = "index/exhaustive-scan/v1"

    def __init__(self, transcriptome: Transcriptome, universes: frozenset[Universe]) -> None:
        self.transcriptome = transcriptome
        self.requested_universes = universes
        self._loci = transcriptome.in_universes(universes)
        #: What the bound reports. The filter is what this index was *willing*
        #: to search; this is what it has. Reporting the filter made the
        #: ledger's claim and the CRITICAL limitation name every universe the
        #: caller admitted, for a corpus that held one -- and made
        #: `universe_gaps` subtract the declaration from itself, so the gate
        #: whose whole purpose is to catch a suspiciously clean off-target
        #: report could not fire on the documented command.
        self.universes = frozenset(locus.biotype for locus in self._loci)
        # Built lazily on the first query. Building it eagerly would cost one
        # byte per base up front and make "the scanner costs nothing to build"
        # -- which the index-selection cost model relies on -- untrue for an
        # index that is never queried.
        self._corpus: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        self._ends: np.ndarray | None = None

    def _built(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode every locus once, into one array, with locus spans alongside.

        Scanning locus by locus meant ~22 numpy calls per locus per query, and
        a real transcriptome is hundreds of thousands of loci of a couple of
        kilobases each -- the regime where numpy's per-call overhead, not the
        comparisons, is the cost. Measured on 3 Mb split into 2 kb loci, the
        per-locus form ran at 0.116 s/Mb/query against 0.016 for a single large
        array: a 7x penalty paid purely in dispatch.

        Loci are joined by :data:`_CORPUS_SEPARATOR`, which is not a base and
        so pairs with nothing. That is the first of two independent defenses
        against a window spanning two loci -- the same belt-and-braces the
        FM-index uses -- and it is *not* sufficient on its own: a separator
        costs only one mismatch, so at a budget of 4 a window could still
        straddle a boundary and stay inside the envelope. :meth:`_within_locus`
        is the exact check.
        """
        if self._corpus is None:
            pieces: list[str] = []
            offsets: list[int] = []
            cursor = 0
            for locus in self._loci:
                offsets.append(cursor)
                pieces.append(locus.sequence)
                pieces.append(_CORPUS_SEPARATOR)
                cursor += len(locus.sequence) + len(_CORPUS_SEPARATOR)
            self._corpus = _encode("".join(pieces))
            self._offsets = np.array(offsets, dtype=np.int64)
            self._ends = np.array(
                [start + len(locus.sequence) for start, locus in zip(offsets, self._loci)],
                dtype=np.int64,
            )
        # Narrowed for the return: all three are filled together above, and a
        # reader (or a type checker) should not have to prove that from the
        # branch structure.
        assert self._corpus is not None
        assert self._offsets is not None
        assert self._ends is not None
        return self._corpus, self._offsets, self._ends

    def _within_locus(self, starts: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Drop windows that cross a locus boundary; return ``(locus_index, start)``.

        The exact defense. ``searchsorted`` names the locus each global start
        falls in, and the window is kept only if it ends inside that same
        locus. Starts arrive ascending and both operations preserve order, so
        the surviving hits come out in (locus order, local start) order --
        exactly what scanning locus by locus produced.
        """
        _corpus, offsets, ends = self._built()
        if starts.size == 0 or offsets.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        index = np.searchsorted(offsets, starts, side="right") - 1
        keep = (index >= 0) & (starts + length <= ends[index])
        return index[keep], starts[keep] - offsets[index[keep]]

    def search(self, guide_id: str, spacer: str, max_mismatches: int) -> SearchResult:
        expected = revcomp(spacer)
        length = len(expected)
        corpus, _offsets, _ends = self._built()
        raw_starts, _costs = _scan_array(corpus, expected, max_mismatches)
        indexes, locals_ = self._within_locus(raw_starts, length)

        hits: list[SearchHit] = []
        for position, start in zip(indexes.tolist(), locals_.tolist()):
            locus = self._loci[position]
            sequence = locus.sequence
            window = sequence[start : start + length]
            end = start + length
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
                    five_prime_flank=sequence[start - 1] if start > 0 else None,
                    three_prime_flank=sequence[end] if end < len(sequence) else None,
                )
            )
        bound = ExhaustivenessBound(
            max_mismatches=max_mismatches,
            wobble_aware=True,
            universes=self.universes,
            exhaustive=True,
            method=self.version,
            corpus_universes=self.transcriptome.covered_universes(),
        )
        return SearchResult(hits=tuple(hits), bound=bound)

    def any_hit(self, spacer: str, max_mismatches: int) -> bool:
        """Does ``spacer`` bind anywhere within the envelope?

        Exactly ``bool(self.search(...).hits)``, but it stops after the array
        scan: no ``SearchHit`` and no ``profile_duplex`` per match. Control
        generation asks this question up to ``attempts`` times per panel member
        and throws away everything else a full search produces -- and at the
        default envelope a random scramble almost always hits something, so
        almost every one of those searches was enumerated in full and discarded.

        The boundary filter still runs: a window that only "hits" by straddling
        two loci is not a hit, and must not disqualify a control.
        """
        expected = revcomp(spacer)
        corpus, _offsets, _ends = self._built()
        starts, _costs = _scan_array(corpus, expected, max_mismatches)
        indexes, _locals = self._within_locus(starts, len(expected))
        return bool(indexes.size)


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
        self.requested_universes = universes
        self.seed_length = seed_length
        self._loci = transcriptome.in_universes(universes)
        # See ExhaustiveScanIndex: the bound reports what is held, not the
        # filter that was admitted.
        self.universes = frozenset(locus.biotype for locus in self._loci)

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
            corpus_universes=self.transcriptome.covered_universes(),
            note=(
                f"requires an exact {k}nt terminal seed; hits with mismatches "
                "in both seeds are not guaranteed to be found"
            ),
        )
        return SearchResult(hits=tuple(hits), bound=bound)


class UnionSearchIndex:
    """Several indexes searched as one, hits merged, weakest bound reported.

    Why this has to exist. ``--blast-db`` *replaced* the context's index, and
    the same index answers both questions the pipeline asks: where does this
    guide bind off-target, and does it bind its intended target at all. A BLAST
    database answers only the first -- it never sees the ``Transcriptome`` the
    run was built around, and its subject ids are database accessions that
    ``annotate_hit`` cannot match to the run's own target ids. So every guide
    came back with no on-target evidence, ``required_coverage`` fell to
    ``NO_EVIDENCE``, and a run with ``--blast-db`` failed **every** candidate:
    measured 39/39 FAIL against 28 REVIEW for the identical design without it.
    A flag that turns every guide into a FAIL is indistinguishable from a
    target that has no usable guides.

    So BLAST is *additional*, not alternative: the local exhaustive scan keeps
    answering the on-target question and BLAST widens the off-target corpus,
    which is what the README says the flag is for.

    Two invariants:

    * **Deduplicated by identity of the site**, ``(locus_id, start)``. Two
      indexes over overlapping corpora would otherwise report one off-target
      twice, and ``cumulative_burden`` is a sum -- double-counting a locus
      inflates the hazard of the guide that found it twice.
    * **The bound is the weakest**, via :func:`is_weaker`, and the universes are
      the union. A heuristic member makes the whole search heuristic, which is
      the honest reading: the union inherits the doubt of its worst member. The
      union of universes is likewise right, since a hit anywhere in either
      corpus is a hit.
    """

    def __init__(self, indexes: Sequence[SearchIndex]) -> None:
        if not indexes:
            raise ValueError("a union of no indexes cannot search anything")
        self.indexes = tuple(indexes)
        self.version = (
            "index/union/v1["
            + "+".join(getattr(index, "version", "?") for index in self.indexes)
            + "]"
        )

    def search(self, guide_id: str, spacer: str, max_mismatches: int) -> SearchResult:
        hits: list[SearchHit] = []
        seen: set[tuple[str, int]] = set()
        bound: ExhaustivenessBound | None = None
        universes: set[Universe] = set()
        corpora: list[frozenset[Universe] | None] = []
        for index in self.indexes:
            result = index.search(guide_id, spacer, max_mismatches)
            corpora.append(result.bound.corpus_universes)
            for hit in result.hits:
                key = (hit.locus_id, hit.start)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(hit)
            universes |= result.bound.universes
            if bound is None or is_weaker(result.bound, bound):
                bound = result.bound
        assert bound is not None  # non-empty by construction
        hits.sort(key=lambda h: (h.locus_id, h.start))
        return SearchResult(
            hits=tuple(hits),
            bound=replace(
                bound,
                universes=frozenset(universes),
                method=self.version,
                # None from any member poisons the union: one index that cannot
                # see its corpus is enough to make the union unable to prove a
                # sequence was available.
                corpus_universes=(
                    None
                    if any(c is None for c in corpora)
                    else frozenset(u for c in corpora if c is not None for u in c)
                ),
            ),
        )

    def any_hit(self, spacer: str, max_mismatches: int) -> bool:
        """Existence-only query, short-circuiting across members.

        Offered because every member offers it; ``g_output`` uses it to verify
        a non-targeting control, and falling back to full enumeration there
        would pay the cost the seam exists to avoid.
        """
        for index in self.indexes:
            probe = getattr(index, "any_hit", None)
            if probe is not None:
                if probe(spacer, max_mismatches):
                    return True
            elif index.search("__any_hit__", spacer, max_mismatches).hits:
                return True
        return False
