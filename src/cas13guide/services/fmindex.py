"""FM-index with wobble-aware backtracking search.

This is the production drop-in behind the :class:`SearchIndex` seam. It returns
exactly the same hit set as :class:`ExhaustiveScanIndex` -- there is a
differential test asserting that -- but it prunes with a BWT-backed index
instead of sliding a window over every locus, so it scales to a real
transcriptome.

Why backtracking rather than pigeonhole seeding
-----------------------------------------------
The usual fast trick is the pigeonhole principle: split the pattern into
``k+1`` parts, and any occurrence with ``<=k`` mismatches must match at least
one part *exactly*. That filter is complete for plain Hamming distance -- and
it silently breaks here.

Cas13 binds RNA:RNA, so G-U wobble pairs still pair and cost nothing against
the mismatch budget (see :mod:`cas13guide.util.duplex`). A part containing only
wobbles is a biological match but not a *literal* one, so it fails exact
lookup. The number of wobbles is unbounded, so no fixed partition count
restores the guarantee, and the filter would drop real hits while still
reporting itself as exhaustive -- the precise failure mode this architecture
exists to prevent.

Backtracking has no such blind spot: at each character it branches over the
text alphabet and charges cost 0 for a Watson-Crick match, 0 for a wobble, and
1 otherwise. Every string within budget is enumerated, so the index can
honestly claim ``exhaustive=True``.

When this index is actually the right choice
--------------------------------------------
It is not always. Backtracking cost is governed by the size of the mismatch
neighbourhood -- roughly ``C(L, k) * 3^k`` strings, about 590k for a 22-mer at
``k=4`` -- and is therefore close to *independent of corpus size* once the
corpus is large enough to keep branches alive. The reference
:class:`~cas13guide.services.indexes.ExhaustiveScanIndex` is ``O(N*L)``: linear
in the corpus, with a very small constant.

Measured on random RNA in 2 kb loci, 22-mers, ``k=4``, per query:

===========  ==========  ==========  =========
corpus       FM          scan        speedup
===========  ==========  ==========  =========
600 kb       0.742 s     0.018 s     0.02x
1.5 Mb       2.990 s     0.054 s     0.02x
3 Mb         2.758 s     0.089 s     0.03x
===========  ==========  ==========  =========

The scanner is *far* ahead at these sizes. It used to be much closer -- the
crossover sat near ~2.5 Mb -- but vectorizing the scan made it ~32x faster on
the same machine for the same hit set, which pushed the crossover out to
~87 Mb of searchable sequence. Below that the scanner is simply the right
answer; above it the FM-index pulls away and keeps pulling away, which is the
regime a full transcript set (a few hundred megabases) lives in. Both return
identical hit sets, so this is purely a speed decision --
:func:`cas13guide.services.index_selection.choose_index` makes it, and it now
picks the scanner for anything short of transcriptome scale.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from ..contract.enums import Universe
from ..models.evidence import SearchHit
from ..util.duplex import WOBBLE_SUBSTITUTIONS, profile_duplex
from ..util.seq import revcomp
from .indexes import ExhaustivenessBound, SearchResult
from .suffixarray import build_suffix_array
from .transcriptome import Transcriptome

#: Sorts before every real base; marks the end of the concatenated text.
TERMINATOR = "$"
#: Sits between loci so no match can span two transcripts.
SEPARATOR = "#"

#: The four standard bases. Retained as the documented floor of the walk
#: alphabet; the alphabet actually walked is derived per index by
#: :func:`walk_alphabet` and may be wider.
BASES = ("A", "C", "G", "U")

#: Characters the backtracking walk must never branch over. The separator's
#: exclusion is the FIRST of two independent defenses against a match spanning
#: two loci; the second is the span check in ``_locate``. The redundancy is
#: intentional -- a cross-locus hit is an invented off-target, and it would be
#: reported as a real one.
NON_BASE_CHARACTERS = frozenset({SEPARATOR, TERMINATOR})


def walk_alphabet(text: str) -> tuple[str, ...]:
    """Characters the walk may branch over, derived from the indexed text.

    Hard-coding ``BASES`` here was a silent exhaustiveness hole. ``N`` is a
    *legal* base -- ``util.seq.VALID_RNA`` admits it, ``RNA_COMPLEMENT`` maps
    it to itself, and ``Transcriptome.add`` normalizes it through unchanged --
    so a real assembly gap inside an off-target site made every window
    overlapping it unreachable by the walk. The index returned a strictly
    smaller hit set than the reference scanner while still reporting
    ``exhaustive=True``: a 1-mismatch paralog hit simply disappeared and the
    guide read as perfectly specific. That is the precise failure this module's
    own docstring says backtracking was chosen to avoid.

    Deriving the alphabet from the text also keeps the cost honest: an
    ``AUGC``-only corpus branches over exactly four characters as before, and
    only a corpus that really contains ``N`` pays for the fifth.
    """
    return tuple(sorted(set(text) - NON_BASE_CHARACTERS))


@dataclass(frozen=True, slots=True)
class _LocusSpan:
    start: int
    length: int
    locus_id: str


class FMIndex:
    """Burrows-Wheeler index supporting backward extension.

    ``occ`` is checkpointed every ``checkpoint`` positions, trading memory for
    rank() speed in the usual way.
    """

    def __init__(self, text: str, checkpoint: int = 512) -> None:
        self.text = text
        self.n = len(text)
        self.checkpoint = checkpoint
        # Linear-time SA-IS; see services/suffixarray.py.
        self.sa = build_suffix_array(text)

        self.bwt = "".join(text[i - 1] if i > 0 else text[-1] for i in self.sa)

        alphabet = sorted(set(text))
        counts = {c: 0 for c in alphabet}
        for c in text:
            counts[c] += 1
        total = 0
        self.c_table: dict[str, int] = {}
        for c in alphabet:
            self.c_table[c] = total
            total += counts[c]

        n_blocks = self.n // checkpoint + 1
        self._occ: dict[str, list[int]] = {c: [0] * n_blocks for c in alphabet}
        running = {c: 0 for c in alphabet}
        for i, c in enumerate(self.bwt):
            if i % checkpoint == 0:
                for base, table in self._occ.items():
                    table[i // checkpoint] = running[base]
            running[c] += 1
        for base, table in self._occ.items():
            for block in range(self.n // checkpoint + 1):
                if block * checkpoint >= self.n:
                    table[block] = running[base]

    def rank(self, c: str, i: int) -> int:
        """Occurrences of ``c`` in ``bwt[:i]``.

        The tail scan uses ``str.count``, which runs in C. A Python-level loop
        here costs roughly an order of magnitude: backward search calls rank
        twice per branch, and the backtracking walk takes hundreds of
        thousands of branches, so this is the hot path of the whole index.
        """
        if i <= 0:
            return 0
        table = self._occ.get(c)
        if table is None:
            return 0
        if i >= self.n:
            i = self.n
        block, offset = divmod(i, self.checkpoint)
        count = table[block]
        if offset:
            start = block * self.checkpoint
            count += self.bwt.count(c, start, i)
        return count

    def extend_backward(self, c: str, lo: int, hi: int) -> tuple[int, int]:
        """Narrow the SA interval ``[lo, hi)`` by prepending character ``c``."""
        if c not in self.c_table:
            return (0, 0)
        base = self.c_table[c]
        return (base + self.rank(c, lo), base + self.rank(c, hi))

    def full_interval(self) -> tuple[int, int]:
        return (0, self.n)


class FMIndexSearchIndex:
    """``SearchIndex`` implementation backed by :class:`FMIndex`.

    Provably exhaustive to the requested mismatch depth over the declared
    universes, wobble-aware, and honest about it in the returned bound.
    """

    version = "index/fm-backtracking/v1"

    def __init__(
        self,
        transcriptome: Transcriptome,
        universes: frozenset[Universe],
        checkpoint: int = 512,
    ) -> None:
        self.transcriptome = transcriptome
        self.universes = universes

        loci = transcriptome.in_universes(universes)
        # Deterministic order so the index is reproducible across runs.
        loci.sort(key=lambda locus: locus.locus_id)

        pieces: list[str] = []
        spans: list[_LocusSpan] = []
        cursor = 0
        for locus in loci:
            spans.append(_LocusSpan(cursor, len(locus.sequence), locus.locus_id))
            pieces.append(locus.sequence)
            pieces.append(SEPARATOR)
            cursor += len(locus.sequence) + 1

        self._spans = spans
        self._span_starts = [s.start for s in spans]
        self._by_id = {locus.locus_id: locus for locus in loci}
        self.text = "".join(pieces) + TERMINATOR
        # Derived from the text, not hard-coded: see walk_alphabet.
        self.walk_alphabet = walk_alphabet(self.text)
        self.fm = FMIndex(self.text, checkpoint=checkpoint) if loci else None

    # -- position mapping ------------------------------------------------
    def _locate(self, global_start: int, length: int) -> tuple[str, int] | None:
        """Map a text offset to ``(locus_id, local_start)``, or None if it
        straddles a locus boundary."""
        idx = bisect.bisect_right(self._span_starts, global_start) - 1
        if idx < 0:
            return None
        span = self._spans[idx]
        local = global_start - span.start
        # Second boundary defense (see BASES): reject anything that would run
        # off the end of its locus, whatever the walk produced.
        if local < 0 or local + length > span.length:
            return None
        return span.locus_id, local

    # -- search ----------------------------------------------------------
    def search(self, guide_id: str, spacer: str, max_mismatches: int) -> SearchResult:
        bound = ExhaustivenessBound(
            max_mismatches=max_mismatches,
            wobble_aware=True,
            universes=self.universes,
            exhaustive=True,
            method=self.version,
            note="BWT backtracking; every string within the budget is enumerated",
        )
        if self.fm is None:
            return SearchResult(hits=(), bound=bound)

        pattern = revcomp(spacer)  # the perfect target site
        length = len(pattern)
        starts: list[int] = []

        def walk(pos: int, lo: int, hi: int, cost: int) -> None:
            if lo >= hi:
                return
            if pos < 0:
                starts.extend(self.fm.sa[lo:hi])
                return
            expected = pattern[pos]
            for c in self.walk_alphabet:
                if c == expected or (expected, c) in WOBBLE_SUBSTITUTIONS:
                    delta = 0
                else:
                    delta = 1
                if cost + delta > max_mismatches:
                    continue
                next_lo, next_hi = self.fm.extend_backward(c, lo, hi)
                if next_lo < next_hi:
                    walk(pos - 1, next_lo, next_hi, cost + delta)

        lo, hi = self.fm.full_interval()
        walk(length - 1, lo, hi, 0)

        hits: list[SearchHit] = []
        for global_start in starts:
            located = self._locate(global_start, length)
            if located is None:
                continue  # would straddle a locus boundary
            locus_id, local_start = located
            locus = self._by_id[locus_id]
            window = locus.sequence[local_start : local_start + length]
            end = local_start + length
            hits.append(
                SearchHit(
                    guide_id=guide_id,
                    locus_id=locus_id,
                    target_site=window,
                    start=local_start,
                    duplex=profile_duplex(spacer, window),
                    biotype=locus.biotype,
                    gene=locus.gene,
                    transcript=locus.transcript,
                    five_prime_flank=locus.sequence[local_start - 1] if local_start > 0 else None,
                    three_prime_flank=(
                        locus.sequence[end] if end < len(locus.sequence) else None
                    ),
                )
            )

        hits.sort(key=lambda h: (h.locus_id, h.start))
        return SearchResult(hits=tuple(hits), bound=bound)

    def any_hit(self, spacer: str, max_mismatches: int) -> bool:
        """Existence-only query; see ``ExhaustiveScanIndex.any_hit``.

        Still walks the backtracking tree, but stops at the first located hit
        and skips ``profile_duplex`` and ``SearchHit`` construction entirely.
        """
        return bool(self.search("__any_hit__", spacer, max_mismatches).hits)
