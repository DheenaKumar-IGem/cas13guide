"""Conserved regions across related sequences (FR7.1, FR7.2).

Given a set of isoforms, strain variants or paralogs, find the windows where a
*single* guide can work against all of them, and say which guides lose a
variant.

Conservation here is measured in duplex space, not sequence space
--------------------------------------------------------------------
The obvious implementation compares sequences for identity. That is the wrong
question. What matters is not whether the variants are the same letter at a
position but whether one spacer still **pairs** with all of them, and Cas13
binds RNA:RNA, so a G-U wobble is a real base pair at a position where the two
variants differ. Scoring identity would discard guides that work perfectly well
across every variant, and would do it silently.

So a window's conservation is the worst *duplex* mismatch count its spacer
incurs against any variant -- computed by the same wobble-aware, provably
exhaustive index the off-target search uses, rather than by a second alignment
routine that could drift from it.

Relationship to the robustness metric
-------------------------------------
This module answers FR7.1: *where* can a shared guide exist. FR7.2 -- flagging
a guide that loses activity against one variant -- is already the pipeline's
``robustness`` metric, which is the minimum required coverage across evaluated
contexts. :func:`variant_contexts` builds one context per variant so that metric
does its job; this module is the cheap pre-analysis that tells you whether to
bother, and where to tile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..contract.enums import Universe
from ..util.seq import revcomp, windows
from .fasta import SequenceRecord
from .indexes import ExhaustiveScanIndex
from .transcriptome import Locus, Transcriptome

#: Universe the synthetic variant loci are filed under. Arbitrary but constant:
#: these loci exist only to be searched, never to be reported as off-targets.
_VARIANT_UNIVERSE = Universe.SYNTHETIC


@dataclass(frozen=True, slots=True)
class ConservedWindow:
    """One window of the reference and how it fares against every variant."""

    #: 0-based start in the reference sequence.
    start: int
    #: The target site on the reference.
    target_site: str
    #: The spacer that would be designed against it.
    spacer: str
    #: Variant name -> fewest duplex mismatches achievable against that variant.
    #: A variant absent from the map has no site within the search envelope.
    mismatches_by_variant: dict[str, int]
    #: Names of variants with no site at all within the envelope.
    missing_variants: tuple[str, ...]

    @property
    def worst_mismatches(self) -> int | None:
        """Mismatches against the variant this window fits worst.

        ``None`` when some variant has no site at all -- which is worse than any
        finite count, and must not be flattened into one.
        """
        if self.missing_variants:
            return None
        return max(self.mismatches_by_variant.values(), default=0)

    @property
    def fully_conserved(self) -> bool:
        """A perfect duplex against every variant."""
        return not self.missing_variants and self.worst_mismatches == 0

    def covers_all(self, max_mismatches: int) -> bool:
        worst = self.worst_mismatches
        return worst is not None and worst <= max_mismatches


def find_conserved_windows(
    records: Sequence[SequenceRecord],
    spacer_length: int,
    max_mismatches: int = 0,
    step: int = 1,
    reference: str | None = None,
) -> list[ConservedWindow]:
    """Windows of the reference whose spacer pairs with every variant (FR7.1).

    The first record is the reference unless ``reference`` names another. One
    index is built over all the variants and queried once per window, rather
    than aligning every pair: the index is wobble-aware and exhaustive, which an
    ad-hoc comparison would have to re-derive.
    """
    if len(records) < 2:
        raise ValueError(
            "conserved-region analysis needs at least two sequences; "
            f"got {len(records)}"
        )
    by_name = {record.name: record for record in records}
    if reference is not None and reference not in by_name:
        raise KeyError(
            f"reference {reference!r} is not among the supplied sequences: "
            f"{', '.join(by_name)}"
        )
    anchor = by_name[reference] if reference else records[0]
    variants = [record for record in records if record is not anchor]
    if not variants:
        raise ValueError("every record is the reference; supply distinct sequences")

    world = Transcriptome("variants", "v1")
    for variant in variants:
        world.add(Locus(variant.name, variant.sequence, _VARIANT_UNIVERSE, expression=1.0))
    index = ExhaustiveScanIndex(world, frozenset({_VARIANT_UNIVERSE}))

    variant_names = [variant.name for variant in variants]
    out: list[ConservedWindow] = []
    for start, site in windows(anchor.sequence, spacer_length, step):
        spacer = revcomp(site)
        best: dict[str, int] = {}
        for hit in index.search("conservation", spacer, max_mismatches).hits:
            count = hit.duplex.n_mismatch
            if count < best.get(hit.locus_id, count + 1):
                best[hit.locus_id] = count
        missing = tuple(name for name in variant_names if name not in best)
        out.append(
            ConservedWindow(
                start=start,
                target_site=site,
                spacer=spacer,
                mismatches_by_variant=best,
                missing_variants=missing,
            )
        )
    return out


def best_conserved_windows(
    records: Sequence[SequenceRecord],
    spacer_length: int,
    max_mismatches: int = 0,
    step: int = 1,
    reference: str | None = None,
) -> list[ConservedWindow]:
    """Only the windows that reach every variant, worst-fit first.

    Sorted so the most conserved sites come first; ties keep reference order, so
    the result is deterministic.
    """
    found = find_conserved_windows(
        records, spacer_length, max_mismatches, step, reference
    )
    usable = [window for window in found if window.covers_all(max_mismatches)]
    usable.sort(key=lambda window: (window.worst_mismatches, window.start))
    return usable


@dataclass(frozen=True, slots=True)
class ConservationSummary:
    """What a variant set can support, before any design run."""

    reference: str
    variants: tuple[str, ...]
    spacer_length: int
    max_mismatches: int
    windows_examined: int
    windows_covering_all: int
    fully_conserved_windows: int
    #: Variants that no window reached -- usually a sign the set is not
    #: homologous at this spacer length, or that the envelope is too tight.
    unreachable_variants: tuple[str, ...]

    @property
    def designable(self) -> bool:
        return self.windows_covering_all > 0

    def summary(self) -> str:
        if not self.designable:
            detail = ""
            if self.unreachable_variants:
                detail = (
                    " No window reached: "
                    + ", ".join(self.unreachable_variants)
                    + "."
                )
            return (
                f"no shared guide is possible at {self.spacer_length} nt within "
                f"<={self.max_mismatches} mismatches across "
                f"{len(self.variants)} variant(s).{detail} Widen the envelope, "
                "shorten the spacer, or design one guide per variant."
            )
        return (
            f"{self.windows_covering_all}/{self.windows_examined} windows reach "
            f"every variant within <={self.max_mismatches} mismatches "
            f"({self.fully_conserved_windows} of them perfectly)."
        )


def summarize_conservation(
    records: Sequence[SequenceRecord],
    spacer_length: int,
    max_mismatches: int = 0,
    step: int = 1,
    reference: str | None = None,
) -> ConservationSummary:
    """A one-look verdict on whether a shared guide is even possible."""
    found = find_conserved_windows(
        records, spacer_length, max_mismatches, step, reference
    )
    anchor = (
        next(r for r in records if r.name == reference) if reference else records[0]
    )
    variants = tuple(r.name for r in records if r is not anchor)
    reached = {
        name
        for window in found
        for name in window.mismatches_by_variant
    }
    return ConservationSummary(
        reference=anchor.name,
        variants=variants,
        spacer_length=spacer_length,
        max_mismatches=max_mismatches,
        windows_examined=len(found),
        windows_covering_all=sum(1 for w in found if w.covers_all(max_mismatches)),
        fully_conserved_windows=sum(1 for w in found if w.fully_conserved),
        unreachable_variants=tuple(name for name in variants if name not in reached),
    )


def variant_contexts(
    records: Iterable[SequenceRecord],
    universes: frozenset[Universe],
    target_id: str,
    biotype: Universe = Universe.NCRNA,
    expression: float | None = None,
    expected_queries: int = 1000,
):
    """One :class:`DesignContext` per variant, for FR7.2.

    The pipeline's ``robustness`` metric is the minimum required coverage across
    contexts, so putting each variant in its own context is what makes "flag
    guides that lose activity against any one variant" fall out of machinery
    that already exists. Every context maps its locus to the *same* target id,
    which is what makes them alternative realisations of one target rather than
    several different targets.
    """
    from ..pipeline.orchestrator import DesignContext

    contexts = []
    for index, record in enumerate(records):
        world = Transcriptome("variants", record.name)
        world.add(
            Locus(record.name, record.sequence, biotype, gene=record.name,
                  expression=expression)
        )
        contexts.append(
            DesignContext.auto(
                "reference" if index == 0 else record.name,
                world,
                universes,
                {record.name: target_id},
                expected_queries=expected_queries,
            )
        )
    return contexts
