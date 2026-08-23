"""Reference and transcript graph: the sequence universe that gets searched."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..contract.enums import Universe
from ..util.seq import normalize


@dataclass(frozen=True, slots=True)
class Locus:
    """One searchable RNA sequence."""

    locus_id: str
    sequence: str
    biotype: Universe
    gene: str | None = None
    transcript: str | None = None
    #: TPM-like abundance. ``None`` means unmeasured -- an evidence gap.
    expression: float | None = None
    #: 0-based offsets of exon-exon junctions within ``sequence``.
    junctions: tuple[int, ...] = ()


@dataclass(slots=True)
class Transcriptome:
    """A versioned, universe-tagged collection of loci."""

    build: str
    annotation: str
    loci: dict[str, Locus] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return f"{self.build}/{self.annotation}"

    def add(self, locus: Locus) -> None:
        seq = normalize(locus.sequence)
        self.loci[locus.locus_id] = Locus(
            locus_id=locus.locus_id,
            sequence=seq,
            biotype=locus.biotype,
            gene=locus.gene,
            transcript=locus.transcript,
            expression=locus.expression,
            junctions=locus.junctions,
        )

    def covered_universes(self) -> frozenset[Universe]:
        return frozenset(locus.biotype for locus in self.loci.values())

    def in_universes(self, universes: frozenset[Universe]):
        return [lo for lo in self.loci.values() if lo.biotype in universes]

    def __len__(self) -> int:
        return len(self.loci)
