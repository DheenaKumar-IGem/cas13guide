"""Reference and transcript graph: the sequence universe that gets searched."""

from __future__ import annotations

import hashlib
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
        """What the caller *asserted* this world is.

        Built from ``--genome-build``/``--annotation``, which default to
        GRCh38/v44 and are never checked against the sequences loaded. It is a
        label, not a measurement -- see :attr:`digest` for the identity that is
        one.
        """
        return f"{self.build}/{self.annotation}"

    @property
    def digest(self) -> str:
        """Content hash of the sequences this world actually contains.

        ``version`` could not tell two off-target worlds apart. It is a
        caller-asserted string that is constant by default, so the only thing
        distinguishing two ``--background`` corpora in the ledger was the locus
        count -- and two corpora with the same number of records produced
        byte-identical provenance while a guide called specific against one was
        never searched against the other. "Specific against what you gave it"
        is the tool's central caveat, and the record of *what you gave it* was
        a number.

        Hashes what the answers depend on: which loci, their sequences, their
        biotypes (hazard weights are per-biotype) and their abundances (the
        hazard product reads expression). Sorted by locus id, so load order
        does not change the digest and two identical corpora still agree.
        """
        payload = "\n".join(
            f"{locus.locus_id}\t{locus.biotype.value}\t{locus.expression}\t"
            f"{locus.sequence}"
            for locus in sorted(self.loci.values(), key=lambda lo: lo.locus_id)
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

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

    def in_universes(self, universes: frozenset[Universe]):
        return [lo for lo in self.loci.values() if lo.biotype in universes]

    def __len__(self) -> int:
        return len(self.loci)
