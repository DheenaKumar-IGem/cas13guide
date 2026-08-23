"""Off-target search against a BLAST database (FR4.1, FR4.2, FR4.3).

Plugs into the same :class:`~cas13guide.services.indexes.SearchIndex` seam as
the built-in indexes, so nothing downstream changes -- and reports
``exhaustive=False``, which the gate turns into an R2 REVIEW.

Why BLAST is the *heuristic* index here, not the reference one
--------------------------------------------------------------
The requirement asks for BLAST, and there are good reasons to want it: it is
what reviewers recognise, and it searches a curated RefSeq database this package
does not ship. But it is not a drop-in replacement for the exhaustive scan, and
pretending otherwise would be the one failure this architecture exists to
prevent -- a clean off-target report that is clean because the search missed
things.

Three specific gaps, all of which this module is explicit about:

* **Seeded, so it can miss.** BLAST finds alignments by extending exact word
  matches. A site whose differences are spread so that no exact word of
  ``word_size`` survives is simply not found. ``-task blastn-short`` with a
  small word size narrows the gap; it does not close it.
* **Not wobble-aware.** BLAST works in DNA space and scores G:U as a mismatch,
  because for DNA it is one. For a Cas13 RNA:RNA duplex a G-U wobble is a real
  base pair (see :mod:`cas13guide.util.duplex`), so BLAST both *under-seeds*
  wobble-rich sites and *over-counts* their mismatches. The mismatch count
  reported here is therefore recomputed with :func:`profile_duplex` from the
  aligned subject sequence, never taken from BLAST's own column.
* **Only as complete as its database.** ``ExhaustivenessBound.universes`` records
  what the caller declared the database covers -- an assertion about the
  database, not something this module can verify. If that declaration is wrong,
  the universe-adequacy gate is reading a claim rather than a fact.

Use it to search a database too large to scan, and read the R2 for what it is.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field

from ..contract.enums import Universe
from ..models.evidence import SearchHit
from ..util.duplex import profile_duplex
from ..util.seq import normalize, revcomp
from .indexes import ExhaustivenessBound, SearchResult

#: Fields requested from ``-outfmt 6``, in order. ``sseq`` is what lets the
#: duplex be re-scored under the wobble rule instead of trusting BLAST's count.
OUTFMT_FIELDS: tuple[str, ...] = (
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore", "sstrand", "sseq",
)


class BlastUnavailable(RuntimeError):
    """The BLAST+ executable or the database is missing (DR1, DR3)."""


@dataclass(frozen=True, slots=True)
class BlastHitRow:
    """One parsed ``-outfmt 6`` row."""

    query: str
    subject: str
    length: int
    blast_mismatches: int
    gap_opens: int
    subject_start: int
    subject_end: int
    strand: str
    subject_sequence: str

    @property
    def is_full_length(self) -> bool:
        return self.gap_opens == 0

    @property
    def zero_based_start(self) -> int:
        """0-based start on the subject, in transcript orientation."""
        return min(self.subject_start, self.subject_end) - 1


def parse_outfmt6(text: str) -> list[BlastHitRow]:
    """Parse tab-separated BLAST output produced with :data:`OUTFMT_FIELDS`."""
    rows: list[BlastHitRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != len(OUTFMT_FIELDS):
            raise ValueError(
                f"expected {len(OUTFMT_FIELDS)} BLAST columns, got {len(parts)}: "
                f"{line[:80]!r}. The caller must request exactly "
                f"'6 {' '.join(OUTFMT_FIELDS)}'."
            )
        record = dict(zip(OUTFMT_FIELDS, parts))
        rows.append(
            BlastHitRow(
                query=record["qseqid"],
                subject=record["sseqid"],
                length=int(record["length"]),
                blast_mismatches=int(record["mismatch"]),
                gap_opens=int(record["gapopen"]),
                subject_start=int(record["sstart"]),
                subject_end=int(record["send"]),
                strand=record["sstrand"],
                subject_sequence=record["sseq"].replace("-", ""),
            )
        )
    return rows


@dataclass(slots=True)
class BlastOffTargetIndex:
    """A ``SearchIndex`` backed by ``blastn`` against a transcript database.

    ``universes`` is what the *caller asserts* the database covers. This module
    cannot verify that, and the universe-adequacy gate will be reading that
    assertion -- so declare it honestly or the gate is checking nothing.
    """

    database: str
    universes: frozenset[Universe]
    #: Biotype assigned to hits. A single value because a BLAST database has no
    #: universe column; split databases and separate indexes if that matters.
    biotype: Universe = Universe.CODING
    executable: str = "blastn"
    #: Short queries need a small word size and a permissive E-value, or almost
    #: nothing is reported: a 22-mer simply cannot reach a conventional E-value.
    word_size: int = 7
    evalue: float = 1000.0
    max_target_seqs: int = 500
    threads: int = 1
    timeout_seconds: float = 300.0
    version: str = "index/blast-short/v1"
    _checked: bool = field(default=False, repr=False, compare=False)

    def ensure_available(self) -> None:
        """Fail early and legibly rather than deep inside a design run."""
        if self._checked:
            return
        if shutil.which(self.executable) is None:
            raise BlastUnavailable(
                f"{self.executable!r} is not on PATH. Install the BLAST+ toolkit "
                "(DR3), or use ExhaustiveScanIndex / FMIndexSearchIndex against a "
                "local transcriptome instead."
            )
        self._checked = True

    def command(self, query_fasta: str) -> list[str]:
        """The exact argv used. Exposed so a run can be reproduced by hand."""
        return [
            self.executable,
            "-task", "blastn-short",
            "-db", self.database,
            "-query", query_fasta,
            "-outfmt", "6 " + " ".join(OUTFMT_FIELDS),
            "-word_size", str(self.word_size),
            "-evalue", str(self.evalue),
            "-max_target_seqs", str(self.max_target_seqs),
            "-num_threads", str(self.threads),
            "-strand", "plus",
            "-dust", "no",
            "-soft_masking", "false",
        ]

    def _run(self, query_dna: str) -> str:
        self.ensure_available()
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            query_path = Path(directory) / "query.fa"
            query_path.write_text(f">query\n{query_dna}\n", encoding="utf-8")
            try:
                completed = subprocess.run(
                    self.command(str(query_path)),
                    capture_output=True, text=True, timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise BlastUnavailable(
                    f"blastn timed out after {self.timeout_seconds:g}s against "
                    f"{self.database!r}"
                ) from exc
        if completed.returncode != 0:
            raise BlastUnavailable(
                f"blastn failed (exit {completed.returncode}) against "
                f"{self.database!r}: {completed.stderr.strip()[:300]}"
            )
        return completed.stdout

    def bound(self, max_mismatches: int) -> ExhaustivenessBound:
        return ExhaustivenessBound(
            max_mismatches=max_mismatches,
            # It is not wobble-aware, in either direction: it under-seeds
            # wobble-rich sites and over-counts their mismatches.
            wobble_aware=False,
            universes=self.universes,
            exhaustive=False,
            method=self.version,
            note=(
                f"blastn -task blastn-short, word_size={self.word_size}, "
                f"db={self.database!r}. Seeded search: a site whose differences "
                "leave no exact word intact is not found, and G-U wobbles are "
                "scored as mismatches when seeding. Hit mismatch counts are "
                "re-derived under the wobble rule, but recall is not guaranteed."
            ),
        )

    def search(self, guide_id: str, spacer: str, max_mismatches: int) -> SearchResult:
        """Search the database for sites this spacer could bind (FR4.1, FR4.2).

        The *target site* is what gets searched -- ``revcomp(spacer)`` -- on the
        plus strand only. A transcript database holds sense strands, so a
        minus-strand alignment does not correspond to a site the guide can bind
        in the cell; counting one would invent an off-target.
        """
        spacer = normalize(spacer)
        expected = revcomp(spacer)
        # BLAST is a DNA tool; transcribe out and back.
        raw = self._run(expected.replace("U", "T"))

        hits: list[SearchHit] = []
        for row in parse_outfmt6(raw):
            if row.strand not in ("plus", "plus/plus", "1"):
                continue
            if not row.is_full_length or row.length != len(expected):
                # A partial alignment is not a binding site for a fixed-length
                # spacer, and padding one out would be inventing sequence.
                continue
            site = normalize(row.subject_sequence)
            if len(site) != len(spacer):
                continue
            duplex = profile_duplex(spacer, site)
            # Re-derived under the wobble rule, never BLAST's own count.
            if duplex.n_mismatch > max_mismatches:
                continue
            hits.append(
                SearchHit(
                    guide_id=guide_id,
                    locus_id=row.subject,
                    target_site=site,
                    start=row.zero_based_start,
                    duplex=duplex,
                    biotype=self.biotype,
                    gene=row.subject,
                    transcript=row.subject,
                )
            )
        hits.sort(key=lambda hit: (hit.locus_id, hit.start))
        return SearchResult(hits=tuple(hits), bound=self.bound(max_mismatches))
