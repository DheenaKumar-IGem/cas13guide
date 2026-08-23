"""Sequence input and export (FR1.1, FR1.2, FR1.4, FR8.3).

The front door of the pipeline, and the place NFR3 is either satisfied or not:
malformed input has to produce a message that says *what* is wrong and *where*,
not a stack trace and not a silently mangled sequence.

Two rules this module holds to
------------------------------
**Say where the problem is.** ``util.seq.normalize`` raises
``non-RNA characters in sequence: ['X']``, which is true and nearly useless
when the user pasted a 900 nt multi-record file. Every error here names the
record and the 1-based offset of the first offending character.

**Never quietly repair.** ``T`` is transcribed to ``U`` because DNA input is an
explicitly supported format (FR1.1), and whitespace and FASTA line wrapping are
removed because they are formatting rather than content. Anything else --
ambiguity codes beyond ``N``, protein letters, digits from a pasted alignment --
is refused. A guide designed against a sequence the tool silently altered is
worse than no guide, because nothing downstream can tell.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..util.seq import VALID_RNA

#: Characters stripped as formatting rather than treated as content.
_IGNORED = frozenset(" \t\r\n\v\f-*")


class SequenceInputError(ValueError):
    """Malformed input, with a message a non-bioinformatician can act on."""


@dataclass(frozen=True, slots=True)
class SequenceRecord:
    """One named sequence, already normalized to the RNA alphabet."""

    name: str
    sequence: str
    description: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)


def _clean(raw: str, record_name: str) -> str:
    """Uppercase, drop formatting, transcribe DNA, and refuse anything else."""
    out: list[str] = []
    for index, character in enumerate(raw, start=1):
        if character in _IGNORED:
            continue
        upper = character.upper()
        if upper == "T":
            upper = "U"
        if upper not in VALID_RNA:
            if character == ">":
                # Far and away the most common paste error: a FASTA file with
                # something above the first header, or headers separated by
                # stray text. Saying "unexpected character '>'" is technically
                # true and tells the user nothing about what to fix.
                raise SequenceInputError(
                    f"{record_name}: found a '>' at position {index}, which "
                    "looks like a FASTA header in the middle of sequence data. "
                    "If this is a FASTA file, make sure it starts with a '>' "
                    "header line and that nothing precedes it."
                )
            raise SequenceInputError(
                f"{record_name}: unexpected character {character!r} at position "
                f"{index}. Sequences may contain A, C, G, U (or T, which is "
                "transcribed to U) and N for an unknown base."
            )
        out.append(upper)
    if not out:
        raise SequenceInputError(f"{record_name}: no sequence characters found.")
    return "".join(out)


def parse_fasta(text: str) -> list[SequenceRecord]:
    """Parse FASTA text into records (FR1.2).

    Duplicate record names are kept rather than merged or renamed: they are
    usually a sign the user concatenated files, and silently deduplicating
    would hide that.
    """
    records: list[SequenceRecord] = []
    name: str | None = None
    description = ""
    chunks: list[str] = []

    def flush() -> None:
        if name is None:
            return
        records.append(
            SequenceRecord(
                name=name,
                sequence=_clean("".join(chunks), f"record {name!r}"),
                description=description,
            )
        )

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            flush()
            header = stripped[1:].strip()
            if not header:
                raise SequenceInputError(
                    f"FASTA record {len(records) + 1} has an empty header line "
                    '(">" with no name).'
                )
            name, _, description = header.partition(" ")
            chunks = []
        else:
            if name is None:
                raise SequenceInputError(
                    "FASTA input must start with a '>' header line; found "
                    f"sequence data first ({stripped[:30]!r}...)."
                )
            chunks.append(stripped)
    flush()

    if not records:
        raise SequenceInputError("no FASTA records found in the input.")
    return records


def parse_sequence_input(text: str, default_name: str = "query") -> list[SequenceRecord]:
    """Accept either raw sequence text or FASTA (FR1.1).

    Which one it is is decided by whether the first non-blank line starts with
    ``>``, not by guessing from the alphabet -- a raw sequence and a FASTA body
    are the same characters, so any content-based sniff would be a coin flip on
    some input.
    """
    if not text or not text.strip():
        raise SequenceInputError(
            "no sequence supplied. Paste a sequence, or a FASTA record starting "
            'with ">".'
        )
    for line in text.splitlines():
        if line.strip():
            first = line.strip()
            break
    if first.startswith(">"):
        return parse_fasta(text)
    return [SequenceRecord(name=default_name, sequence=_clean(text, "input"))]


def read_fasta(path: str | Path) -> list[SequenceRecord]:
    """Read a FASTA file (FR1.2)."""
    path = Path(path)
    if not path.exists():
        raise SequenceInputError(f"no such file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise SequenceInputError(
            f"{path.name} is not readable as text -- is it a compressed or "
            "binary file?"
        ) from exc
    return parse_fasta(text)


def write_fasta(
    path: str | Path, records: Iterable[SequenceRecord], line_width: int = 60
) -> None:
    """Write records as wrapped FASTA."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(format_fasta(records, line_width=line_width))


def format_fasta(records: Iterable[SequenceRecord], line_width: int = 60) -> str:
    out = io.StringIO()
    for record in records:
        header = record.name if not record.description else f"{record.name} {record.description}"
        out.write(f">{header}\n")
        sequence = record.sequence
        if line_width <= 0:
            out.write(sequence + "\n")
            continue
        for start in range(0, len(sequence), line_width):
            out.write(sequence[start : start + line_width] + "\n")
    return out.getvalue()


# ---------------------------------------------------------------------------
# Export (FR8.3)
# ---------------------------------------------------------------------------

#: Columns of the ranked-candidate export, in order. Kept explicit rather than
#: derived from whatever keys a report happens to carry, so adding a field
#: upstream cannot silently reshape a file someone's script parses.
RANKED_COLUMNS: tuple[str, ...] = (
    "rank",
    "guide_id",
    "spacer",
    "score",
    "verdict",
    "review_tier",
    "spans_junction",
    "pfs_3prime",
    "pfs_preferred",
    "accessibility",
    "duplex_dg_per_nt",
    "predicted_signal",
    "reasons",
    "qc_warnings",
)


def ranked_rows(report_dict: dict) -> list[dict]:
    """Flatten a :meth:`DesignReport.to_dict` ranking into export rows."""
    rows = []
    for entry in report_dict.get("ranked", []):
        row = {column: entry.get(column) for column in RANKED_COLUMNS}
        metrics = entry.get("metrics") or {}
        row["required_coverage"] = metrics.get("required_coverage")
        row["worst_off_target"] = metrics.get("worst_off_target")
        row["cumulative_burden"] = metrics.get("cumulative_burden")
        for key in ("reasons", "qc_warnings"):
            value = row.get(key)
            if isinstance(value, (list, tuple)):
                row[key] = ";".join(str(item) for item in value)
        rows.append(row)
    return rows


def ranked_to_csv(report_dict: dict) -> str:
    """Ranked candidates as CSV text (FR8.3)."""
    rows = ranked_rows(report_dict)
    columns = list(RANKED_COLUMNS) + [
        "required_coverage",
        "worst_off_target",
        "cumulative_burden",
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return out.getvalue()


def ranked_to_fasta(report_dict: dict, limit: int | None = None) -> str:
    """Ranked spacers as FASTA (FR8.3).

    The description carries the numbers a wet-lab notebook needs beside the
    sequence, so the file is useful on its own rather than only as a companion
    to the CSV.
    """
    records = []
    for entry in report_dict.get("ranked", [])[:limit]:
        parts = [f"rank={entry.get('rank')}", f"score={entry.get('score')}"]
        for key in ("accessibility", "duplex_dg_per_nt", "predicted_signal"):
            value = entry.get(key)
            if value is not None:
                parts.append(f"{key}={value}")
        parts.append(f"verdict={entry.get('verdict')}")
        records.append(
            SequenceRecord(
                name=str(entry.get("guide_id")),
                sequence=str(entry.get("spacer")),
                description=" ".join(parts),
            )
        )
    return format_fasta(records)


def controls_to_fasta(report_dict: dict) -> str:
    """Non-targeting controls as FASTA, so they can be ordered alongside."""
    records = [
        SequenceRecord(
            name=str(control.get("guide_id")),
            sequence=str(control.get("spacer")),
            description=(
                f"non_targeting matched_to={control.get('matched_to')} "
                f"gc={control.get('gc')} verified={control.get('verified_no_hits')}"
            ),
        )
        for control in report_dict.get("controls", [])
        if control.get("spacer")
    ]
    return format_fasta(records)


def write_exports(
    report_dict: dict, directory: str | Path, stem: str = "candidates"
) -> dict[str, Path]:
    """Write the CSV and FASTA exports side by side. Returns the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for label, suffix, text in (
        ("csv", ".csv", ranked_to_csv(report_dict)),
        ("fasta", ".fasta", ranked_to_fasta(report_dict)),
        ("controls", "_controls.fasta", controls_to_fasta(report_dict)),
    ):
        path = directory / f"{stem}{suffix}"
        path.write_text(text, encoding="utf-8", newline="\n")
        written[label] = path
    return written


def records_to_target_specs(
    records: Sequence[SequenceRecord], target_class=None
) -> Iterator:
    """Turn parsed records into contract targets.

    Imported lazily to keep this module usable for plain sequence handling
    without pulling in pydantic and the whole contract layer.
    """
    from ..contract.enums import TargetClass
    from ..contract.spec import TargetSpec

    chosen = target_class or TargetClass.REQUIRED
    for record in records:
        yield TargetSpec(
            target_id=record.name,
            sequence=record.sequence,
            target_class=chosen,
            gene=record.name,
        )
