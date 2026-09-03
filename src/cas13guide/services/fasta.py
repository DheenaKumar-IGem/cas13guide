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
from typing import Iterable, Sequence

from ..util.seq import VALID_RNA

#: Line prefixes that mark a comment rather than sequence data.
_COMMENT_PREFIXES = frozenset(";#")

#: Characters stripped as formatting rather than treated as content: whitespace
#: and nothing else. ``-`` and ``*`` used to be in here, and deleting a gap is
#: not formatting -- it splices the base *before* the run onto the base
#: *after* it, and the tiler then walks a window across a junction that exists
#: in no molecule. Measured on a 400 nt target with a 30-base gapped run, 21 of
#: 297 exported guides had a target site occurring nowhere in the real
#: sequence. This module's own rule is that nothing is quietly repaired, and a
#: pasted alignment's ruler digits were refused with a located message while
#: the same alignment's gaps were removed without one.
_IGNORED = frozenset(" \t\r\n\v\f")

#: Refused with their own message rather than the generic one, because
#: "unexpected character '-'" tells someone who pasted an aligned row nothing
#: about what to do next.
_GAP_CHARACTERS = frozenset("-*")


#: Byte-order mark. Encoding, not content -- and :func:`read_fasta` consumes it
#: via ``utf-8-sig``, so the pasted and uploaded paths, which decode their own
#: bytes, have to consume it too. Otherwise the same file behaves differently
#: depending on which door it came through.
_BOM = "﻿"


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
            if character in _GAP_CHARACTERS:
                raise SequenceInputError(
                    f"{record_name}: found {character!r} at position {index}, "
                    "which looks like an alignment gap or a masked base. "
                    "Remove the gaps before designing, or write masked bases "
                    "as 'N' -- deleting them would splice the flanking bases "
                    "together and tile a window across a junction that is not "
                    "in the molecule."
                )
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
    text = text.lstrip(_BOM)
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
        if stripped[0] in _COMMENT_PREFIXES:
            # ';' is the original FASTA comment character and '#' is the
            # widespread convention. Both must be skipped rather than parsed as
            # sequence -- not least because this module writes a ';'-prefixed
            # provenance header itself, so treating them as data made the
            # exports unreadable by the parser that produced them.
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
    text = text.lstrip(_BOM) if text else text
    if not text or not text.strip():
        raise SequenceInputError(
            "no sequence supplied. Paste a sequence, or a FASTA record starting "
            'with ">".'
        )
    # Comment lines are skipped before the sniff *and* before the raw branch.
    # `parse_fasta` skips ';' and '#' -- this module writes a ';'-prefixed
    # provenance header itself -- and this function did not, so a FASTA whose
    # first line is a comment was handed to `_clean` whole and refused at
    # character one with "unexpected character ';' at position 1": a valid
    # FASTA file reported as a malformed raw sequence. It split the two doors,
    # because `--fasta` goes through `parse_fasta` and the textarea, `--sequence`
    # and the browser's file upload come here -- and the file it most reliably
    # rejected was this tool's own exported FASTA.
    body = [
        line for line in text.splitlines()
        if not (line.strip() and line.strip()[0] in _COMMENT_PREFIXES)
    ]
    first = next((line.strip() for line in body if line.strip()), "")
    if not first:
        raise SequenceInputError(
            "no sequence supplied: the input is only comment lines. Paste a "
            'sequence, or a FASTA record starting with ">".'
        )
    if first.startswith(">"):
        return parse_fasta(text)
    return [
        SequenceRecord(name=default_name, sequence=_clean("\n".join(body), "input"))
    ]


def decode_text(payload: bytes, source: str) -> str:
    """Bytes to text, strictly, with one located refusal for both doors.

    ``utf-8-sig`` so a byte-order mark is consumed as the encoding artefact it
    is, and *strict* so an undecodable byte is refused rather than repaired.
    The web upload used ``errors="replace"``, which turned a cp1252 header into
    U+FFFD and ran the design anyway: a Latin-1 FASTA the CLI refuses outright
    produced, in the browser, a completed report whose target id had been
    silently altered. One copy of the rule, so the doors cannot drift again.
    """
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SequenceInputError(
            f"{source} is not readable as text -- is it a compressed or "
            "binary file?"
        ) from exc


def read_fasta(path: str | Path) -> list[SequenceRecord]:
    """Read a FASTA file (FR1.2)."""
    path = Path(path)
    if not path.exists():
        raise SequenceInputError(f"no such file: {path}")
    try:
        # utf-8-sig, so a byte-order mark is consumed as the encoding artefact
        # it is. With plain utf-8 the BOM survived into the text, the first
        # line no longer started with ">", and a file whose first visible
        # character is ">" was rejected with "FASTA input must start with a
        # '>' header line" -- a located message naming the one thing that was
        # not wrong. Notepad writes these by default.
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SequenceInputError(
            f"{path.name} is not readable as text -- is it a compressed or "
            "binary file?"
        ) from exc
    except OSError as exc:
        # Everything else the filesystem can say: a directory where a file was
        # meant (a tab-completed `--fasta data/`), a file the user cannot read,
        # a broken symlink. Only UnicodeDecodeError was caught, so these
        # escaped run_request as a raw traceback and exit 1 -- the status
        # `design --help` documents as "the run completed and nothing survived
        # the gate", which a wrapper script reads as an empty but valid result.
        # This module's own docstring says malformed input "has to produce a
        # message that says what is wrong and where, not a stack trace".
        raise SequenceInputError(
            f"cannot read {path}: {exc.strerror or exc}"
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
    # Appended, never inserted: a positional parser of an existing file must
    # not shift. The README says "nothing is dropped from the data; the exports
    # carry the flag", and until this row they did not -- so an exported table
    # of 58 guides, 56 of them single-base shifts of two sites, had nothing in
    # it saying so. That is the precise failure the distinct-site marking
    # exists to prevent, surviving in the file that outlives the session.
    "distinct_site",
    # Appended for the same reason, and under the same rule. The gate composes
    # a number for every verdict it reaches and the export carried only the
    # reason code, so a REVIEW row in a file somebody opens next month said
    # which rule spoke without saying by how much.
    "detail",
    # Where the guide binds, and on which record. Both were added to the report
    # dict when the accessibility track turned out to be drawing guides from
    # one record at another record's coordinates -- "start alone cannot place a
    # guide" -- and then reached no export at all. So a CSV of sixty spacers
    # from a multi-record design said which guide, and never which target.
    "source_target_id",
    "start",
    # Measured by QC for every candidate and, until now, reported nowhere.
    "gc",
    "max_homopolymer",
    "self_fold_stem",
    # Blank unless measured kinetic constants were supplied, which is the
    # point: a column that is empty says the figure was not earned.
    "predicted_signal_nm",
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
        for key in ("reasons", "qc_warnings", "detail"):
            value = row.get(key)
            if isinstance(value, (list, tuple)):
                row[key] = ";".join(str(item) for item in value)
        rows.append(row)
    return rows


def _render_detail(value) -> str:
    """A detail mapping as one stable, comment-safe line.

    Sorted, so two runs with the same parameters produce the same bytes; no
    newlines, because this goes inside a comment line that a parser skips by
    prefix.
    """
    def _flat(item) -> str:
        # Newlines stripped from every value, not only the scalar branch. A
        # provenance header is comment lines a parser skips *by prefix*, so an
        # embedded newline puts the rest of the value on an unprefixed line and
        # the consumer reads it as data -- the corruption this module already
        # refuses to allow through the FASTA comment character.
        return " ".join(str(item).split())

    if isinstance(value, dict):
        return ", ".join(f"{key}={_flat(value[key])}" for key in sorted(value))
    return _flat(value)


def provenance_lines(
    report_dict: dict,
    subtitle: str = "",
    limitations: Sequence[str] = (),
) -> list[str]:
    """Everything needed to know how a file was produced (NFR4).

    An exported CSV outlives the session that made it. Without this it is a
    table of numbers carrying no model version, no parameters and no statement
    of what was not checked -- and it looks exactly like a table that does carry
    those things. The limitations travel with it rather than staying in the app,
    because the person reading the file in six months is precisely the person
    who cannot see the app.
    """
    from .. import __version__

    lines = [f"cas13guide {__version__}"]
    if subtitle:
        lines.append(subtitle)
    fingerprint = report_dict.get("spec_fingerprint")
    if fingerprint:
        lines.append(f"spec fingerprint: {fingerprint}")
    counts = report_dict.get("verdict_counts") or {}
    if counts:
        lines.append(
            "verdicts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            # ``ranked``, not ``exported``: this header is shared by every
            # file, and the controls FASTA exports none of these rows. A count
            # that is true of the run is true in all three; a count claiming to
            # be of the file was false in one of them.
            + f"; ranked={len(report_dict.get('ranked', []))}"
        )
    for record in (report_dict.get("provenance") or {}).get("records", []):
        kind = record.get("kind")
        # `transcriptome` and `exclusion` were both filtered out, and both
        # belong in a file that outlives the session more than anything else
        # here does. A transcriptome record is the only entry naming the world
        # a guide was called specific against; an exclusion is something the
        # run deliberately did not check, and this module's own docstring says
        # a silent exclusion is the bug. Without them a CSV could not say what
        # it was searched over even when the ledger could.
        #
        # Still omitted, deliberately: `spec` (the fingerprint is already on
        # its own line above), `hazard_weights` and `control_flow` (both are
        # in report.json, and neither changes what the rows mean).
        if kind == "exclusion":
            detail = record.get("detail") or {}
            lines.append(f"excluded: {record.get('name')} -- {detail.get('why', '')}".rstrip(" -"))
            continue
        if kind in ("model", "calibration", "search", "transcriptome"):
            line = f"{kind}: {record.get('name')} = {record.get('version')}"
            # The parameters that generated a model, not only its version
            # string. Recording the calibration's seed and noise in the ledger
            # and not here left the exported header byte-identical for two runs
            # whose every number differed -- which is the failure the ledger
            # entry was added to fix, surviving one level out in the file that
            # is supposed to outlive the session.
            detail = record.get("detail") or {}
            source = detail.get("source")
            if source:
                line += f" [{_render_detail(source)}]"
            elif kind == "transcriptome":
                # The digest is the load-bearing one: the build string beside
                # it is asserted, not verified, so it is the digest that tells
                # two worlds apart six months from now.
                shown = {
                    key: detail[key]
                    for key in ("loci", "digest", "expression", "index")
                    if key in detail
                }
                if shown:
                    line += f" [{_render_detail(shown)}]"
            lines.append(line)
    for shortfall in report_dict.get("shortfalls") or []:
        lines.append(f"shortfall: {shortfall}")
    for limitation in limitations:
        lines.append(f"limitation: {limitation}")
    return lines


def provenance_header(
    report_dict: dict,
    subtitle: str = "",
    limitations: Sequence[str] = (),
    prefix: str = "# ",
) -> str:
    return "".join(
        f"{prefix}{line}\n"
        for line in provenance_lines(report_dict, subtitle, limitations)
    )


def ranked_to_csv(
    report_dict: dict,
    subtitle: str = "",
    limitations: Sequence[str] = (),
) -> str:
    """Ranked candidates as CSV, behind a provenance header (FR8.3, NFR4).

    The header is comment-prefixed, which every reader this is aimed at
    (pandas' ``comment='#'``, R's ``read.csv(comment.char='#')``, a spreadsheet
    import dialog) can skip, and which none of them mistakes for data.
    """
    rows = ranked_rows(report_dict)
    columns = list(RANKED_COLUMNS) + [
        "required_coverage",
        "worst_off_target",
        "cumulative_burden",
    ]
    out = io.StringIO()
    out.write(provenance_header(report_dict, subtitle, limitations))
    writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column) for column in columns})
    return out.getvalue()


def ranked_to_fasta(
    report_dict: dict,
    limit: int | None = None,
    subtitle: str = "",
    limitations: Sequence[str] = (),
) -> str:
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
        # Same reason the CSV carries it: without this, an exported FASTA of
        # sixty spacers looks like sixty independent candidates when it may be
        # two sites and fifty-eight single-base shifts of them.
        if "distinct_site" in entry:
            parts.append(f"distinct_site={entry['distinct_site']}")
        # This file's whole claim is that it is "useful on its own rather than
        # only as a companion to the CSV", and a spacer with no target and no
        # coordinate is not: recovering either means searching the sequence
        # back against the input by hand.
        for key, label in (("source_target_id", "target"), ("start", "start"),
                           ("gc", "gc")):
            if entry.get(key) is not None:
                parts.append(f"{label}={entry[key]}")
        records.append(
            SequenceRecord(
                name=str(entry.get("guide_id")),
                sequence=str(entry.get("spacer")),
                description=" ".join(parts),
            )
        )
    # ';' is the FASTA comment character; '#' is read as sequence by some
    # parsers, and a provenance line concatenated onto a spacer is the kind
    # of corruption that survives review.
    return (provenance_header(report_dict, subtitle, limitations, prefix="; ")
            + format_fasta(records))


def controls_to_fasta(
    report_dict: dict,
    subtitle: str = "",
    limitations: Sequence[str] = (),
) -> str:
    """Non-targeting controls as FASTA, so they can be ordered alongside.

    Carries the same provenance header as every other export, and for a
    sharper reason than the ranked file does. A control spacer is a *negative*
    result by construction: it looks like any other 22-mer, it is ordered into
    the same plate, and six months later nothing distinguishes it from a guide
    except a record saying which panel it was matched to.

    It also states its own count. An empty control set used to produce a
    zero-byte file, announced in the same list as the CSV and the report --
    which reads as "controls: none needed" rather than "the panel these would
    have been matched to is empty", and those are opposite facts.
    """
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
    notes = [f"non-targeting controls: {len(records)}"]
    if not records:
        notes.append(
            "none were generated, because the panel they are matched to is "
            "empty -- a control is a shuffle of a panel spacer. The shortfalls "
            "above say why there is no panel."
        )
    return (
        provenance_header(report_dict, subtitle, limitations, prefix="; ")
        + "".join(f"; {note}\n" for note in notes)
        + format_fasta(records)
    )


def write_exports(
    report_dict: dict,
    directory: str | Path,
    stem: str = "candidates",
    subtitle: str = "",
    limitations: Sequence[str] = (),
) -> dict[str, Path]:
    """Write the CSV and FASTA exports side by side. Returns the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for label, suffix, text in (
        ("csv", ".csv", ranked_to_csv(report_dict, subtitle, limitations)),
        ("fasta", ".fasta", ranked_to_fasta(report_dict, None, subtitle, limitations)),
        ("controls", "_controls.fasta",
         controls_to_fasta(report_dict, subtitle, limitations)),
    ):
        path = directory / f"{stem}{suffix}"
        path.write_text(text, encoding="utf-8", newline="\n")
        written[label] = path
    return written

