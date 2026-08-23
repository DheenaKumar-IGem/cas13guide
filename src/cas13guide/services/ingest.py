"""Load a real tiling screen and refuse to train on a bad one.

Everything in this package has been validated on simulated screens. That is a
real limitation, and this module is the shortest path out of it: point it at a
published screen and it produces the :class:`TrainingExample` list the training
harness expects -- or tells you precisely why it will not.

Real screen files break in predictable ways, and each is checked here:

* DNA alphabet, lowercase, or ambiguity codes instead of clean RNA
* activity reported as **log2 fold change**, where *more negative* means
  *better* knockdown -- silently training on that inverts the model
* inconsistent spacer lengths (mixed effector designs in one file)
* missing or non-numeric activity values
* duplicate guides, often the same guide measured in several replicates
* **no transcript column at all**

That last one is fatal, not a warning. Without a grouping column there is no
way to split by transcript, and the resulting model would be evaluated on
near-duplicates of its own training rows. This module refuses rather than
producing a number that would look fine and mean nothing.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ..util.seq import VALID_RNA, normalize, revcomp
from .training import TrainingExample


class DatasetError(ValueError):
    """Raised when a screen cannot be used for training at all."""


@dataclass(frozen=True, slots=True)
class ColumnMap:
    """Which column holds what. Defaults match the most common layout."""

    spacer: str = "spacer"
    activity: str = "activity"
    group: str = "transcript"
    target_site: str | None = None
    #: ``True`` when larger activity means a better guide (the usual case).
    #: Set ``False`` for log2 fold change / depletion scores.
    higher_is_better: bool = True


@dataclass(slots=True)
class DatasetReport:
    """What the loader found. Warnings are survivable; errors are not."""

    rows_read: int = 0
    rows_kept: int = 0
    groups: int = 0
    spacer_lengths: dict[int, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"read {self.rows_read} rows, kept {self.rows_kept}, "
            f"{self.groups} transcript group(s), "
            f"spacer lengths {dict(sorted(self.spacer_lengths.items()))}"
        ]
        lines.extend(f"  warning: {w}" for w in self.warnings)
        lines.extend(f"  ERROR:   {e}" for e in self.errors)
        return "\n".join(lines)


def _rank_normalize(values: list[float], higher_is_better: bool) -> list[float]:
    """Map to ``(0, 1)`` by within-screen rank.

    Rank rather than min-max because screen scores arrive on wildly different
    native scales (log2 fold change, percent knockdown, z-scores) and a single
    outlier compresses every other row under min-max. Rank also makes the
    direction flip exact rather than approximate.
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [0.5]

    order = sorted(range(n), key=lambda i: values[i], reverse=not higher_is_better)
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # Average rank across ties, mapped into the open interval (0, 1).
        shared = (i + j) / 2
        for k in range(i, j + 1):
            out[order[k]] = (shared + 0.5) / n
        i = j + 1
    return out


def _clean_sequence(raw: str) -> str | None:
    try:
        seq = normalize(raw)
    except ValueError:
        return None
    if not seq or set(seq) - VALID_RNA or "N" in seq:
        return None
    return seq


def load_screen(
    path: str | Path,
    columns: ColumnMap | None = None,
    effector: str = "RfxCas13d",
    assay: str = "cell_culture_knockdown",
    delimiter: str | None = None,
    already_normalized: bool = False,
) -> tuple[list[TrainingExample], DatasetReport]:
    """Read a screen into training examples, reporting every problem found.

    ``already_normalized=True`` takes activity values as-is (they must lie in
    ``[0, 1]``); otherwise values are rank-normalized within the screen, which
    also applies the ``higher_is_better`` direction flip.
    """
    columns = columns or ColumnMap()
    path = Path(path)
    report = DatasetReport()

    if not path.exists():
        raise DatasetError(f"no such file: {path}")

    if delimiter is None:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab", ".txt"} else ","

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        header = reader.fieldnames or []
        missing = [
            name
            for name in (columns.spacer, columns.activity, columns.group)
            if name not in header
        ]
        if missing:
            if columns.group in missing:
                report.errors.append(
                    f"no grouping column {columns.group!r} in {header}. A transcript "
                    "column is required: without it the model cannot be split by "
                    "transcript, and every score would be measured on near-duplicates "
                    "of the training rows."
                )
            for name in missing:
                if name != columns.group:
                    report.errors.append(f"missing required column {name!r} in {header}")
            return [], report

        rows = list(reader)

    report.rows_read = len(rows)

    spacers: list[str] = []
    sites: list[str] = []
    raw_activity: list[float] = []
    groups: list[str] = []
    dropped_sequence = 0
    dropped_activity = 0

    for row in rows:
        spacer = _clean_sequence(row.get(columns.spacer) or "")
        if spacer is None:
            dropped_sequence += 1
            continue

        if columns.target_site:
            site = _clean_sequence(row.get(columns.target_site) or "")
            if site is None or len(site) != len(spacer):
                dropped_sequence += 1
                continue
        else:
            # Tiling screens measure perfect on-target matches.
            site = revcomp(spacer)

        try:
            value = float(row.get(columns.activity) or "")
        except (TypeError, ValueError):
            dropped_activity += 1
            continue
        if math.isnan(value) or math.isinf(value):
            dropped_activity += 1
            continue

        group = (row.get(columns.group) or "").strip()
        if not group:
            dropped_activity += 1
            continue

        spacers.append(spacer)
        sites.append(site)
        raw_activity.append(value)
        groups.append(group)

    if dropped_sequence:
        report.warnings.append(
            f"dropped {dropped_sequence} row(s) with unusable sequence "
            "(non-RNA characters, ambiguity codes, or a length mismatch)"
        )
    if dropped_activity:
        report.warnings.append(
            f"dropped {dropped_activity} row(s) with missing or non-numeric activity"
        )

    if not spacers:
        report.errors.append("no usable rows after cleaning")
        return [], report

    lengths = Counter(len(s) for s in spacers)
    report.spacer_lengths = dict(lengths)
    if len(lengths) > 1:
        dominant, _ = lengths.most_common(1)[0]
        report.warnings.append(
            f"mixed spacer lengths {dict(lengths)}; keeping {dominant}nt only "
            "(a feature layout is fixed-width, and mixing effector designs in one "
            "model is rarely what you want)"
        )
        keep = [i for i, s in enumerate(spacers) if len(s) == dominant]
        spacers = [spacers[i] for i in keep]
        sites = [sites[i] for i in keep]
        raw_activity = [raw_activity[i] for i in keep]
        groups = [groups[i] for i in keep]

    duplicates = sum(count - 1 for count in Counter(spacers).values() if count > 1)
    if duplicates:
        report.warnings.append(
            f"{duplicates} duplicate spacer(s) -- often replicate measurements; "
            "they stay in, but they inflate apparent sample size"
        )

    if already_normalized:
        out_of_range = [v for v in raw_activity if not 0.0 <= v <= 1.0]
        if out_of_range:
            report.errors.append(
                f"already_normalized=True but {len(out_of_range)} value(s) fall "
                f"outside [0, 1] (e.g. {out_of_range[0]})"
            )
            return [], report
        activity = list(raw_activity)
    else:
        activity = _rank_normalize(raw_activity, columns.higher_is_better)
        if not columns.higher_is_better:
            report.warnings.append(
                "higher_is_better=False: treating activity as a depletion score, "
                "so the most negative values become the most active guides"
            )

    unique_groups = sorted(set(groups))
    report.groups = len(unique_groups)
    if len(unique_groups) < 3:
        report.errors.append(
            f"only {len(unique_groups)} transcript group(s); a grouped "
            "train/calibrate/test split needs at least 3"
        )
        return [], report
    if len(unique_groups) < 6:
        report.warnings.append(
            f"only {len(unique_groups)} transcript groups -- grouped CV will be "
            "coarse and its variance high"
        )

    examples = [
        TrainingExample(
            spacer=spacer,
            target_site=site,
            observed_activity=value,
            group=group,
            effector=effector,
            assay=assay,
        )
        for spacer, site, value, group in zip(spacers, sites, activity, groups)
    ]
    report.rows_kept = len(examples)
    return examples, report


def write_screen(
    path: str | Path,
    examples: Iterable[TrainingExample],
    columns: ColumnMap | None = None,
) -> None:
    """Write examples back out. Useful for fixtures and round-trip checks."""
    columns = columns or ColumnMap()
    rows = list(examples)
    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([columns.spacer, columns.activity, columns.group])
        for e in rows:
            writer.writerow([e.spacer, e.observed_activity, e.group])
