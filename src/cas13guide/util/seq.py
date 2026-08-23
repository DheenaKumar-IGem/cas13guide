"""Sequence primitives. RNA alphabet internally; DNA input is transcribed."""

from __future__ import annotations

import re

RNA_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"}
VALID_RNA = frozenset("AUGCN")


def normalize(seq: str) -> str:
    """Uppercase, strip whitespace, and transcribe T -> U."""
    s = "".join(seq.split()).upper().replace("T", "U")
    bad = set(s) - VALID_RNA
    if bad:
        raise ValueError(f"non-RNA characters in sequence: {sorted(bad)}")
    return s


def complement(seq: str) -> str:
    return "".join(RNA_COMPLEMENT[b] for b in seq)


def revcomp(seq: str) -> str:
    return complement(seq)[::-1]


def gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(1 for b in seq if b in "GC") / len(seq)


def max_homopolymer(seq: str) -> int:
    """Length of the longest single-base run."""
    if not seq:
        return 0
    return max(len(m.group(0)) for m in re.finditer(r"(.)\1*", seq))


def has_polyu_terminator(seq: str, run: int = 4) -> bool:
    """U-runs of >=4 terminate Pol III transcription of the guide cassette."""
    return "U" * run in seq


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("hamming requires equal lengths")
    return sum(1 for x, y in zip(a, b) if x != y)


def windows(seq: str, size: int, step: int = 1):
    """Yield (start, subsequence) tiling windows, 0-based half-open starts."""
    for i in range(0, len(seq) - size + 1, step):
        yield i, seq[i : i + size]
