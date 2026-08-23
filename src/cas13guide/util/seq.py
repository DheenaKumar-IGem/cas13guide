"""Sequence primitives. RNA alphabet internally; DNA input is transcribed."""

from __future__ import annotations

import re

RNA_COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"}
VALID_RNA = frozenset("AUGCN")

class _NonRnaBase(ValueError):
    """Internal signal from the translation table. Never escapes this module."""


class _StrictTranslation(dict):
    """Translation table that refuses characters outside the RNA alphabet.

    A plain ``str.maketrans`` table leaves unmapped characters *unchanged*,
    which would quietly turn a malformed sequence into a plausible-looking
    duplex instead of an error.

    The raised type is deliberately **not** a ``LookupError``: CPython's
    charmap translation catches ``LookupError`` from the mapping and treats it
    as "no entry, keep the character", so a ``__missing__`` that raises
    ``KeyError`` is silently swallowed and the strictness evaporates.
    :func:`complement` converts this back into the ``KeyError`` that a bare
    ``RNA_COMPLEMENT[b]`` lookup used to raise.
    """

    def __missing__(self, key: int) -> str:
        raise _NonRnaBase(chr(key))


#: Derived from :data:`RNA_COMPLEMENT` rather than written out, so the two
#: cannot drift apart. ``complement`` runs once per tiled window and repeatedly
#: inside the QC self-folding and direct-repeat checks, so the C-level
#: translate is worth ~11x on a spacer and ~67x on a transcript over a
#: per-character generator.
_COMPLEMENT_TABLE = _StrictTranslation(str.maketrans(RNA_COMPLEMENT))


def normalize(seq: str) -> str:
    """Uppercase, strip whitespace, and transcribe T -> U."""
    s = "".join(seq.split()).upper().replace("T", "U")
    bad = set(s) - VALID_RNA
    if bad:
        raise ValueError(f"non-RNA characters in sequence: {sorted(bad)}")
    return s


def complement(seq: str) -> str:
    """Base-for-base complement. ``KeyError`` on any non-RNA character."""
    try:
        return seq.translate(_COMPLEMENT_TABLE)
    except _NonRnaBase as bad:
        raise KeyError(bad.args[0]) from None


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
    """Plain positional difference count. **Not** a duplex distance.

    Deliberately wobble-blind, and therefore the wrong function for comparing a
    spacer to a target site: G-U pairs are literal differences but real base
    pairs, so this over-counts mismatches and inflates apparent specificity.
    Use :func:`cas13guide.util.duplex.profile_duplex` for anything that a
    specificity claim depends on. Kept as a generic string primitive.
    """
    if len(a) != len(b):
        raise ValueError("hamming requires equal lengths")
    return sum(1 for x, y in zip(a, b) if x != y)


def windows(seq: str, size: int, step: int = 1):
    """Yield (start, subsequence) tiling windows, 0-based half-open starts."""
    for i in range(0, len(seq) - size + 1, step):
        yield i, seq[i : i + size]
