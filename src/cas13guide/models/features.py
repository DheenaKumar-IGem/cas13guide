"""Feature extraction for Cas13 activity prediction.

Every feature is named. The activity model is the one component whose output
the entire bound algebra rests on, so being able to read its coefficients --
and to notice when a feature is doing suspiciously heavy lifting -- matters
more here than squeezing out the last point of skill.

Feature blocks, for a spacer of length ``L``:

===================  ======  =========================================
block                count   what it encodes
===================  ======  =========================================
``pos{i}_{base}``    4L      positional one-hot of the spacer
``mm{i}``            L       mismatch at spacer position i
``gu{i}``            L       G-U wobble at spacer position i
``di_{XY}``          16      dinucleotide composition
global               8       GC, seed GC, counts, penalty, structure
===================  ======  =========================================

Positional one-hot plus positional mismatch indicators is deliberate: Cas13
activity is strongly position-dependent (the central seed), and a model given
only aggregate counts cannot express that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..util.duplex import DuplexProfile, duplex_penalty
from ..util.seq import gc_fraction, max_homopolymer

BASES = ("A", "C", "G", "U")
DINUCLEOTIDES = tuple(a + b for a in BASES for b in BASES)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Fixed-width feature layout for one spacer length."""

    spacer_length: int
    names: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return len(self.names)

    @property
    def blocks(self) -> dict[str, tuple[int, int]]:
        """Half-open ``[start, end)`` offsets of each feature block."""
        L = self.spacer_length
        return {
            "onehot": (0, 4 * L),
            "mismatch": (4 * L, 5 * L),
            "wobble": (5 * L, 6 * L),
            "dinucleotide": (6 * L, 6 * L + 16),
            "global": (6 * L + 16, 6 * L + 24),
        }

    @classmethod
    def for_length(cls, spacer_length: int) -> "FeatureSpec":
        names: list[str] = []
        for i in range(1, spacer_length + 1):
            names.extend(f"pos{i}_{b}" for b in BASES)
        names.extend(f"mm{i}" for i in range(1, spacer_length + 1))
        names.extend(f"gu{i}" for i in range(1, spacer_length + 1))
        names.extend(f"di_{d}" for d in DINUCLEOTIDES)
        names.extend(
            [
                "gc",
                "gc_seed",
                "n_mismatch",
                "n_wobble",
                "duplex_penalty",
                "max_homopolymer",
                "accessibility",
                "has_accessibility",
            ]
        )
        return cls(spacer_length=spacer_length, names=tuple(names))

    def transform(
        self,
        spacer: str,
        duplex: DuplexProfile,
        accessibility: float | None = None,
    ) -> np.ndarray:
        """One feature row. Deterministic and side-effect free."""
        length = self.spacer_length
        if len(spacer) != length:
            raise ValueError(
                f"spacer length {len(spacer)} does not match feature spec ({length})"
            )
        if duplex.length != length:
            raise ValueError("duplex profile length does not match the spacer")

        x = np.zeros(self.n_features, dtype=np.float64)
        cursor = 0

        # positional one-hot
        for i, base in enumerate(spacer):
            if base in BASES:
                x[cursor + i * 4 + BASES.index(base)] = 1.0
        cursor += 4 * length

        # positional mismatch / wobble indicators (1-based spacer positions)
        for pos in duplex.mismatch_positions:
            x[cursor + pos - 1] = 1.0
        cursor += length
        for pos in duplex.wobble_positions:
            x[cursor + pos - 1] = 1.0
        cursor += length

        # dinucleotide composition, normalized
        if length > 1:
            index = {d: i for i, d in enumerate(DINUCLEOTIDES)}
            for i in range(length - 1):
                key = spacer[i : i + 2]
                if key in index:
                    x[cursor + index[key]] += 1.0
            x[cursor : cursor + 16] /= length - 1
        cursor += 16

        # global features
        seed_lo = int(0.35 * length)
        seed_hi = max(seed_lo + 1, int(0.75 * length))

        x[cursor + 0] = gc_fraction(spacer)
        x[cursor + 1] = gc_fraction(spacer[seed_lo:seed_hi])
        x[cursor + 2] = duplex.n_mismatch / length
        x[cursor + 3] = duplex.n_wobble / length
        x[cursor + 4] = duplex_penalty(duplex) / max(1.0, length)
        x[cursor + 5] = max_homopolymer(spacer) / length
        x[cursor + 6] = 0.0 if accessibility is None else accessibility
        x[cursor + 7] = 0.0 if accessibility is None else 1.0
        return x

    def transform_many(self, rows) -> np.ndarray:
        """Stack feature rows into a design matrix."""
        items = list(rows)
        if not items:
            return np.zeros((0, self.n_features), dtype=np.float64)
        return np.vstack([self.transform(*item) for item in items])
