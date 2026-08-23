"""Context evidence: expression and localization, always as bounds (fix #6).

The original burden formula bounded activity but multiplied it by a *point*
expression estimate -- the axis with by far the larger uncertainty. Here every
expression value becomes an :class:`Interval`, and an unmeasured transcript
becomes maximal ignorance rather than an implicit zero. "Not detected" is not
"absent".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..metrics.bounds import Interval

#: Half-saturation constant for the abundance factor, in TPM-like units.
SATURATION_K = 50.0


def abundance_factor(tpm: float, k: float = SATURATION_K) -> float:
    """Map abundance onto ``[0, 1)`` with saturating (dose-response) shape.

    Keeps the hazard product dimensionless so gate thresholds stay meaningful,
    and stops one enormously abundant transcript from dominating every sum.
    """
    if tpm < 0:
        raise ValueError("expression cannot be negative")
    return tpm / (tpm + k)


@runtime_checkable
class ExpressionSource(Protocol):
    """Seam for a real expression source (GTEx, user RNA-seq, scRNA panel)."""

    version: str

    def expression_tpm(self, locus_id: str) -> float | None: ...


class StaticExpressionSource:
    """Reference implementation backed by values already on the loci."""

    version = "expression/static/v1"

    def __init__(self, values: dict[str, float | None]) -> None:
        self._values = dict(values)

    def expression_tpm(self, locus_id: str) -> float | None:
        return self._values.get(locus_id)


def expression_interval(
    tpm: float | None,
    relative_uncertainty: float = 0.25,
    max_plausible_tpm: float = 1000.0,
) -> Interval:
    """Bounded abundance factor for one locus.

    ``tpm is None`` yields ``[0, abundance_factor(max_plausible)]`` flagged
    ``imputed``, which the gate reads as an R2 evidence gap.
    """
    if tpm is None:
        return Interval.unknown(
            lo=0.0, hi=abundance_factor(max_plausible_tpm), stratum="expression/unmeasured"
        )
    lo = abundance_factor(max(0.0, tpm * (1.0 - relative_uncertainty)))
    hi = abundance_factor(tpm * (1.0 + relative_uncertainty))
    return Interval(
        point=abundance_factor(tpm), lower=lo, upper=hi, stratum="expression/measured"
    )
