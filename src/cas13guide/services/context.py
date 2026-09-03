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
) -> Interval:
    """Bounded abundance factor for one locus.

    ``tpm is None`` yields ``[0, 1]`` flagged ``imputed``, which the gate reads
    as an R2 evidence gap.

    The unmeasured ceiling used to be ``abundance_factor(1000.0)`` = 0.952,
    which is not a bound. ``abundance_factor`` is monotone and approaches 1, so
    any locus measured above 800 TPM produced an upper bound *above* that
    ceiling -- 0.9955 at 9,000 TPM. The "unknown" interval therefore failed to
    contain the very values a measurement could return.

    Hazard reads expression at ``.upper``, so the consequence ran the one
    direction a hazard bound must never move: **deleting a measurement made the
    hazard smaller**. A guide with an off-target on a highly expressed rRNA
    locus failed the hazard budget when the abundance was known and survived as
    REVIEW when it was not, so less evidence bought a more permissive verdict --
    on the metric the report describes as "read high on purpose; a low point
    estimate is not evidence of safety". The shipped demo world alone carries
    rRNA at 9,000 TPM and mito at 2,500, the two biotypes with the highest
    hazard weights.

    1.0 is the supremum of ``abundance_factor``, so it is the only ceiling that
    contains every measurement. That makes an unmeasured off-target read at the
    top of the range, which is what "maximal ignorance" has to mean for a bound
    that is read at its upper end. The R2 tier and the ``expression_unknown``
    flag are what tell a user this is missing evidence rather than a measured
    hazard.
    """
    if tpm is None:
        return Interval.unknown(lo=0.0, hi=1.0, stratum="expression/unmeasured")
    lo = abundance_factor(max(0.0, tpm * (1.0 - relative_uncertainty)))
    hi = abundance_factor(tpm * (1.0 + relative_uncertainty))
    return Interval(
        point=abundance_factor(tpm), lower=lo, upper=hi, stratum="expression/measured"
    )
