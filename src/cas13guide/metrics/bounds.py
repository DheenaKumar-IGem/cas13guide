"""The bound algebra every score in the system is expressed in.

The asymmetry is the whole point: benefit is read at its *lower* confidence
bound, hazard at its *upper* one. A low point prediction is not evidence of
safety, so hazards must never be read at their point estimate.

This module imports nothing, deliberately. :class:`Interval` is a primitive --
``models/evidence.py`` is built out of it, and ``metrics/aggregate.py`` consumes
both -- so the ``metrics`` package spans two layers: this file sits *below*
``models`` and its sibling sits above. That reads as an upward dependency if you
group by package and is acyclic by module, which is what actually matters. Add a
top-level import of ``models`` here and it stops being: ``tests/test_layering.py``
fails if anything in this package grows one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Interval:
    """A calibrated interval with provenance about how it was produced.

    ``lower``/``upper`` are the calibrated bounds at the spec's alpha.
    ``ood`` marks a prediction the calibration service could not support with
    an in-domain stratum -- the bound is still returned, but widened, and the
    flag propagates all the way to the gate.
    """

    point: float
    lower: float
    upper: float
    ood: bool = False
    stratum: str = "unknown"
    #: True when the underlying quantity was never measured (e.g. expression
    #: with no evidence). Distinct from ood, which is about model domain.
    imputed: bool = False

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(f"inverted interval: [{self.lower}, {self.upper}]")
        # And the estimate has to be inside its own interval.
        #
        # The class already refused an inverted one and said nothing about
        # where `point` sits, which leaves the two halves free to disagree
        # about the same quantity: `point` is the number the report prints,
        # `lower`/`upper` are what the gate reads, and a guide could be shown
        # an activity its own bound excludes.
        #
        # A guard rather than a fix: 119,762 intervals built across the whole
        # suite -- the trained model, kinetics, variant contexts, BLAST --
        # already satisfy it, so nothing here is currently wrong. What was
        # missing is anything that would say so if it stopped being true.
        if not self.lower <= self.point <= self.upper:
            raise ValueError(
                f"point {self.point} outside its own bounds "
                f"[{self.lower}, {self.upper}]"
            )

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def clamped(self, lo: float = 0.0, hi: float = 1.0) -> "Interval":
        return replace(
            self,
            point=min(max(self.point, lo), hi),
            lower=min(max(self.lower, lo), hi),
            upper=min(max(self.upper, lo), hi),
        )

    @classmethod
    def exact(cls, value: float, stratum: str = "exact") -> "Interval":
        return cls(point=value, lower=value, upper=value, stratum=stratum)

    @classmethod
    def unknown(cls, lo: float = 0.0, hi: float = 1.0, stratum: str = "unknown") -> "Interval":
        """Maximal ignorance. Reads as harmless at LCB and worst-case at UCB."""
        return cls(point=(lo + hi) / 2, lower=lo, upper=hi, ood=True, stratum=stratum, imputed=True)


def hazard_product(*intervals: Interval, scale: float = 1.0) -> float:
    """Upper bound on a product of non-negative quantities.

    Used for ``UCB(activity) x UCB(expression) x hazard_weight``. Taking the
    upper bound on *every* factor is fix #6: the original bounded activity but
    passed expression through as a point estimate, which quietly understated
    hazard on the axis with the largest uncertainty.
    """
    out = scale
    for iv in intervals:
        # ``lower``, not ``upper``. The precondition is that every factor is
        # non-negative, and an interval straddling zero satisfies `upper >= 0`
        # while violating it -- so the guard passed exactly the inputs it
        # exists to reject. Multiplying uppers is an upper bound on the product
        # only when no factor can be negative: for [-5, 2] and [-4, 3] the
        # product reaches 20 at the two lower ends, while the old code returned
        # 2 x 3 = 6 and called it a bound.
        #
        # Latent rather than active -- the three real factors (UCB activity,
        # UCB expression, hazard weight) are non-negative by construction -- but
        # a guard that cannot fire in the case it names is not a guard.
        if iv.lower < 0:
            raise ValueError("hazard_product requires non-negative quantities")
        out *= iv.upper
    return out


def benefit_lower(interval: Interval) -> float:
    """Read a benefit at its lower confidence bound."""
    return interval.lower
