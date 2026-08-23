"""Guide:target duplex free energy, and why it is a window rather than a score
(FR5.1, FR5.2).

``util/duplex.py`` already scores a duplex, but positionally: it counts
mismatches and weights them by where they sit relative to the seed. That is a
*specificity* statement. It says nothing about how tightly the guide actually
binds, because it cannot distinguish a GC-rich perfect duplex from an AU-rich
one -- both are "0 mismatches".

Free energy is the missing axis, and it behaves differently from every other
metric in this package: it has an **optimum, not a direction**.

* too weak -- the guide does not load or does not stay bound, so no signal;
* too strong -- the guide tolerates mismatches it should reject, so
  discrimination collapses and off-targets light up.

That is why ``duplex_dg`` is never a Pareto objective. Every objective in
:meth:`MetricVector.objectives` is oriented so larger is better; a quantity with
an interior optimum cannot be, and forcing it in would make the frontier claim
that the most tightly-bound guide dominates -- exactly backwards for a
discrimination assay. It is carried as a reported fact and a range flag.

Where the range comes from
--------------------------
It is assay-specific and this package does not invent one.
:data:`DEFAULT_DG_PER_NT_RANGE` is expressed per nucleotide so it at least
transfers across spacer lengths, and :func:`summarize` derives an empirical
range from a candidate set so a spec can be set from data rather than from a
number somebody remembered. ``Thresholds.duplex_dg_per_nt_min/max`` default to
``None``, i.e. no flagging, until a range is chosen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev

#: Per-nucleotide free energy window, kcal/mol/nt. **Provisional**: a plausible
#: band for a fully-matched RNA:RNA duplex, provided so the machinery is
#: exercisable, not because this package measured it. Set a real range from
#: :func:`summarize` on your own candidates before relying on the flag.
DEFAULT_DG_PER_NT_RANGE: tuple[float, float] = (-2.2, -1.2)


class ThermodynamicsUnavailable(RuntimeError):
    """ViennaRNA is not importable, so no free energy can be computed."""


def _require_rna():
    try:
        import RNA
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ThermodynamicsUnavailable(
            "ViennaRNA is not installed. `pip install ViennaRNA` to enable "
            "duplex free-energy filtering (FR5.x)."
        ) from exc
    return RNA


@dataclass(frozen=True, slots=True)
class DuplexEnergy:
    """Predicted guide:target hybridization energy (FR5.1)."""

    spacer: str
    target_site: str
    #: Total duplex free energy, kcal/mol. Negative is more stable.
    dg: float
    #: Length-normalized, so specs transfer across spacer lengths.
    dg_per_nt: float
    method: str = "thermodynamics/vienna-duplexfold/v1"

    def in_range(self, low: float | None, high: float | None) -> bool:
        """Is this inside the configured optimal window (FR5.2)?

        ``low``/``high`` are per-nucleotide and ordered as numbers, so ``low``
        is the *more negative* (more stable) end. A ``None`` bound is open.
        """
        if low is not None and self.dg_per_nt < low:
            return False
        if high is not None and self.dg_per_nt > high:
            return False
        return True

    def describe(self, low: float | None, high: float | None) -> str:
        if self.in_range(low, high):
            return f"{self.dg:.1f} kcal/mol ({self.dg_per_nt:.2f}/nt), within range"
        if low is not None and self.dg_per_nt < low:
            return (
                f"{self.dg:.1f} kcal/mol ({self.dg_per_nt:.2f}/nt) is more stable "
                f"than {low:.2f}/nt: over-stable duplexes tolerate mismatches "
                "they should reject, which costs discrimination"
            )
        return (
            f"{self.dg:.1f} kcal/mol ({self.dg_per_nt:.2f}/nt) is weaker than "
            f"{high:.2f}/nt: the guide may not stay bound, which costs signal"
        )


def duplex_free_energy(spacer: str, target_site: str) -> DuplexEnergy:
    """Hybridization energy of one guide against one site (FR5.1).

    Uses ``RNAduplex``, which considers only intermolecular pairs. That is the
    right model here: intramolecular folding of the spacer is already a
    separate QC check (``longest_self_complement``) and of the target is
    already the accessibility axis, so letting either back in would double-count
    it inside a number meant to describe the hybrid alone.
    """
    RNA = _require_rna()
    if not spacer or not target_site:
        raise ValueError("duplex_free_energy needs two non-empty sequences")
    duplex = RNA.duplexfold(spacer, target_site)
    dg = float(duplex.energy)
    return DuplexEnergy(
        spacer=spacer,
        target_site=target_site,
        dg=dg,
        dg_per_nt=dg / len(spacer),
    )


@dataclass(frozen=True, slots=True)
class EnergySummary:
    """Distribution of duplex energies over a candidate set."""

    n: int
    mean_per_nt: float
    sd_per_nt: float
    min_per_nt: float
    max_per_nt: float

    def suggested_range(self, sd: float = 1.0) -> tuple[float, float]:
        """A data-derived window: mean +/- ``sd`` standard deviations.

        Offered so a spec can set :class:`Thresholds` from the actual candidate
        distribution instead of a remembered constant. It is a starting point
        for tuning against measured signal, not a validated band.
        """
        return (self.mean_per_nt - sd * self.sd_per_nt,
                self.mean_per_nt + sd * self.sd_per_nt)


def summarize(pairs) -> EnergySummary:
    """Energy distribution over ``(spacer, target_site)`` pairs."""
    values = [duplex_free_energy(s, t).dg_per_nt for s, t in pairs]
    if not values:
        raise ValueError("no pairs to summarize")
    return EnergySummary(
        n=len(values),
        mean_per_nt=mean(values),
        sd_per_nt=pstdev(values) if len(values) > 1 else 0.0,
        min_per_nt=min(values),
        max_per_nt=max(values),
    )


@dataclass(slots=True)
class ViennaRNADuplexEnergyModel:
    """``DuplexEnergyModel`` adapter over :func:`duplex_free_energy`.

    Caches by ``(spacer, target_site)``: a design run asks about the same
    spacer against many sites and the same site for many spacers, and
    ``RNAduplex`` is not free.
    """

    version: str = "thermodynamics/vienna-duplexfold/v1"
    _cache: dict[tuple[str, str], float] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        _require_rna()

    def duplex_dg_per_nt(self, spacer: str, target_site: str) -> float | None:
        if not spacer or not target_site:
            return None
        key = (spacer, target_site)
        cached = self._cache.get(key)
        if cached is None:
            cached = duplex_free_energy(spacer, target_site).dg_per_nt
            self._cache[key] = cached
        return cached
