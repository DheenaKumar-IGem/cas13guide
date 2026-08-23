"""Collateral-cleavage kinetics and predicted signal strength (FR6.1, FR6.2).

This is the axis that separates a *diagnostic* guide designer from a knockdown
one. Every other metric in this package asks "will this guide bind its target
and nothing else". A diagnostic asks a further question: once bound, how much
**trans**-cleavage does the activated complex do, and therefore how bright does
the well get? Two guides with identical specificity can differ several-fold in
signal, and nothing upstream of this module can see that difference.

The model
---------
A minimal two-step scheme, integrated as an ODE over the assay window:

    activation      d[C*]/dt = k_on * ([C]tot - [C*]) * [T]eff - k_off * [C*]
    trans-cleavage  d[F]/dt  = k_cat * [C*] * [R] / (K_M + [R])
                    d[R]/dt  = -d[F]/dt

``[T]eff`` is the *productive* target concentration: nominal target
concentration scaled by what the rest of the pipeline already established about
this guide -- its calibrated activity lower bound and, when available, the
structural accessibility of the site. A guide that cannot reach its target does
not activate, however good its k_cat is. Reading activity at the **lower**
bound keeps this consistent with the rest of the package: a bright predicted
signal must be one the evidence supports, not one a wide interval permits.

Honesty about the constants
---------------------------
The rate constants belong to PanCRISP's own model (Feng et al. 2024) and are
**not** in this repository. :data:`PROVISIONAL_CONSTANTS` exists so the
machinery is exercisable and testable, and it is flagged ``provisional=True``.

While the constants are provisional:

* :meth:`CollateralKineticsModel.relative_signal` works, because a *relative*
  ranking is invariant to the constants that scale it -- every candidate is
  pushed through the same ODE with the same parameters, so the ordering
  reflects the evidence, which is exactly what FR6.2 asks for;
* :meth:`CollateralKineticsModel.absolute_signal` **raises**, because a
  fluorescence number in real units computed from placeholder constants is a
  fabricated measurement, and this package's entire premise is not doing that.

Supply real constants via :meth:`KineticConstants.from_json` and the absolute
path unlocks itself.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


class ProvisionalConstantsError(RuntimeError):
    """An absolute prediction was requested from placeholder constants."""


@dataclass(frozen=True, slots=True)
class KineticConstants:
    """Rate constants and assay conditions for the collateral-cleavage ODE (DR4).

    Units are declared rather than assumed, because a kinetic constant without
    units is not a constant. Concentrations nM, times seconds.
    """

    #: Target association rate, /nM/s.
    k_on: float
    #: Complex dissociation rate, /s.
    k_off: float
    #: Trans-cleavage turnover number, /s.
    k_cat: float
    #: Michaelis constant for the reporter, nM.
    k_m: float
    #: Total Cas13-crRNA complex, nM.
    complex_nm: float
    #: Nominal target concentration, nM.
    target_nm: float
    #: Initial reporter concentration, nM.
    reporter_nm: float
    #: Assay readout window, seconds.
    assay_seconds: float
    #: Where these came from. Anything but a real citation should stay
    #: provisional.
    source: str = "PLACEHOLDER -- not measured"
    #: True while the values are stand-ins. Gates absolute predictions.
    provisional: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "KineticConstants":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @property
    def version(self) -> str:
        tag = "provisional" if self.provisional else "measured"
        return f"kinetics/collateral-ode/{tag}/v1"


#: Stand-in constants. Order-of-magnitude plausible for a Cas13a fluorescent
#: reporter assay, and explicitly **not** measured. They exist so the ODE and
#: the ranking can be tested; they must be replaced before any absolute claim.
PROVISIONAL_CONSTANTS = KineticConstants(
    k_on=1e-3,
    k_off=1e-4,
    # Chosen so the window ends in the *linear* regime rather than with the
    # reporter exhausted. A saturating assay reports 1.0 for every candidate,
    # which is not a weak ranking but no ranking at all -- the well is simply
    # as bright as it can get. Real constants must be checked the same way:
    # SignalPrediction.saturated says when they have not been.
    k_cat=4e-3,
    k_m=200.0,
    complex_nm=50.0,
    target_nm=1.0,
    reporter_nm=500.0,
    assay_seconds=1800.0,
    source="PLACEHOLDER -- not measured; replace with Feng et al. 2024 values",
    provisional=True,
)


@dataclass(frozen=True, slots=True)
class SignalPrediction:
    """One candidate's predicted collateral behaviour."""

    #: Fraction of complex activated at the end of the window, ``[0, 1]``.
    activated_fraction: float
    #: Peak trans-cleavage rate over the window, nM/s. Peak rather than the
    #: t=0 slope: the system starts with zero activated complex, so its
    #: instantaneous rate at t=0 is identically zero for every candidate and
    #: carries no information. What a plate reader actually reads is the
    #: maximum linear slope, which is this.
    cleavage_rate: float
    #: Cleaved reporter at the end of the window, nM.
    cleaved_reporter_nm: float
    #: ``[0, 1]`` score for ranking. Comparable *within* one run only.
    relative_signal: float
    provisional: bool

    @property
    def saturated(self) -> bool:
        """True when the reporter is nearly exhausted, so the assay cannot
        distinguish this candidate from a stronger one."""
        return self.relative_signal > 0.99


@dataclass(slots=True)
class CollateralKineticsModel:
    """Integrates the collateral-cleavage ODE for one candidate at a time."""

    constants: KineticConstants = PROVISIONAL_CONSTANTS
    #: Integration steps across the assay window. 400 keeps the RK4 error well
    #: below the precision anyone would read off a plate reader.
    steps: int = 400

    @property
    def version(self) -> str:
        return self.constants.version

    def effective_target_nm(
        self, activity_lower: float, accessibility: float | None = None
    ) -> float:
        """Productive target concentration for this guide.

        Activity is read at its lower bound, for the same reason every other
        benefit in this package is: a bright signal predicted off the wide end
        of an uncertain interval is not a prediction, it is a hope.
        """
        productive = max(0.0, min(1.0, activity_lower))
        if accessibility is not None:
            productive *= max(0.0, min(1.0, accessibility))
        return self.constants.target_nm * productive

    def integrate(self, effective_target_nm: float) -> tuple[float, float, float]:
        """RK4 over the two-state system.

        Returns ``(activated_fraction, initial_cleavage_rate, cleaved_nm)``.
        """
        c = self.constants
        dt = c.assay_seconds / self.steps

        def derivatives(state: np.ndarray) -> np.ndarray:
            activated, reporter = state
            free = max(0.0, c.complex_nm - activated)
            d_activated = c.k_on * free * effective_target_nm - c.k_off * activated
            turnover = c.k_cat * activated * reporter / (c.k_m + reporter)
            return np.array([d_activated, -turnover])

        state = np.array([0.0, c.reporter_nm], dtype=float)
        peak_rate = 0.0
        for _step in range(self.steps):
            k1 = derivatives(state)
            peak_rate = max(peak_rate, float(-k1[1]))
            k2 = derivatives(state + 0.5 * dt * k1)
            k3 = derivatives(state + 0.5 * dt * k2)
            k4 = derivatives(state + dt * k3)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            state[0] = min(max(state[0], 0.0), c.complex_nm)
            state[1] = min(max(state[1], 0.0), c.reporter_nm)

        activated_fraction = float(state[0] / c.complex_nm) if c.complex_nm else 0.0
        cleaved = float(c.reporter_nm - state[1])
        return activated_fraction, peak_rate, cleaved

    def predict(
        self, activity_lower: float, accessibility: float | None = None
    ) -> SignalPrediction:
        """FR6.1 + FR6.2 for a single candidate."""
        target = self.effective_target_nm(activity_lower, accessibility)
        activated, rate, cleaved = self.integrate(target)
        relative = cleaved / self.constants.reporter_nm if self.constants.reporter_nm else 0.0
        return SignalPrediction(
            activated_fraction=activated,
            cleavage_rate=rate,
            cleaved_reporter_nm=cleaved,
            relative_signal=min(1.0, max(0.0, relative)),
            provisional=self.constants.provisional,
        )

    def absolute_signal(
        self, activity_lower: float, accessibility: float | None = None
    ) -> float:
        """Cleaved reporter in nM -- only with real constants.

        Refuses while :attr:`KineticConstants.provisional` is set. A
        fluorescence figure in real units derived from placeholder rate
        constants would be a fabricated measurement wearing a unit label, and
        it would be indistinguishable downstream from a measured one.
        """
        if self.constants.provisional:
            raise ProvisionalConstantsError(
                "kinetic constants are provisional "
                f"({self.constants.source!r}), so an absolute signal would be "
                "fabricated. Use relative_signal() for ranking, or supply "
                "measured constants via KineticConstants.from_json(...) with "
                "provisional=False."
            )
        return self.predict(activity_lower, accessibility).cleaved_reporter_nm


@dataclass(frozen=True, slots=True)
class NullKineticsModel:
    """No kinetics evidence. ``None`` means unmodelled, not zero signal."""

    version: str = "kinetics/none/v1"

    def predict(
        self, activity_lower: float, accessibility: float | None = None
    ) -> SignalPrediction | None:
        return None
