"""Model and structure registry.

Models declare the domain they support. Asking a model outside its domain is
allowed, but the answer is marked out-of-domain and the calibration service
widens the interval accordingly -- the combination is what stops a confident
number from being produced for a case nobody ever validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from ..contract.effectors import EFFECTORS
from ..contract.enums import Effector
from ..util.duplex import DuplexProfile, duplex_penalty
from ..util.seq import gc_fraction


@dataclass(frozen=True, slots=True)
class ModelDomain:
    """What a model was actually fit on.

    Effector and spacer length are **paired**, not independent axes. A flat
    ``spacer_lengths`` set is only a domain when the model covers one
    ortholog; across two it is a union, and a union admits every ortholog's
    lengths for every other ortholog. That is not a subtle over-admission:
    RfxCas13d publishes 22/23/30 nt and LwaCas13a publishes 28, so a union of
    the two accepts *all four* for *both* -- exactly the geometries neither was
    validated at, and precisely the case the package promises to report
    out-of-domain and widen. It is the single check standing between an
    unvalidated spacer length and a certified narrow bound
    (``pipeline/d_pair_evidence.py`` reads nothing else).

    So a multi-effector model must supply ``lengths_by_effector``. A
    single-effector model may leave it ``None``, where the flat set is the
    per-effector set and there is nothing to conflate.
    """

    effectors: frozenset[Effector]
    #: The union of lengths across :attr:`effectors`. Kept as the public
    #: summary, and correct on its own only for a single-effector domain.
    spacer_lengths: frozenset[int]
    max_mismatches: int
    #: Per-ortholog lengths. ``None`` means "the same lengths for every
    #: effector", which is true for one effector and false for two.
    lengths_by_effector: Mapping[Effector, frozenset[int]] | None = None

    def __post_init__(self) -> None:
        if self.lengths_by_effector is None and len(self.effectors) > 1:
            raise ValueError(
                "a domain covering more than one effector must state its "
                "lengths per effector: a union admits each ortholog's lengths "
                f"for the others. Got effectors={sorted(e.value for e in self.effectors)} "
                "with a flat spacer_lengths set."
            )

    def lengths_for(self, effector: Effector) -> frozenset[int]:
        """The lengths this model was fit on *for this ortholog*."""
        if self.lengths_by_effector is None:
            return self.spacer_lengths if effector in self.effectors else frozenset()
        return self.lengths_by_effector.get(effector, frozenset())

    def supports(self, effector: Effector, spacer_length: int, n_mismatch: int) -> bool:
        return (
            effector in self.effectors
            and spacer_length in self.lengths_for(effector)
            and n_mismatch <= self.max_mismatches
        )


@runtime_checkable
class ActivityModel(Protocol):
    """The seam a trained model plugs into.

    ``runtime_checkable`` lets callers assert that a plugged-in object
    satisfies the interface. Note that it verifies *member presence* only, not
    signatures -- it catches a missing ``predict``, not a wrong one.
    """

    version: str
    domain: ModelDomain

    def predict(self, spacer: str, duplex: DuplexProfile, accessibility: float | None) -> float: ...


@runtime_checkable
class AccessibilityModel(Protocol):
    version: str
    #: True only for a source independent of the activity model -- a probing
    #: experiment (SHAPE, DMS-seq, icSHAPE). A thermodynamic fold is a
    #: prediction from the same sequence the model already sees, so it sets
    #: this False and is used to rank rather than to tighten a bound. See
    #: ``services/accessibility.py``.
    is_independent_evidence: bool

    def accessibility(self, sequence: str, start: int, length: int) -> float | None: ...


class HeuristicCas13Model:
    """Reference activity model.

    Deliberately simple and clearly labelled: a logistic over
    position-weighted duplex disruption, GC content and (when available)
    independent target accessibility. It exists so the pipeline runs
    end-to-end; a trained CNN or the published tiling-screen models drop in
    behind the same Protocol without touching anything downstream.
    """

    version = "model/heuristic-cas13/v1"

    def __init__(self) -> None:
        effectors = frozenset({Effector.RFX_CAS13D, Effector.LWA_CAS13A})
        self.domain = ModelDomain(
            effectors=effectors,
            # Derived from contract/effectors.py rather than restated. The
            # literal set was {22, 23, 28, 30} -- the same union, but maintained
            # by hand, so contract/effectors.py's claim that "the registry
            # cross-checks its supported domain against these values" was simply
            # untrue, and a spacer length added to an ortholog would never have
            # reached the model's declared domain.
            spacer_lengths=frozenset(
                length for e in effectors for length in EFFECTORS[e].spacer_lengths
            ),
            max_mismatches=6,
            # Paired with the ortholog, not flattened across orthologs. The
            # flat union above is {22, 23, 28, 30}, which would admit
            # LwaCas13a at 22 nt (RfxCas13d's length) and RfxCas13d at 28 nt
            # (LwaCas13a's) as in-domain -- and every miRNA preset resolves to
            # 22 or 23 nt against the default LwaCas13a, so that union was
            # certifying the package's own headline path at a geometry no
            # model here was fit on.
            lengths_by_effector={
                e: frozenset(EFFECTORS[e].spacer_lengths) for e in effectors
            },
        )

    def predict(self, spacer: str, duplex: DuplexProfile, accessibility: float | None) -> float:
        penalty = duplex_penalty(duplex)
        gc = gc_fraction(spacer)
        # Mild optimum around 50% GC; strong penalty for duplex disruption.
        gc_term = -4.0 * (gc - 0.5) ** 2
        access_term = 0.0 if accessibility is None else 1.2 * (accessibility - 0.5)
        logit = 2.6 - 1.9 * penalty + gc_term + access_term
        return 1.0 / (1.0 + math.exp(-logit))


class NullAccessibilityModel:
    """No independent structural evidence available.

    Returning ``None`` is meaningful: the design only credits accessibility
    when it comes from a source independent of the activity model (SHAPE,
    DMS-seq, icSHAPE), never from the thermodynamic fold the model already
    saw.
    """

    version = "accessibility/none/v1"
    #: Moot -- this model never returns a value -- but the attribute has to
    #: exist for the Protocol to be satisfiable.
    is_independent_evidence = False

    def accessibility(self, sequence: str, start: int, length: int) -> float | None:
        return None


@runtime_checkable
class DuplexEnergyModel(Protocol):
    """Seam for guide:target hybridization energy (FR5.1)."""

    version: str

    def duplex_dg_per_nt(self, spacer: str, target_site: str) -> float | None: ...


class NullDuplexEnergyModel:
    """No thermodynamic evidence. ``None`` means unmeasured, not zero."""

    version = "thermodynamics/none/v1"

    def duplex_dg_per_nt(self, spacer: str, target_site: str) -> float | None:
        return None


@runtime_checkable
class KineticsModel(Protocol):
    """Seam for collateral-cleavage signal prediction (FR6.1, FR6.2)."""

    version: str

    def predict(self, activity_lower: float, accessibility: float | None = None): ...


def _null_kinetics():
    # Imported lazily: services.kinetics pulls in numpy, and registry is on the
    # import path of every module in the package.
    from .kinetics import NullKineticsModel

    return NullKineticsModel()


@dataclass(slots=True)
class ModelRegistry:
    """Versioned handles to every model the pipeline may consult."""

    activity: ActivityModel
    accessibility: AccessibilityModel
    #: Optional so every existing two-argument construction keeps working.
    energy: DuplexEnergyModel = field(default_factory=NullDuplexEnergyModel)
    #: Collateral-cleavage kinetics (FR6). Only consulted when the assay treats
    #: collateral activity as the *readout* rather than as a hazard.
    kinetics: "KineticsModel" = field(default_factory=lambda: _null_kinetics())

    def versions(self) -> dict[str, str]:
        return {
            "activity": self.activity.version,
            "accessibility": self.accessibility.version,
            "energy": self.energy.version,
            "kinetics": self.kinetics.version,
        }
