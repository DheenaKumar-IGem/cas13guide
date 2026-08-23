"""Model and structure registry.

Models declare the domain they support. Asking a model outside its domain is
allowed, but the answer is marked out-of-domain and the calibration service
widens the interval accordingly -- the combination is what stops a confident
number from being produced for a case nobody ever validated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..contract.enums import Effector
from ..util.duplex import DuplexProfile, duplex_penalty
from ..util.seq import gc_fraction


@dataclass(frozen=True, slots=True)
class ModelDomain:
    """What a model was actually fit on."""

    effectors: frozenset[Effector]
    spacer_lengths: frozenset[int]
    max_mismatches: int

    def supports(self, effector: Effector, spacer_length: int, n_mismatch: int) -> bool:
        return (
            effector in self.effectors
            and spacer_length in self.spacer_lengths
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
        self.domain = ModelDomain(
            effectors=frozenset({Effector.RFX_CAS13D, Effector.LWA_CAS13A}),
            spacer_lengths=frozenset({22, 23, 28, 30}),
            max_mismatches=6,
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

    def accessibility(self, sequence: str, start: int, length: int) -> float | None:
        return None


@dataclass(slots=True)
class ModelRegistry:
    """Versioned handles to every model the pipeline may consult."""

    activity: ActivityModel
    accessibility: AccessibilityModel

    def versions(self) -> dict[str, str]:
        return {"activity": self.activity.version, "accessibility": self.accessibility.version}
