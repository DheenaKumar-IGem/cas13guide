"""ResolvedDesignSpec -- the versioned, validated, machine-readable contract.

Everything downstream reads this object and nothing else for intent. In
particular the biological hazard weights live here (fix #7) rather than being
an implicit constant inside the burden formula, so they are auditable and
diffable alongside the rest of the design.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from ..util.seq import normalize
from .effectors import EFFECTORS, EffectorParams
from .enums import (
    ASSAY_REQUIRED_UNIVERSES,
    AssayType,
    Effector,
    Task,
    TargetClass,
    Universe,
)

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TargetSpec(_Frozen):
    """One RNA the design has an opinion about."""

    target_id: str
    sequence: str
    target_class: TargetClass
    gene: str | None = None
    transcript: str | None = None
    biotype: Universe = Universe.CODING
    #: Weight used only for DESIRED targets in the utility metric.
    weight: float = Field(default=1.0, ge=0.0)
    #: Relative abundance (TPM-like). ``None`` means genuinely unknown, which
    #: is an evidence gap, not zero -- "not detected" != absent.
    expression: float | None = Field(default=None, ge=0.0)
    #: Multiplicative widening applied to the expression upper bound when the
    #: measurement is noisy or absent.
    expression_uncertainty: float = Field(default=0.25, ge=0.0)

    @model_validator(mode="after")
    def _normalize_sequence(self):
        object.__setattr__(self, "sequence", normalize(self.sequence))
        return self


#: Default hazard prior per intent class. Module level so that a partial
#: override can be completed from it rather than leaving a hole.
DEFAULT_HAZARD_BY_CLASS: dict[TargetClass, float] = {
    TargetClass.REQUIRED: 0.0,
    TargetClass.DESIRED: 0.0,
    TargetClass.TOLERATED: 0.1,
    TargetClass.AVOID: 1.0,
    TargetClass.FORBIDDEN: 5.0,
}

#: Default hazard prior per sequence universe.
DEFAULT_HAZARD_BY_BIOTYPE: dict[Universe, float] = {
    Universe.RRNA: 2.0,
    Universe.MITO: 1.5,
    Universe.CODING: 1.0,
    Universe.NCRNA: 1.0,
    Universe.PRE_MRNA: 0.7,
    Universe.CIRCRNA: 0.7,
    Universe.REPEAT: 0.5,
    Universe.VIRAL: 0.2,
    Universe.SYNTHETIC: 0.1,
}


class HazardWeights(_Frozen):
    """Versioned biological hazard priors (fix #7).

    These multiply into the off-target burden. They encode judgment, so they
    are versioned and travel in the provenance ledger.

    A partial override means *override these, keep the rest*: both maps are
    completed from the defaults at construction. Tightening one class used to
    require restating all five, and restating four of five raised ``KeyError``
    from deep inside stage D, halfway through a design run.
    """

    version: str = "hazard-weights/v1"
    by_class: dict[TargetClass, float] = Field(
        default_factory=lambda: dict(DEFAULT_HAZARD_BY_CLASS)
    )
    #: Unannotated hits (not named in the contract) still carry hazard.
    unannotated: float = Field(default=0.6, ge=0.0)
    by_biotype: dict[Universe, float] = Field(
        default_factory=lambda: dict(DEFAULT_HAZARD_BY_BIOTYPE)
    )

    @model_validator(mode="after")
    def _complete_maps(self):
        """Fill any class or universe the caller did not mention.

        Every enum member gets an entry, so :meth:`weight_for` cannot raise and
        -- more importantly -- a class the caller forgot cannot silently become
        a zero hazard, which is the one direction this system must never fail.
        """
        by_class = {**DEFAULT_HAZARD_BY_CLASS, **self.by_class}
        by_biotype = {**DEFAULT_HAZARD_BY_BIOTYPE, **self.by_biotype}
        object.__setattr__(self, "by_class", by_class)
        object.__setattr__(self, "by_biotype", by_biotype)
        return self

    def weight_for(self, target_class: TargetClass | None, biotype: Universe) -> float:
        if target_class is None:
            base = self.unannotated
        else:
            # Defence in depth: ``_complete_maps`` guarantees the key exists,
            # but an unannotated-strength prior is the right answer for an
            # unknown class -- never 0.0, which would erase the hazard.
            base = self.by_class.get(target_class, self.unannotated)
        return base * self.by_biotype.get(biotype, 1.0)


class Thresholds(_Frozen):
    """Gate thresholds. Borderline bands drive the R1 REVIEW tier."""

    #: A REQUIRED target must clear this activity lower bound.
    required_coverage_min: Fraction = 0.50
    #: Width of the band below which a passing metric is only "borderline".
    borderline_band: Fraction = 0.10
    #: A FORBIDDEN target above this activity upper bound is a credible hit.
    forbidden_activity_max: Fraction = 0.20
    #: Worst single off-target hazard tolerated.
    worst_off_target_max: float = 1.0
    #: Total off-target hazard tolerated.
    cumulative_burden_max: float = 5.0
    #: Discrimination margin tau for allele/SNV tasks (fix #2).
    discrimination_margin_min: float = 2.0
    #: Search must be provably exhaustive to at least this mismatch depth.
    required_search_mismatches: int = 4
    #: Nominal miscoverage for the conformal calibration service.
    alpha: Fraction = 0.10
    #: FR3.4 -- mean unpaired probability below which the intended site counts
    #: as too structured, raising an R1 REVIEW. ``0.0`` disables the check, which
    #: is the default: what counts as "too structured" is an assay decision, and
    #: a threshold nobody chose should not silently start flagging designs.
    min_accessibility: Fraction = 0.0
    #: FR5.2 -- optimal guide:target duplex free-energy window, kcal/mol/nt.
    #: ``low`` is the more-negative (more stable) end. Both default to ``None``
    #: (no flagging): the window is assay-specific, and
    #: ``services.thermodynamics.summarize`` derives one from your own
    #: candidates rather than this package asserting a number it did not measure.
    duplex_dg_per_nt_min: float | None = None
    duplex_dg_per_nt_max: float | None = None


class SearchBudget(_Frozen):
    """Bounds for the backtracking control flow (fix #5)."""

    max_search_escalations: int = 2
    max_panel_retiles: int = 2
    max_candidates: int = 5000


class TargetContract(_Frozen):
    task: Task
    targets: tuple[TargetSpec, ...]

    @model_validator(mode="after")
    def _check(self):
        if not self.targets:
            raise ValueError("target contract needs at least one target")
        ids = [t.target_id for t in self.targets]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate target_id in contract")
        if not any(t.target_class is TargetClass.REQUIRED for t in self.targets):
            raise ValueError("target contract needs at least one REQUIRED target")
        return self

    def by_class(self, cls: TargetClass) -> tuple[TargetSpec, ...]:
        return tuple(t for t in self.targets if t.target_class is cls)

    def get(self, target_id: str) -> TargetSpec | None:
        return next((t for t in self.targets if t.target_id == target_id), None)


class EffectorAssayProfile(_Frozen):
    effector: Effector
    assay: AssayType
    spacer_length: int | None = None
    #: Whether collateral activity is a hazard (cell assay) or the readout
    #: (in vitro detection). Set automatically from the assay if left None.
    collateral_is_hazard: bool | None = None

    @property
    def params(self) -> EffectorParams:
        return EFFECTORS[self.effector]

    @property
    def spacer_length_is_canonical(self) -> bool:
        """True when the chosen length is one the ortholog is published at.

        A non-canonical length is legal but unvalidated: the activity model's
        domain is derived from the canonical set **per ortholog**, so such a
        design is reported out-of-domain, and out-of-domain refuses to narrow
        the interval at all rather than merely widening it. Benefit is read at
        its lower bound, so the coverage gate cannot be cleared and the guide
        is refused rather than certified. (R3 REVIEW is what an *in-domain*
        pair with a weak calibration stratum earns; a length no model was fit
        at is the stronger statement.)

        Expressible and refused is still the right handling, and different from
        rejecting it at the contract: the design can be *stated*, so a caller
        can see what it would take -- an ortholog published at that length, or
        an amplicon target -- instead of the length being un-configurable.
        """
        assert self.spacer_length is not None
        return self.params.is_canonical_length(self.spacer_length)

    @model_validator(mode="after")
    def _resolve(self):
        p = EFFECTORS[self.effector]
        if self.spacer_length is None:
            object.__setattr__(self, "spacer_length", p.preferred_spacer_length)
        elif not p.supports_length(self.spacer_length):
            low, high = p.spacer_length_range
            raise ValueError(
                f"{self.effector.value} does not support spacer length "
                f"{self.spacer_length}; tolerated range is {low}-{high} nt "
                f"(canonical: {p.spacer_lengths})"
            )
        if self.collateral_is_hazard is None:
            hazard = self.assay is not AssayType.IN_VITRO_DETECTION
            object.__setattr__(self, "collateral_is_hazard", hazard)
        return self


class ContextScope(_Frozen):
    organism: str
    genome_build: str
    annotation_version: str
    cell_or_tissue: str | None = None
    #: Universes the search will actually cover. Adequacy is gated (fix #3).
    declared_universes: frozenset[Universe] = frozenset({Universe.CODING})

    @field_serializer("declared_universes")
    def _canonical_universes(self, universes: frozenset[Universe]) -> list[str]:
        """Sorted, so the serialisation is a function of the value alone.

        ``fingerprint()`` promises "two identical designs share a fingerprint",
        and ``json.dumps(sort_keys=True)`` orders dict *keys* -- a frozenset
        serialises as a list in iteration order. ``Universe`` is a str Enum
        whose hash is a per-process-randomised string hash, so the order, and
        therefore the SHA-256, changed between runs of the same design. Every
        real run declares all nine universes, so the fingerprint that the
        provenance ledger, the report field and every export header carry was
        unstable exactly where it is load-bearing. Sorting at serialisation
        fixes the dumped document too, not only the digest.
        """
        return sorted(u.value for u in universes)

    def missing_universes(self, assay: AssayType) -> frozenset[Universe]:
        return frozenset(ASSAY_REQUIRED_UNIVERSES[assay] - self.declared_universes)


class ResolvedDesignSpec(_Frozen):
    """The design contract. Immutable, and carried in provenance.

    Not hashable, though every other model here is. ``_Frozen`` sets
    ``frozen=True``, so pydantic *does* generate a ``__hash__`` -- and
    :class:`HazardWeights` holds ``by_class`` and ``by_biotype`` as plain
    ``dict``s, so hashing recurses into them and raises ``TypeError:
    unhashable type: 'dict'``. The ``contract`` tuple tree hashes fine; the two
    maps are the whole reason. Identity across runs is :meth:`fingerprint`,
    which is what provenance records and what a caller should compare.
    """

    spec_version: str = "cas13guide/spec/v2"
    contract: TargetContract
    profile: EffectorAssayProfile
    context: ContextScope
    hazard: HazardWeights = Field(default_factory=HazardWeights)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    budget: SearchBudget = Field(default_factory=SearchBudget)

    @property
    def spacer_length(self) -> int:
        assert self.profile.spacer_length is not None
        return self.profile.spacer_length

    @property
    def requires_discrimination(self) -> bool:
        """Allele/SNV tasks need the ratio constraint, not two thresholds."""
        return self.contract.task is Task.ALLELE

    def fingerprint(self) -> str:
        """Stable content hash. Two identical designs share a fingerprint."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
