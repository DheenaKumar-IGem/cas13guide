"""Closed vocabularies shared across the whole system.

Every enum here is part of the contract: values are persisted in the
provenance ledger and referenced by reason codes, so renaming a member is a
breaking change.
"""

from __future__ import annotations

from enum import Enum


class Task(str, Enum):
    KNOCKDOWN = "knockdown"
    ISOFORM = "isoform"
    ALLELE = "allele"
    ANTIVIRAL = "antiviral"
    DETECTION = "detection"
    EDITING = "editing"


class TargetClass(str, Enum):
    """Intent classes. Ordering is meaningful: hazard rises toward FORBIDDEN."""

    REQUIRED = "required"
    DESIRED = "desired"
    TOLERATED = "tolerated"
    AVOID = "avoid"
    FORBIDDEN = "forbidden"


class Effector(str, Enum):
    RFX_CAS13D = "RfxCas13d"
    LWA_CAS13A = "LwaCas13a"
    PSP_CAS13B = "PspCas13b"


class AssayType(str, Enum):
    """Assay determines which sequence universes the guide is exposed to."""

    CELL_CULTURE_KNOCKDOWN = "cell_culture_knockdown"
    IN_VIVO_KNOCKDOWN = "in_vivo_knockdown"
    ANTIVIRAL = "antiviral"
    IN_VITRO_DETECTION = "in_vitro_detection"


class Universe(str, Enum):
    """Sequence universes that a search may or may not cover."""

    CODING = "coding"
    NCRNA = "ncrna"
    PRE_MRNA = "pre_mrna"
    CIRCRNA = "circrna"
    REPEAT = "repeat"
    MITO = "mito"
    RRNA = "rrna"
    VIRAL = "viral"
    SYNTHETIC = "synthetic"


#: Universes a given assay is genuinely exposed to. Missing one of these does
#: not produce a warning -- it produces a suspiciously *clean* off-target
#: report, which is why universe adequacy is a gate condition (fix #3).
ASSAY_REQUIRED_UNIVERSES: dict[AssayType, frozenset[Universe]] = {
    AssayType.CELL_CULTURE_KNOCKDOWN: frozenset(
        {Universe.CODING, Universe.NCRNA, Universe.PRE_MRNA, Universe.RRNA, Universe.MITO}
    ),
    AssayType.IN_VIVO_KNOCKDOWN: frozenset(
        {
            Universe.CODING,
            Universe.NCRNA,
            Universe.PRE_MRNA,
            Universe.RRNA,
            Universe.MITO,
            Universe.REPEAT,
        }
    ),
    AssayType.ANTIVIRAL: frozenset(
        {Universe.CODING, Universe.NCRNA, Universe.PRE_MRNA, Universe.VIRAL, Universe.RRNA}
    ),
    AssayType.IN_VITRO_DETECTION: frozenset({Universe.SYNTHETIC, Universe.VIRAL}),
}


class Verdict(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


class ReviewTier(str, Enum):
    """Graded REVIEW (fix #12) so the gate cannot collapse to click-through."""

    R1 = "R1"  # borderline metric
    R2 = "R2"  # evidence gap
    R3 = "R3"  # out of domain / unsupported


#: Severity ordering used when several findings apply to one guide.
SEVERITY: dict[str, int] = {
    Verdict.PASS.value: 0,
    ReviewTier.R1.value: 1,
    ReviewTier.R2.value: 2,
    ReviewTier.R3.value: 3,
    Verdict.FAIL.value: 4,
}


class ReasonCode(str, Enum):
    """Machine-readable justification attached to every verdict.

    No guide ever leaves the gate without at least one of these.
    """

    # --- FAIL ---
    DETERMINISTIC_INCOMPATIBLE = "deterministic_incompatible"
    REQUIRED_TARGET_MISSED = "required_target_missed"
    CREDIBLE_FORBIDDEN_HIT = "credible_forbidden_hit"
    UNIVERSE_INADEQUATE = "universe_inadequate"
    DISCRIMINATION_MARGIN_INSUFFICIENT = "discrimination_margin_insufficient"
    HAZARD_BUDGET_EXCEEDED = "hazard_budget_exceeded"

    # --- REVIEW ---
    MODEL_OUT_OF_DOMAIN = "model_out_of_domain"
    CALIBRATION_UNSUPPORTED = "calibration_unsupported"
    SEARCH_NOT_EXHAUSTIVE = "search_not_exhaustive"
    EXPRESSION_UNKNOWN = "expression_unknown"
    BORDERLINE_OFF_TARGET = "borderline_off_target"
    BORDERLINE_COVERAGE = "borderline_coverage"
    TARGET_INACCESSIBLE = "target_inaccessible"
    DUPLEX_ENERGY_OUT_OF_RANGE = "duplex_energy_out_of_range"

    # --- PASS ---
    CONSTRAINTS_SATISFIED = "constraints_satisfied"

    # --- pipeline shortfalls (control flow, fix #5) ---
    BUDGET_EXHAUSTED_NO_FEASIBLE_PANEL = "budget_exhausted_no_feasible_panel"
    BUDGET_EXHAUSTED_SEARCH = "budget_exhausted_search"


class QCFlag(str, Enum):
    """Deterministic QC findings.

    Members appear in :attr:`QCResult.flags` (hard incompatibilities, which the
    gate turns into a FAIL) or :attr:`QCResult.warnings` (soft findings, which
    only deprioritize). The soft members are listed in :data:`SOFT_QC_FLAGS` --
    named there rather than restated here, because this sentence said
    "``PFS_DISFAVOURED`` is the only soft member" from the commit that added
    the second one.
    """

    GC_OUT_OF_RANGE = "gc_out_of_range"
    HOMOPOLYMER_RUN = "homopolymer_run"
    POLY_U_TERMINATOR = "poly_u_terminator"
    SPACER_SELF_FOLDING = "spacer_self_folding"
    DR_SPACER_INTERACTION = "dr_spacer_interaction"
    PFS_INCOMPATIBLE = "pfs_incompatible"
    #: Soft: the flanking base is outside the ortholog's preferred set. Lowers
    #: the ranking, never fails the guide.
    PFS_DISFAVOURED = "pfs_disfavoured"
    #: Soft: a spacer length the ortholog tolerates but was not published at.
    #: Usable as a *candidate*; the activity model reports it out-of-domain,
    #: which refuses to narrow its interval, so the gate then refuses it.
    SPACER_LENGTH_NONCANONICAL = "spacer_length_noncanonical"
    SPACER_LENGTH_UNSUPPORTED = "spacer_length_unsupported"
    #: Emitted by neither list today: no QC check raises it. Kept because the
    #: manufacturability score exists and a synthesis rule is the obvious thing
    #: to route through it.
    SYNTHESIS_CONSTRAINT = "synthesis_constraint"


#: The soft members: findings that only ever reach :attr:`QCResult.warnings`
#: and never fail a guide. A single source of truth rather than a sentence in
#: the class docstring, which went stale in the same commit that added the
#: second member to this set.
SOFT_QC_FLAGS: frozenset[QCFlag] = frozenset(
    {QCFlag.PFS_DISFAVOURED, QCFlag.SPACER_LENGTH_NONCANONICAL}
)
