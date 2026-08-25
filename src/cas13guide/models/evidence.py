"""Typed objects that move between pipeline stages.

The normalization that makes the whole design explainable: evidence is held at
``guide x transcript x context`` granularity (PairEvidence) and only then
aggregated into a per-guide metric vector (GuideAssessment). Nothing collapses
to a scalar before the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contract.enums import QCFlag, ReasonCode, ReviewTier, TargetClass, Universe, Verdict
from ..metrics.bounds import Interval
from ..util.duplex import DuplexProfile


@dataclass(frozen=True, slots=True)
class Candidate:
    """A proposed spacer, before any evidence is attached."""

    guide_id: str
    spacer: str
    source_target_id: str
    #: 0-based start of the target site on the source transcript.
    start: int
    #: Bases flanking the target site, for PFS checks. None at transcript ends.
    five_prime_flank: str | None = None
    three_prime_flank: str | None = None
    #: Set when the guide deliberately spans an exon-exon junction.
    spans_junction: bool = False

    @property
    def length(self) -> int:
        return len(self.spacer)


@dataclass(frozen=True, slots=True)
class QCResult:
    guide_id: str
    passed: bool
    #: Hard incompatibilities. Non-empty means ``passed`` is False and the gate
    #: FAILs the guide.
    flags: tuple[QCFlag, ...] = ()
    gc: float = 0.0
    max_homopolymer: int = 0
    self_fold_score: float = 0.0
    manufacturability: float = 1.0
    #: Soft findings. Recorded and surfaced, but never fail a guide.
    warnings: tuple[QCFlag, ...] = ()
    #: PFS preference score in ``(0, 1]``; 1.0 when the flanking bases sit in
    #: the ortholog's preferred set. Feeds the ranking, not the gate.
    pfs_preference: float = 1.0


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One credible binding site found for a spacer, before scoring."""

    guide_id: str
    locus_id: str
    target_site: str
    start: int
    duplex: DuplexProfile
    biotype: Universe
    gene: str | None = None
    transcript: str | None = None
    #: Filled when the locus corresponds to a target named in the contract.
    target_id: str | None = None
    target_class: TargetClass | None = None
    five_prime_flank: str | None = None
    three_prime_flank: str | None = None


@dataclass(frozen=True, slots=True)
class PairEvidence:
    """Calibrated evidence for one guide against one locus in one context."""

    guide_id: str
    locus_id: str
    context: str
    hit: SearchHit
    activity: Interval
    expression: Interval
    hazard_weight: float
    accessibility: Interval | None = None
    #: Guide:target hybridization free energy, kcal/mol/nt (FR5.1). ``None``
    #: when no thermodynamics model was configured. Not an Interval and not an
    #: objective: it has an optimum rather than a direction, so it is reported
    #: and range-flagged rather than maximized.
    duplex_dg_per_nt: float | None = None
    #: False when the activity model was asked about a case outside its
    #: declared domain. Distinct from ``activity.ood``, which also covers a
    #: missing calibration stratum -- the gate reports the two separately.
    domain_ok: bool = True

    @property
    def target_class(self) -> TargetClass | None:
        return self.hit.target_class

    @property
    def is_intended(self) -> bool:
        return self.target_class in (TargetClass.REQUIRED, TargetClass.DESIRED)

    @property
    def ood(self) -> bool:
        return self.activity.ood

    @property
    def expression_unknown(self) -> bool:
        return self.expression.imputed


@dataclass(frozen=True, slots=True)
class MetricVector:
    """The six named metrics. Deliberately not reducible to one number."""

    required_coverage: float
    desired_utility: float
    worst_off_target: float
    cumulative_burden: float
    robustness: float
    manufacturability: float
    #: None when the task does not require allele discrimination.
    discrimination_margin: float | None = None
    #: Ortholog PFS *preference* in ``(0, 1]``. A soft efficacy signal, so it
    #: ranks rather than gates -- see ``contract/effectors.py``.
    pfs_preference: float = 1.0
    #: Mean unpaired probability at the best intended site, or None when no
    #: accessibility model was supplied. Monotone -- a more open site is a
    #: better site -- so unlike duplex free energy it can be a Pareto
    #: objective. Free energy has an *optimum* rather than a direction, so it
    #: is reported and flagged instead of ranked.
    accessibility: float | None = None
    #: Duplex free energy at the best intended site, kcal/mol/nt. Deliberately
    #: absent from :meth:`objectives`: see ``services/thermodynamics.py``.
    duplex_dg_per_nt: float | None = None
    #: Relative predicted fluorescence signal in ``[0, 1]`` (FR6.2). Present
    #: only when collateral activity is the assay's *readout*; in a knockdown
    #: assay collateral is a hazard, and ranking guides by how much of it they
    #: produce would be exactly backwards.
    predicted_signal: float | None = None
    #: Whether :attr:`predicted_signal` earns a place among the objectives.
    #:
    #: The signal model is driven by ``activity_lower * accessibility``, and
    #: both are already objectives here. In the ODE's linear regime -- which is
    #: where the shipped placeholder constants sit -- its output is exactly
    #: ``k * activity_lower * accessibility``, so it cannot break a Pareto tie
    #: its own inputs did not already break, and in the weighted score it
    #: silently re-weights them at 2.5. Measured with those constants the ratio
    #: varies by 0.159% across the unit square.
    #:
    #: So it stays a *reported* column and leaves the objective set. The
    #: pipeline probes the model rather than assuming
    #: (``CollateralKineticsModel.adds_information``), and the axis returns on
    #: its own once constants that saturate the ODE are supplied.
    predicted_signal_ranks: bool = True

    def objectives(self) -> dict[str, float]:
        """Metrics oriented so that larger is always better, for Pareto."""
        out = {
            "required_coverage": self.required_coverage,
            "desired_utility": self.desired_utility,
            "worst_off_target": -self.worst_off_target,
            "cumulative_burden": -self.cumulative_burden,
            "robustness": self.robustness,
            "manufacturability": self.manufacturability,
            "pfs_preference": self.pfs_preference,
        }
        if self.discrimination_margin is not None:
            out["discrimination_margin"] = self.discrimination_margin
        if self.accessibility is not None:
            out["accessibility"] = self.accessibility
        if self.predicted_signal is not None and self.predicted_signal_ranks:
            out["predicted_signal"] = self.predicted_signal
        return out


@dataclass(frozen=True, slots=True)
class GateResult:
    verdict: Verdict
    reasons: tuple[ReasonCode, ...]
    review_tier: ReviewTier | None = None
    detail: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GuideAssessment:
    """A guide, its metric vector, its evidence flags and its verdict."""

    guide_id: str
    candidate: Candidate
    qc: QCResult
    metrics: MetricVector
    gate: GateResult
    evidence: tuple[PairEvidence, ...] = ()

    @property
    def spacer(self) -> str:
        return self.candidate.spacer

    @property
    def passed(self) -> bool:
        return self.gate.verdict is Verdict.PASS

    def off_targets(self) -> tuple[PairEvidence, ...]:
        return tuple(e for e in self.evidence if not e.is_intended)
