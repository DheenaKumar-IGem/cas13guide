"""Typed objects that move between pipeline stages.

The normalization that makes the whole design explainable: evidence is held at
``guide x transcript x context`` granularity (PairEvidence) and only then
aggregated into a per-guide metric vector (GuideAssessment). Nothing collapses
to a scalar before the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contract.enums import QCFlag, ReasonCode, ReviewTier, TargetClass, Universe, Verdict
from ..metrics.bounds import Interval, benefit_lower
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
    #: Cleaved reporter in nM at the best intended site, and only when the
    #: kinetic constants were *measured* rather than shipped.
    #:
    #: ``kinetics.predict`` computes it on the same call the relative score
    #: comes from, and it was thrown away: ``absolute_signal()`` is a library
    #: method nothing in the pipeline called, so ``--kinetics-constants``,
    #: advertised as unlocking an absolute figure, reached no surface a user
    #: reads. It stays ``None`` on provisional constants for the reason
    #: ``absolute_signal`` refuses there -- real units derived from placeholder
    #: rates are a fabricated measurement wearing a unit label, and downstream
    #: nothing tells it from a measured one.
    #:
    #: Reported, never ranked on: it is ``predicted_signal`` rescaled, and an
    #: axis that is a rescaling of another axis does not earn a vote.
    predicted_signal_nm: float | None = None
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
    #: The **reference** context's evidence. Deliberately one context: a
    #: guide's site coordinate is a position on the reference, and averaging
    #: that across haplotypes would be meaningless.
    evidence: tuple[PairEvidence, ...] = ()
    #: Every context's evidence, which is what the metrics are aggregated over
    #: and what the gate was handed.
    #:
    #: Both exist because they answer different questions. `worst_off_target`
    #: is a max across contexts, so a hazard living only in a variant world
    #: drove the metric and the verdict while `off_targets()` -- reading the
    #: reference alone -- showed nothing: a guide could FAIL for a hit the
    #: off-target explorer did not list, under a validation plan that says
    #: "sequence the top off-target loci from the explorer table".
    context_evidence: tuple[PairEvidence, ...] = ()

    def binding_sites(self) -> dict[str, int]:
        """Where this guide binds, per intended target: one coordinate each.

        Two places need this and they used to derive it separately.
        ``f_decide.distinct_site_ids`` read every *intended* hit and took the
        site with the highest calibrated lower bound; ``optimizer._profile``,
        enforcing panel spacing, read REQUIRED hits only and recorded a site
        only when that bound was strictly greater than zero. So the ranked list
        and the panel selector answered two similar questions, under a comment
        in ``f_decide`` claiming they answered one.

        The gap had two halves, and both are reachable from an ordinary run:

        * an overlap on a DESIRED target was invisible to the optimizer, so a
          panel could hold two guides binding the same site of a desired
          transcript while the report beside it marked them as one site;
        * an overlap at a site whose lower bound is exactly 0.0 was invisible
          too -- and conformal intervals are clamped to ``[0, 1]``, so 0.0 is
          the *ordinary* lower bound for a weak hit, not an edge case.

        Either one breaks the promise in the optimizer's own docstring: "two
        guides tiling the same target must not sit within ``min_spacing`` nt of
        each other, so a panel cannot be three shifted copies of one site."

        Highest lower bound rather than lowest coordinate: filing a guide under
        a site it barely binds compares it at a near-zero-activity coordinate
        instead of its real one. At the shipped 4-mismatch wobble-aware
        envelope a 22-mer picks up a second hit on its own transcript often
        enough to matter -- 61 of 1650 guides on one 2 kb target -- and
        measured, 3 of 68 marked rows became single-base shifts of a row
        already listed, under the line saying shifted copies are hidden.

        The tiled coordinate is the fallback, used only when a guide has no
        intended evidence at all, which is then the only information there is.
        """
        sites: dict[str, int] = {}
        best: dict[str, float] = {}
        for evidence in self.evidence:
            target_id = evidence.hit.target_id
            if not evidence.is_intended or not target_id:
                continue
            lower = benefit_lower(evidence.activity)
            if target_id not in best or lower > best[target_id]:
                best[target_id] = lower
                sites[target_id] = evidence.hit.start
        if not sites:
            return {self.candidate.source_target_id: self.candidate.start}
        return sites

    @property
    def spacer(self) -> str:
        return self.candidate.spacer

    @property
    def passed(self) -> bool:
        return self.gate.verdict is Verdict.PASS

    def off_targets(self) -> tuple[PairEvidence, ...]:
        """Every off-target hit behind this guide's hazard, in any context.

        Falls back to the reference evidence for an assessment built without
        the full set, so an older caller keeps working.
        """
        source = self.context_evidence or self.evidence
        return tuple(e for e in source if not e.is_intended)
