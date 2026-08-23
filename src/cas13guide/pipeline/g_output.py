"""Stage G -- transparent output.

Everything a reviewer needs to disagree with the result: ranked guides with
reason codes, the isoform/EEJ coverage map, an off-target explorer, matched
control guides, the full provenance record and a validation plan.

Control guides are generated here as first-class output (fix #10). A knockdown
experiment without a GC- and structure-matched non-targeting control cannot be
interpreted, so the design produces them with the same machinery -- including
running them through the same search to *prove* they are non-targeting.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from ..contract.enums import TargetClass, Verdict
from ..contract.spec import ResolvedDesignSpec
from ..metrics.aggregate import pair_hazard
from ..models.evidence import Candidate, GuideAssessment
from ..services.indexes import ExhaustivenessBound, SearchIndex
from ..services.provenance import ProvenanceLedger
from ..util.seq import gc_fraction
from .b_qc import qc_candidate
from .f_decide import RankedGuide, distinct_site_ids, review_rate
from .optimizer import Panel


def _has_any_hit(index: SearchIndex, guide_id: str, spacer: str, depth: int) -> bool:
    """Does ``spacer`` bind anywhere within ``depth`` mismatches?

    ``generate_controls`` consumes nothing but this yes/no, yet it used to pay
    for a full ``index.search`` -- every hit materialized into a ``SearchHit``
    with a ``profile_duplex`` -- for each of up to ``attempts`` scrambles per
    panel member. At the default envelope a random scramble almost always hits
    something, so nearly every one of those searches was enumerated in full and
    then discarded on ``if not result.hits``.

    An index may offer ``any_hit`` to answer this without enumerating; the
    fallback keeps the ``SearchIndex`` Protocol unchanged, so third-party
    implementations still work untouched.
    """
    probe = getattr(index, "any_hit", None)
    if probe is not None:
        return bool(probe(spacer, depth))
    return bool(index.search(guide_id, spacer, depth).hits)


@dataclass(frozen=True, slots=True)
class ControlGuide:
    """A non-targeting control, matched to the panel and verified."""

    guide_id: str
    spacer: str
    kind: str
    gc: float
    matched_to: str
    verified_no_hits: bool
    note: str = ""


def generate_controls(
    panel: Panel,
    spec: ResolvedDesignSpec,
    index: SearchIndex,
    seed: int = 0,
    attempts: int = 200,
    envelope: int | None = None,
) -> tuple[ControlGuide, ...]:
    """GC- and composition-matched non-targeting controls, search-verified.

    Each control is a shuffle of a real panel spacer, so GC content and
    nucleotide composition match exactly; it is then searched against the same
    universes and kept only if it has no credible binding site.

    "Credible" means the *same* envelope the pipeline used to judge the real
    guides -- ``spec.thresholds.required_search_mismatches``, or the escalated
    value the run actually reached. Verifying controls at a shallower depth
    than the guides would ship a control the pipeline itself would flag as an
    off-target.
    """
    depth = envelope if envelope is not None else spec.thresholds.required_search_mismatches
    rng = random.Random(seed)
    controls: list[ControlGuide] = []

    for member in panel.guides:
        spacer = member.spacer
        target_gc = gc_fraction(spacer)
        chosen: str | None = None
        for _ in range(attempts):
            shuffled = list(spacer)
            rng.shuffle(shuffled)
            candidate = "".join(shuffled)
            if candidate == spacer:
                continue
            # Stage B's own QC, not a hand-copied subset of it. The old inline
            # check tested a hardcoded homopolymer limit of 4 and the poly-U
            # terminator and nothing else, so a control could ship carrying a
            # GC violation, a self-folding stem or a direct-repeat interaction
            # -- i.e. a control the pipeline's own gate would FAIL as
            # deterministically incompatible.
            probe = Candidate(
                guide_id=f"ctrl_{member.guide_id}",
                spacer=candidate,
                source_target_id=member.candidate.source_target_id,
                start=member.candidate.start,
                five_prime_flank=member.candidate.five_prime_flank,
                three_prime_flank=member.candidate.three_prime_flank,
            )
            if not qc_candidate(probe, spec).passed:
                continue
            if _has_any_hit(index, f"ctrl_{member.guide_id}", candidate, depth):
                continue
            chosen = candidate
            break
        if chosen is None:
            controls.append(
                ControlGuide(
                    guide_id=f"ctrl_{member.guide_id}",
                    spacer="",
                    kind="non_targeting",
                    gc=target_gc,
                    matched_to=member.guide_id,
                    verified_no_hits=False,
                    note=(
                        f"no composition-matched scramble was free of hits at "
                        f"<={depth}mm; widen the pool or relax the envelope"
                    ),
                )
            )
        else:
            controls.append(
                ControlGuide(
                    guide_id=f"ctrl_{member.guide_id}",
                    spacer=chosen,
                    kind="non_targeting",
                    gc=gc_fraction(chosen),
                    matched_to=member.guide_id,
                    verified_no_hits=True,
                )
            )
    return tuple(controls)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Distribution of key metrics across *every* assessed guide.

    Reported separately from the ranking, which only contains gate survivors.
    When a run produces zero PASS the first question is whether the model
    discriminated between guides at all or handed back one flat number, and
    that cannot be answered from the ranked list -- it is empty.
    """

    n_assessed: int
    coverage_min: float
    coverage_median: float
    coverage_max: float
    distinct_coverage_values: int
    worst_off_target_max: float

    @property
    def coverage_spread(self) -> float:
        return self.coverage_max - self.coverage_min

    @property
    def discriminated(self) -> bool:
        """False when every guide received effectively the same score."""
        return self.distinct_coverage_values > 1 and self.coverage_spread > 1e-6


def summarize_metrics(assessments: list[GuideAssessment]) -> MetricSummary:
    if not assessments:
        return MetricSummary(0, 0.0, 0.0, 0.0, 0, 0.0)
    coverages = sorted(a.metrics.required_coverage for a in assessments)
    middle = coverages[len(coverages) // 2]
    return MetricSummary(
        n_assessed=len(assessments),
        coverage_min=coverages[0],
        coverage_median=middle,
        coverage_max=coverages[-1],
        distinct_coverage_values=len({round(c, 9) for c in coverages}),
        worst_off_target_max=max(a.metrics.worst_off_target for a in assessments),
    )


@dataclass(frozen=True, slots=True)
class DesignReport:
    spec_fingerprint: str
    verdict_counts: dict[str, int]
    review_rate: float
    ranked: tuple[RankedGuide, ...]
    #: Guide ids that are the best-ranked representative of a distinct target
    #: site. The rest are shifted copies of a site already listed above them.
    distinct_sites: frozenset[str]
    panel: Panel
    controls: tuple[ControlGuide, ...]
    coverage_map: dict[str, float]
    metric_summary: MetricSummary
    off_target_rows: tuple[dict[str, Any], ...]
    provenance: dict[str, Any]
    validation_plan: tuple[str, ...]
    #: Guide ids on the non-dominated frontier -- the trade-off set a human
    #: should see, distinct from the scalar ranking below it.
    pareto_frontier: tuple[str, ...] = field(default_factory=tuple)
    shortfalls: tuple[str, ...] = field(default_factory=tuple)
    #: The weakest bound the search actually proved, and the universes the assay
    #: needed but the search did not cover. Carried structurally rather than only
    #: as the prose claim in the ledger, so a caller reasoning about scope does
    #: not have to parse a sentence back into facts.
    search_bound: "ExhaustivenessBound | None" = None
    universe_gaps: frozenset = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_fingerprint": self.spec_fingerprint,
            "verdict_counts": self.verdict_counts,
            "review_rate": round(self.review_rate, 4),
            "distinct_site_count": len(self.distinct_sites),
            "ranked": [
                {
                    "rank": r.rank,
                    "guide_id": r.assessment.guide_id,
                    "spacer": r.assessment.spacer,
                    "spans_junction": r.assessment.candidate.spans_junction,
                    # 0-based start of the target site on its source target,
                    # so a report can place the guide on an accessibility track.
                    "start": r.assessment.candidate.start,
                    "distinct_site": r.assessment.guide_id in self.distinct_sites,
                    "pfs_3prime": r.assessment.candidate.three_prime_flank,
                    "pfs_preferred": r.assessment.qc.pfs_preference == 1.0,
                    "qc_warnings": [w.value for w in r.assessment.qc.warnings],
                    "accessibility": (
                        None if r.assessment.metrics.accessibility is None
                        else round(r.assessment.metrics.accessibility, 4)
                    ),
                    "duplex_dg_per_nt": (
                        None if r.assessment.metrics.duplex_dg_per_nt is None
                        else round(r.assessment.metrics.duplex_dg_per_nt, 3)
                    ),
                    "predicted_signal": (
                        None if r.assessment.metrics.predicted_signal is None
                        else round(r.assessment.metrics.predicted_signal, 4)
                    ),
                    "score": round(r.score, 4),
                    "verdict": r.assessment.gate.verdict.value,
                    "review_tier": (
                        r.assessment.gate.review_tier.value
                        if r.assessment.gate.review_tier
                        else None
                    ),
                    "reasons": [c.value for c in r.assessment.gate.reasons],
                    "metrics": asdict(r.assessment.metrics),
                    "rationale": list(r.rationale),
                }
                for r in self.ranked
            ],
            "panel": {
                "guides": [g.guide_id for g in self.panel.guides],
                "coverage": self.panel.coverage,
                "objective": round(self.panel.objective, 4),
                "guarantee": self.panel.guarantee,
                "dropped": list(self.panel.dropped),
            },
            "controls": [asdict(c) for c in self.controls],
            "coverage_map": self.coverage_map,
            "metric_summary": {
                **asdict(self.metric_summary),
                "coverage_spread": round(self.metric_summary.coverage_spread, 6),
                "discriminated": self.metric_summary.discriminated,
            },
            "off_targets": list(self.off_target_rows),
            "pareto_frontier": list(self.pareto_frontier),
            "provenance": self.provenance,
            "validation_plan": list(self.validation_plan),
            "shortfalls": list(self.shortfalls),
        }


def _off_target_rows(assessments: list[GuideAssessment], limit: int = 50) -> tuple[dict, ...]:
    rows: list[dict[str, Any]] = []
    for a in assessments:
        for e in a.off_targets():
            rows.append(
                {
                    "guide_id": a.guide_id,
                    "locus": e.locus_id,
                    "gene": e.hit.gene,
                    "biotype": e.hit.biotype.value,
                    "target_class": e.target_class.value if e.target_class else "unannotated",
                    "mismatches": e.hit.duplex.n_mismatch,
                    "wobbles": e.hit.duplex.n_wobble,
                    "cigar": e.hit.duplex.cigar,
                    "ucb_activity": round(e.activity.upper, 4),
                    "ucb_expression": round(e.expression.upper, 4),
                    "hazard_weight": e.hazard_weight,
                    # Same code path as worst_off_target / cumulative_burden, so the column a
                    # reviewer eyeballs cannot drift from the metric it reproduces.
                    "hazard": round(pair_hazard(e), 4),
                    "expression_unknown": e.expression_unknown,
                    "ood": e.ood,
                }
            )
    rows.sort(key=lambda r: r["hazard"], reverse=True)
    return tuple(rows[:limit])


def _validation_plan(panel: Panel, controls: tuple[ControlGuide, ...], spec: ResolvedDesignSpec) -> tuple[str, ...]:
    plan = [
        f"Transfect the {panel.size}-guide panel alongside "
        f"{len(controls)} composition-matched non-targeting controls.",
        "Quantify knockdown of every REQUIRED target by RT-qPCR "
        "(primers spanning a region no panel guide tiles).",
    ]
    if spec.profile.collateral_is_hazard:
        plan.append(
            "Assay collateral activity directly: total RNA integrity plus a "
            "high-abundance bystander transcript. Collateral scales with "
            "target abundance, not with off-target homology."
        )
    plan.append(
        "Sequence the top off-target loci from the explorer table; prioritize "
        "rows with high UCB(activity) whose expression is currently unknown."
    )
    plan.append(
        "Feed every measured outcome back to the calibration service "
        "(services.benchmark.record_wetlab_result) so the bounds tighten."
    )
    return tuple(plan)


def build_report(
    assessments: list[GuideAssessment],
    ranked: list[RankedGuide],
    panel: Panel,
    controls: tuple[ControlGuide, ...],
    spec: ResolvedDesignSpec,
    ledger: ProvenanceLedger,
    shortfalls: tuple[str, ...] = (),
    pareto_frontier: tuple[str, ...] = (),
    search_bound: ExhaustivenessBound | None = None,
    universe_gaps: frozenset = frozenset(),
) -> DesignReport:
    counts = {v.value: 0 for v in Verdict}
    for a in assessments:
        counts[a.gate.verdict.value] += 1

    # Marked, never filtered: see f_decide.distinct_site_ids for why the raw
    # ranking is misleading and why dropping rows would be the wrong fix.
    distinct = distinct_site_ids(list(ranked), spec.spacer_length)

    coverage_map = {
        t.target_id: panel.coverage.get(t.target_id, 0.0)
        for t in spec.contract.by_class(TargetClass.REQUIRED)
    }

    return DesignReport(
        spec_fingerprint=spec.fingerprint(),
        verdict_counts=counts,
        review_rate=review_rate(assessments),
        ranked=tuple(ranked),
        distinct_sites=distinct,
        panel=panel,
        controls=controls,
        coverage_map=coverage_map,
        metric_summary=summarize_metrics(assessments),
        off_target_rows=_off_target_rows(assessments),
        provenance=ledger.to_dict(),
        validation_plan=_validation_plan(panel, controls, spec),
        pareto_frontier=pareto_frontier,
        shortfalls=shortfalls,
        search_bound=search_bound,
        universe_gaps=universe_gaps,
    )
