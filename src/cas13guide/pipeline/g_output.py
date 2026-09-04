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
import statistics
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
from ..services.panel_confidence import PanelConfidence
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
    fold=None,
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
            # With the same folding engine the run used on its real guides.
            # `qc_candidate(probe, spec)` alone falls back to the stem-length
            # proxy, which catches about a third of what a real fold does -- so
            # a control could pass here and be FAILed by the QC that actually
            # ran, which is the divergence the comment above says it removed.
            # Measured over 4000 shuffles of the spacer b_qc's own docstring
            # cites: 226 of them, 5.7%.
            if not qc_candidate(probe, spec, fold).passed:
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
    # The median, not the upper-middle order statistic. For an even count
    # `coverages[n // 2]` is biased upward by half the central gap, and at
    # n = 2 it equals coverage_max exactly -- in the one field a reader
    # consults to judge whether the activity model discriminated at all.
    middle = statistics.median(coverages)
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
    #: Off-target loci whose site is a site in this run's own target. Only a
    #: search over a corpus this run did not assemble can produce one, and
    #: what it means is that the guide's intended site is being scored as a
    #: perfect off-target under an accession nothing can match.
    self_identical_loci: frozenset = frozenset()
    #: How many distinct (guide, locus) off-target pairs were found, before the
    #: explorer's row cap. `off_target_rows` holds at most fifty of them, and
    #: without this nothing downstream could tell fifty of fifty from fifty of
    #: four thousand.
    off_target_total: int = 0
    #: Why the guides that failed the gate failed it, grouped by reason code
    #: with a few worked examples. ``ranked`` deliberately holds no FAIL rows
    #: -- a refusal is not a recommendation and does not belong in a table
    #: somebody might order from -- but the consequence was that a run could
    #: report "FAIL: 41" and offer nothing else. The gate had already composed
    #: the numbers; nothing read them.
    rejected: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    #: FR6.3 -- bounded detection probability for the panel, or ``None`` when
    #: no confidence model was supplied. Kept beside the panel rather than
    #: inside it because ``Panel`` is what the optimizer returns and this is
    #: an inference *about* that panel, made from the calibration set.
    panel_confidence: "PanelConfidence | None" = None

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
                    # 0-based start of the target site on ``source_target_id``,
                    # so a report can place the guide on that target's
                    # accessibility track. Both fields, because `start` alone
                    # cannot: candidates are tiled from every REQUIRED *and
                    # DESIRED* target, so a ranked row may carry a coordinate on
                    # a record the track does not depict, and no consumer of
                    # this dict could tell.
                    "start": r.assessment.candidate.start,
                    "source_target_id": r.assessment.candidate.source_target_id,
                    "distinct_site": r.assessment.guide_id in self.distinct_sites,
                    "pfs_3prime": r.assessment.candidate.three_prime_flank,
                    "pfs_preferred": r.assessment.qc.pfs_preference == 1.0,
                    "qc_warnings": [w.value for w in r.assessment.qc.warnings],
                    # QC measures three things about every spacer and, until
                    # now, reported none of them. GC fraction in particular is
                    # the first number anybody ordering an oligo writes down,
                    # and the gate FAILs outside [0.30, 0.70] -- so where a
                    # surviving guide sits inside that band is exactly what a
                    # reader wants and could not get.
                    "gc": round(r.assessment.qc.gc, 4),
                    "max_homopolymer": r.assessment.qc.max_homopolymer,
                    # The *proxy* stem length, which is not always what gated
                    # the guide: with ViennaRNA installed the flag comes from a
                    # real fold energy instead. Named for what it is so the two
                    # cannot be confused.
                    "self_fold_stem": r.assessment.qc.self_fold_score,
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
                    # Present only when the constants were measured. This is
                    # what --kinetics-constants was advertised as unlocking,
                    # and it reached no surface a user reads.
                    "predicted_signal_nm": (
                        None if r.assessment.metrics.predicted_signal_nm is None
                        else round(r.assessment.metrics.predicted_signal_nm, 4)
                    ),
                    "score": round(r.score, 4),
                    "verdict": r.assessment.gate.verdict.value,
                    "review_tier": (
                        r.assessment.gate.review_tier.value
                        if r.assessment.gate.review_tier
                        else None
                    ),
                    "reasons": [c.value for c in r.assessment.gate.reasons],
                    # The reason code says which rule spoke; the detail says by
                    # how much. The gate composed both and only the code was
                    # ever exported.
                    "detail": list(r.assessment.gate.detail),
                    "metrics": asdict(r.assessment.metrics),
                    "rationale": list(r.rationale),
                    # The same breakdown the rationale strings render, in the
                    # form a caller can use. It was computed for every guide
                    # and reached no output at all, so the one question a
                    # weighted ranking raises -- how much of this rank is my
                    # preference rather than the evidence -- could only be
                    # answered by parsing prose.
                    "sensitivity": {
                        key: round(value, 6)
                        for key, value in sorted(r.sensitivity.items())
                    },
                }
                for r in self.ranked
            ],
            "panel": {
                "guides": [g.guide_id for g in self.panel.guides],
                "coverage": self.panel.coverage,
                "objective": round(self.panel.objective, 4),
                "guarantee": self.panel.guarantee,
                "dropped": list(self.panel.dropped),
                "confidence": (
                    None if self.panel_confidence is None
                    else self.panel_confidence.to_dict()
                ),
            },
            "controls": [asdict(c) for c in self.controls],
            "coverage_map": self.coverage_map,
            "metric_summary": {
                **asdict(self.metric_summary),
                "coverage_spread": round(self.metric_summary.coverage_spread, 6),
                "discriminated": self.metric_summary.discriminated,
            },
            "off_targets": list(self.off_target_rows),
            "off_target_total": self.off_target_total,
            "pareto_frontier": list(self.pareto_frontier),
            # Sorted, so the machine-readable report is byte-identical across
            # runs; a frozenset's iteration order is not.
            "self_identical_loci": sorted(self.self_identical_loci),
            "provenance": self.provenance,
            "validation_plan": list(self.validation_plan),
            "shortfalls": list(self.shortfalls),
            "rejected": [dict(row) for row in self.rejected],
        }


def _off_target_rows(assessments: list[GuideAssessment], limit: int = 50) -> tuple[dict, ...]:
    # One row per (guide, locus), carrying the WORST context -- which is what
    # `worst_off_target` is a max over. `off_targets()` now spans every
    # context, so a locus present in the reference and in two haplotypes would
    # otherwise take three of the fifty rows to say one thing, and a table
    # capped at fifty would fill with repeats of the same locus.
    worst_row: dict[tuple[str, str], dict[str, Any]] = {}
    for a in assessments:
        for e in a.off_targets():
            row = {
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
                    # Which world this hit was seen in. A hazard that exists
                    # only in a haplotype is still the number the gate acted
                    # on, and a reader told to "sequence the top off-target
                    # loci from the explorer table" needs to know which.
                    "context": e.context,
            }
            key = (a.guide_id, e.locus_id)
            previous = worst_row.get(key)
            if previous is None or row["hazard"] > previous["hazard"]:
                worst_row[key] = row

    rows = sorted(worst_row.values(), key=lambda r: r["hazard"], reverse=True)
    # The cap is on what a page can usefully show, not on what was found, and
    # the two were indistinguishable downstream: every surface reported
    # `len(off_targets)` as though it were the total.
    for rank, row in enumerate(rows[:limit]):
        row["of_total"] = len(rows)
        row["rank"] = rank + 1
    return tuple(rows[:limit])


def _off_target_total(assessments: list[GuideAssessment]) -> int:
    """Distinct (guide, locus) off-target pairs, before the explorer's cap."""
    return len({
        (a.guide_id, e.locus_id) for a in assessments for e in a.off_targets()
    })


def _validation_plan(
    panel: Panel,
    controls: tuple[ControlGuide, ...],
    spec: ResolvedDesignSpec,
    off_target_rows: tuple[dict, ...] = (),
) -> tuple[str, ...]:
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
    if off_target_rows:
        plan.append(
            "Sequence the top off-target loci from the explorer table; "
            "prioritize rows with high UCB(activity) whose expression is "
            "currently unknown."
        )
    else:
        # The explorer only renders when there are rows, so with none this
        # line used to send a reader to a table that was not on the page.
        # What replaces it does not let an empty table read as a clean one:
        # nothing was found because nothing was loaded to find.
        plan.append(
            "There is no off-target explorer on this page: no site was found "
            "in the sequences this run was given. That is a fact about the "
            "corpus, not about these guides -- load a background covering the "
            "universes the assay is exposed to before reading an empty table "
            "as a specificity result."
        )
    plan.append(
        "Feed every measured outcome back to the calibration service "
        "(services.benchmark.record_wetlab_result) so the bounds tighten."
    )
    return tuple(plan)


#: How many worked examples to carry per reason group. The detail strings hold
#: per-guide numbers, so grouping on them would put most guides in a group of
#: one and reproduce the wall of rows this is meant to replace.
REJECTION_EXAMPLES = 3


def _rejection_rows(assessments: list[GuideAssessment]) -> tuple[dict[str, Any], ...]:
    """Group the failed guides by what the gate said about them.

    Ordered by how many guides each reason set accounts for, then by the reason
    text, so the output is stable across runs and the dominant cause is first.
    """
    groups: dict[tuple[str, ...], list[GuideAssessment]] = {}
    for a in assessments:
        if a.gate.verdict is not Verdict.FAIL:
            continue
        groups.setdefault(tuple(c.value for c in a.gate.reasons), []).append(a)

    rows: list[dict[str, Any]] = []
    for reasons, members in groups.items():
        members.sort(key=lambda a: a.guide_id)
        rows.append({
            "reasons": list(reasons),
            "count": len(members),
            "examples": [
                {"guide_id": a.guide_id, "detail": list(a.gate.detail)}
                for a in members[:REJECTION_EXAMPLES]
            ],
        })
    rows.sort(key=lambda row: (-int(row["count"]), row["reasons"]))
    return tuple(rows)


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
    self_identical_loci: frozenset = frozenset(),
    panel_confidence: "PanelConfidence | None" = None,
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
        off_target_total=_off_target_total(assessments),
        provenance=ledger.to_dict(),
        validation_plan=_validation_plan(
            panel, controls, spec, _off_target_rows(assessments)
        ),
        pareto_frontier=pareto_frontier,
        shortfalls=shortfalls,
        rejected=_rejection_rows(assessments),
        search_bound=search_bound,
        universe_gaps=universe_gaps,
        self_identical_loci=self_identical_loci,
        panel_confidence=panel_confidence,
    )
