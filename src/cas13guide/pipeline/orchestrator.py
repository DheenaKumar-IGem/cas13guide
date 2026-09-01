"""Control flow (fix #5) -- the pipeline is not a straight line.

Three decisions send work backwards, each with a stopping rule:

* **C self-loop** -- a suspicious number of zero-hit candidates means the
  mismatch envelope was too tight, so the search escalates and reruns.
* **G -> A** -- when no feasible panel covers every REQUIRED target, control
  returns to candidate design to re-tile more finely.
* **wet-lab -> calibration** -- handled out of band by
  :func:`cas13guide.services.benchmark.record_wetlab_result`, not by this
  loop, because feedback belongs to the calibration service rather than the
  design spec.

Every loop carries an iteration budget. On exhaustion the run returns its best
partial result with an explicit shortfall reason code -- it never spins, and
it never silently reports a truncated search as a complete one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..contract.enums import ReasonCode, TargetClass, Universe, Verdict
from ..contract.spec import ResolvedDesignSpec
from ..models.evidence import Candidate, GuideAssessment, PairEvidence, QCResult
from ..services.calibration import CalibrationService
from ..services.context import ExpressionSource, StaticExpressionSource
from ..services.indexes import ExhaustivenessBound, SearchIndex, is_weaker
from ..services.panel_confidence import BayesianPanelConfidence, PanelDependence
from ..services.provenance import ProvenanceLedger
from ..services.registry import ModelRegistry
from ..services.transcriptome import Transcriptome
from .a_candidates import design_candidates
from .b_qc import MAX_SELF_STEM, SELF_FOLD_MAX_DG, run_qc
from .c_search import run_search
from .d_pair_evidence import build_pair_evidence
from .e_aggregate import assess_guides
from .f_decide import RankedGuide, pareto_front, rank_by_preference
from .g_output import DesignReport, build_report, generate_controls
from .optimizer import Panel, select_panel

#: Fraction of candidates with no hits at all that makes the envelope suspect.
ZERO_HIT_SUSPICION = 0.5


@dataclass(slots=True)
class DesignContext:
    """One evaluated biological context (reference, a haplotype, a cell line)."""

    name: str
    transcriptome: Transcriptome
    index: SearchIndex
    locus_to_target: dict[str, str]
    expression: ExpressionSource | None = None
    #: Why this index was chosen. Recorded in provenance; both indexes are
    #: exhaustive, so it explains speed, never results.
    index_rationale: str = "index supplied by caller"

    @classmethod
    def auto(
        cls,
        name: str,
        transcriptome: Transcriptome,
        universes: frozenset[Universe],
        locus_to_target: dict[str, str],
        expected_queries: int = 1000,
        expression: ExpressionSource | None = None,
    ) -> "DesignContext":
        """Build a context with the index that fits this corpus.

        The FM-index is markedly *slower* than the linear scanner below a few
        megabytes of searchable sequence, so picking one unconditionally is
        wrong half the time. See
        :mod:`cas13guide.services.index_selection`.
        """
        from ..services.index_selection import choose_index

        choice = choose_index(transcriptome, universes, expected_queries)
        return cls(
            name=name,
            transcriptome=transcriptome,
            index=choice.index,
            locus_to_target=locus_to_target,
            expression=expression,
            index_rationale=choice.rationale,
        )

    def expression_source(self) -> ExpressionSource:
        if self.expression is not None:
            return self.expression
        return StaticExpressionSource(
            {k: v.expression for k, v in self.transcriptome.loci.items()}
        )


@dataclass(slots=True)
class RunTelemetry:
    """What the control flow actually did -- reported, never hidden."""

    search_escalations: int = 0
    panel_retiles: int = 0
    final_envelope: int = 0
    final_step: int = 1
    shortfalls: list[str] = field(default_factory=list)
    reason_codes: list[ReasonCode] = field(default_factory=list)


#: Bound ordering now has one definition, in ``services.indexes`` -- the same
#: rule ``c_search`` uses when folding bounds across candidates.
_is_weaker = is_weaker


def _panel_feasible(panel: Panel, spec: ResolvedDesignSpec) -> bool:
    required = spec.contract.by_class(TargetClass.REQUIRED)
    if not required:
        return True
    floor = spec.thresholds.required_coverage_min
    return all(panel.coverage.get(t.target_id, 0.0) >= floor for t in required)


def _self_fold_energy():
    """ViennaRNA MFE of a single sequence, or ``None`` if it is not installed.

    QC's built-in stem count is a proxy for exactly this number and a poor one
    (28.9% recall, measured). When the real thing is available it is used, and
    the run records which -- an unbuyable-back FAIL should not rest on a guess
    when thermodynamics is one import away.
    """
    try:
        import RNA
    except ImportError:
        return None

    def fold(sequence: str) -> float:
        return float(RNA.fold(sequence.upper().replace("T", "U"))[1])

    return fold


def run_design(
    spec: ResolvedDesignSpec,
    contexts: list[DesignContext],
    registry: ModelRegistry,
    calibration: CalibrationService,
    panel_size: int = 3,
    initial_step: int = 1,
    preference_weights: dict[str, float] | None = None,
    on_event: Callable[[str], None] | None = None,
    panel_dependence: "PanelDependence | None" = None,
) -> DesignReport:
    """Run the full pipeline, backtracking within budget."""
    if not contexts:
        raise ValueError("at least one context is required")
    self_fold = _self_fold_energy()
    reference = next((c for c in contexts if c.name == "reference"), contexts[0])

    ledger = ProvenanceLedger(spec.fingerprint())
    ledger.record("spec", "ResolvedDesignSpec", spec.spec_version, fingerprint=spec.fingerprint())
    ledger.record("hazard_weights", "HazardWeights", spec.hazard.version)
    ledger.record(
        "calibration", "CalibrationService", calibration.version,
        strata=calibration.describe(),
        # "unstated" rather than silence: a caller who built their own service
        # is marked as unrecorded rather than certified by omission.
        source=getattr(calibration, "source", None) or "unstated",
    )
    for name, version in registry.versions().items():
        ledger.record("model", name, version)
    # Two comments -- this module's `_self_fold_energy` and b_qc's MAX_SELF_STEM
    # block -- promised that "the run says which one it used". Nothing did. The
    # choice decides SPACER_SELF_FOLDING, which is a hard FAIL nobody can buy
    # back, and the two criteria disagree: on a 400 nt target the proxy and the
    # free energy delete different guides (109 vs 102 FAILs, measured), while
    # the two reports were identical in every field describing the method.
    ledger.record(
        "model", "qc_self_fold",
        "qc/vienna-mfe/v1" if self_fold is not None else "qc/stem-proxy/v1",
        criterion=(
            f"dG <= {SELF_FOLD_MAX_DG} kcal/mol"
            if self_fold is not None
            else f"longest self-complement > {MAX_SELF_STEM} nt (28.9% recall)"
        ),
    )
    for ctx in contexts:
        ledger.record("transcriptome", ctx.name, ctx.transcriptome.version,
                      loci=len(ctx.transcriptome), index=ctx.index.version,
                      index_choice=ctx.index_rationale)

    # An intended target shorter than the window is dropped by
    # `design_candidates` -- correctly, since it cannot be tiled -- and used to
    # be dropped in silence. The run then reported only "no feasible panel
    # covers required target(s): t1", which reads as "t1 has no usable guides"
    # rather than "t1 was never looked at", and on a multi-record FASTA it
    # shipped a full export of the *other* record's guides while saying so.
    # `designability()` produces exactly this explanation, and only the preset
    # path with no explicit --spacer-length ever asked it.
    untileable = [
        target.target_id
        for target in spec.contract.targets
        if target.target_class in (TargetClass.REQUIRED, TargetClass.DESIRED)
        and len(target.sequence) < spec.spacer_length
    ]

    telemetry = RunTelemetry(final_envelope=spec.thresholds.required_search_mismatches)
    if untileable:
        telemetry.shortfalls.append(
            f"{', '.join(untileable)} shorter than the {spec.spacer_length} nt "
            "spacer, so no window fits and they were never tiled. Use a shorter "
            "spacer, or design against an amplicon."
        )
        telemetry.reason_codes.append(ReasonCode.TARGET_TOO_SHORT_TO_TILE)
    envelope = spec.thresholds.required_search_mismatches
    step = initial_step

    def emit(message: str) -> None:
        if on_event:
            on_event(message)

    assessments: list[GuideAssessment] = []
    panel = Panel(guides=(), coverage={}, objective=0.0, guarantee="not computed")
    frontier_ids: tuple[str, ...] = ()
    bound: ExhaustivenessBound | None = None
    universe_gaps: frozenset[Universe] = frozenset()

    # Stages A and B depend only on ``step``, and the C self-loop re-enters the
    # while-body with ``step`` unchanged -- so every search escalation used to
    # re-tile and re-QC an identical candidate set. QC is the second most
    # expensive stage in the pipeline (self-folding and direct-repeat stems are
    # both quadratic in spacer length), and the escalation budget defaults to 2.
    tiling_cache: dict[int, tuple[tuple[Candidate, ...], dict[str, QCResult]]] = {}

    # ``design_candidates`` has always accepted junction offsets and nothing
    # ever passed any, so ``Candidate.spans_junction`` was False on every guide
    # in every real run -- an advertised capability that silently did nothing.
    # The data is already on ``Locus.junctions``; this is the missing wire.
    junctions_by_target = {
        target_id: reference.transcriptome.loci[locus_id].junctions
        for locus_id, target_id in reference.locus_to_target.items()
        if locus_id in reference.transcriptome.loci
        and reference.transcriptome.loci[locus_id].junctions
    }

    def tile_and_qc(step: int) -> tuple[tuple[Candidate, ...], dict[str, QCResult]]:
        cached = tiling_cache.get(step)
        if cached is None:
            cands = design_candidates(
                spec, step=step, junctions_by_target=junctions_by_target
            )
            cached = (cands, run_qc(cands, spec, self_fold))
            tiling_cache[step] = cached
        return cached

    while True:
        # ---------------- A: candidate design ----------------
        # ---------------- B: deterministic QC ----------------
        candidates, qc_results = tile_and_qc(step)
        by_id = {c.guide_id: c for c in candidates}
        emit(f"A: {len(candidates)} candidates (step={step})")
        emit(f"B: {sum(1 for q in qc_results.values() if q.passed)}/{len(qc_results)} passed QC")

        # ---------------- C: sensitivity-bounded search -------
        outcome = run_search(
            candidates, spec, reference.index, reference.locus_to_target, envelope
        )
        zero_hit = sum(1 for hits in outcome.hits_by_guide.values() if not hits)
        fraction = zero_hit / max(1, len(candidates))
        emit(f"C: envelope<={envelope}mm, {zero_hit} zero-hit candidates ({fraction:.0%})")

        # ---- C self-loop: escalate a suspiciously tight envelope ----
        if fraction > ZERO_HIT_SUSPICION:
            if telemetry.search_escalations < spec.budget.max_search_escalations:
                telemetry.search_escalations += 1
                envelope += 1
                ledger.record(
                    "control_flow", "search_escalation", "v1",
                    new_envelope=envelope, zero_hit_fraction=round(fraction, 3),
                )
                emit(f"C-loop: escalating envelope to <={envelope}mm")
                continue
            # Recorded once per run, not once per outer-loop iteration.
            if ReasonCode.BUDGET_EXHAUSTED_SEARCH not in telemetry.reason_codes:
                telemetry.shortfalls.append(
                    f"search escalation budget exhausted at <={envelope}mm with "
                    f"{fraction:.0%} zero-hit candidates"
                )
                telemetry.reason_codes.append(ReasonCode.BUDGET_EXHAUSTED_SEARCH)

        ledger.record("search", "bound", outcome.bound.method,
                      claim=outcome.bound.describe(), note=outcome.bound.note)

        # ---------------- D: pair evidence, per context -------
        spacers = {gid: by_id[gid].spacer for gid in outcome.hits_by_guide}
        per_guide: dict[str, dict[str, tuple[PairEvidence, ...]]] = {
            gid: {} for gid in outcome.hits_by_guide
        }
        weakest_bound = outcome.bound
        # Gaps are unioned across contexts for the same reason the bound is
        # folded to the weakest: a variant context whose index covers fewer
        # universes leaves exactly the same hole, and reading the reference
        # context alone would not see it.
        universe_gaps = outcome.universe_gaps

        for ctx in contexts:
            ctx_outcome = (
                outcome
                if ctx is reference
                else run_search(candidates, spec, ctx.index, ctx.locus_to_target, envelope)
            )
            if is_weaker(ctx_outcome.bound, weakest_bound):
                weakest_bound = ctx_outcome.bound
            universe_gaps |= ctx_outcome.universe_gaps
            evidence = build_pair_evidence(
                ctx_outcome.hits_by_guide, spacers, spec, registry, calibration,
                ctx.transcriptome, ctx.expression_source(), context=ctx.name,
            )
            for gid, ev in evidence.items():
                per_guide.setdefault(gid, {})[ctx.name] = ev
        bound = weakest_bound
        if universe_gaps:
            ledger.exclusion(
                "universes", "required by assay but not proven searched",
                missing=sorted(u.value for u in universe_gaps),
            )
        emit(f"D: evidence built over {len(contexts)} context(s)")

        # ---------------- E + F: aggregate and gate -----------
        assessments = assess_guides(
            by_id, qc_results, per_guide, bound, universe_gaps, spec,
            kinetics=registry.kinetics,
        )
        passing = [a for a in assessments if a.gate.verdict is Verdict.PASS]
        emit(f"E/F: {len(passing)} PASS of {len(assessments)}")

        # ---------------- F: panel design ---------------------
        front = pareto_front(passing)
        frontier_ids = tuple(a.guide_id for a in front)
        panel = select_panel(front, spec, size=panel_size)

        # ---- G -> A loop: no feasible panel, re-tile ----
        if not _panel_feasible(panel, spec):
            if telemetry.panel_retiles < spec.budget.max_panel_retiles and step > 1:
                telemetry.panel_retiles += 1
                step = max(1, step // 2)
                ledger.record("control_flow", "panel_retile", "v1", new_step=step)
                emit(f"G->A loop: re-tiling at step={step}")
                continue
            uncovered = [
                t.target_id
                for t in spec.contract.by_class(TargetClass.REQUIRED)
                if panel.coverage.get(t.target_id, 0.0) < spec.thresholds.required_coverage_min
            ]
            if uncovered:
                telemetry.shortfalls.append(
                    "no feasible panel covers required target(s): " + ",".join(uncovered)
                )
                telemetry.reason_codes.append(ReasonCode.BUDGET_EXHAUSTED_NO_FEASIBLE_PANEL)
        break

    telemetry.final_envelope = envelope
    telemetry.final_step = step
    ledger.record(
        "control_flow", "telemetry", "v1",
        search_escalations=telemetry.search_escalations,
        panel_retiles=telemetry.panel_retiles,
        final_envelope=telemetry.final_envelope,
        shortfall_codes=[c.value for c in telemetry.reason_codes],
    )

    # ---------------- G: transparent output -------------------
    controls = generate_controls(panel, spec, reference.index, envelope=envelope)
    # Rank everything the gate did not FAIL -- PASS *and* REVIEW.
    #
    # REVIEW means "usable, with a human looking at it", and the R1/R2/R3
    # grading exists precisely so that a human can triage it. Ranking only PASS
    # made that grading invisible: with honest OOD flagging the REVIEW bucket is
    # routinely the majority, and a run that graded 135 guides R1/R2/R3 handed
    # back an *empty* table, which tells the user nothing about which of the 135
    # to look at first. The verdict and tier travel on every row, so nothing is
    # being passed off as certified.
    #
    # The *panel* is still PASS-only (see above): the machine's recommendation
    # stays certified even though the table a human reads is wider.
    ranked: list[RankedGuide] = rank_by_preference(
        [a for a in assessments if a.gate.verdict is not Verdict.FAIL],
        preference_weights,
    )
    # ---------------- FR6.3: what the panel is worth -----------------
    # Computed here rather than inside select_panel because it is an inference
    # about the chosen panel made from the calibration set, and the optimizer
    # deliberately knows nothing about calibration beyond the bounds it is
    # handed. The ledger records the model so an exported table says which
    # confidence it carried.
    confidence = BayesianPanelConfidence(
        calibration, spec, dependence=panel_dependence
    ).for_panel(panel)
    ledger.record(
        "model", "panel_confidence", confidence.version,
        dependence=confidence.dependence_version or "unmeasured",
    )

    return build_report(
        assessments, ranked, panel, controls, spec, ledger,
        shortfalls=tuple(telemetry.shortfalls),
        pareto_frontier=frontier_ids,
        search_bound=bound,
        universe_gaps=universe_gaps,
        panel_confidence=confidence,
    )
