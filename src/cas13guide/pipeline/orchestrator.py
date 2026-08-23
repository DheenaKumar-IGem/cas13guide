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
from ..services.provenance import ProvenanceLedger
from ..services.registry import ModelRegistry
from ..services.transcriptome import Transcriptome
from .a_candidates import design_candidates
from .b_qc import run_qc
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


def run_design(
    spec: ResolvedDesignSpec,
    contexts: list[DesignContext],
    registry: ModelRegistry,
    calibration: CalibrationService,
    panel_size: int = 3,
    initial_step: int = 1,
    preference_weights: dict[str, float] | None = None,
    on_event: Callable[[str], None] | None = None,
) -> DesignReport:
    """Run the full pipeline, backtracking within budget."""
    if not contexts:
        raise ValueError("at least one context is required")
    reference = next((c for c in contexts if c.name == "reference"), contexts[0])

    ledger = ProvenanceLedger(spec.fingerprint())
    ledger.record("spec", "ResolvedDesignSpec", spec.spec_version, fingerprint=spec.fingerprint())
    ledger.record("hazard_weights", "HazardWeights", spec.hazard.version)
    ledger.record("calibration", "CalibrationService", calibration.version,
                  strata=calibration.describe())
    for name, version in registry.versions().items():
        ledger.record("model", name, version)
    for ctx in contexts:
        ledger.record("transcriptome", ctx.name, ctx.transcriptome.version,
                      loci=len(ctx.transcriptome), index=ctx.index.version,
                      index_choice=ctx.index_rationale)

    telemetry = RunTelemetry(final_envelope=spec.thresholds.required_search_mismatches)
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
            cached = (cands, run_qc(cands, spec))
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
        universe_gaps: frozenset[Universe] = outcome.universe_gaps

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
    ranked: list[RankedGuide] = rank_by_preference(
        [a for a in assessments if a.gate.verdict is Verdict.PASS], preference_weights
    )
    return build_report(
        assessments, ranked, panel, controls, spec, ledger,
        shortfalls=tuple(telemetry.shortfalls),
        pareto_frontier=frontier_ids,
    )
