"""Stage E -- guide-level aggregation.

Turns per-context pair evidence into one metric vector per guide and runs it
through the gate. This stage is deliberately thin: the metric definitions live
in :mod:`cas13guide.metrics.aggregate` and the decision rules in
:mod:`cas13guide.pipeline.f_decide`, so neither can drift from the other.
"""

from __future__ import annotations

from ..contract.spec import ResolvedDesignSpec
from ..metrics.aggregate import build_metric_vector
from ..models.evidence import Candidate, GuideAssessment, PairEvidence, QCResult
from ..services.indexes import ExhaustivenessBound
from .f_decide import gate


def assess_guides(
    candidates: dict[str, Candidate],
    qc_results: dict[str, QCResult],
    evidence_by_guide_and_context: dict[str, dict[str, tuple[PairEvidence, ...]]],
    bound: ExhaustivenessBound,
    universe_gaps: frozenset,
    spec: ResolvedDesignSpec,
) -> list[GuideAssessment]:
    assessments: list[GuideAssessment] = []

    for guide_id, candidate in candidates.items():
        qc = qc_results[guide_id]
        by_context = evidence_by_guide_and_context.get(guide_id, {})
        if not by_context:
            by_context = {"reference": ()}

        metrics = build_metric_vector(by_context, qc, spec)
        # ``is None``, not ``or``: an empty reference tuple is falsy but is
        # still the reference, and must not silently fall through to a
        # variant context whose evidence the metrics were not computed from.
        reference = by_context.get("reference")
        if reference is None:
            reference = next(iter(by_context.values()))
        # The gate reads evidence from every context, not just the reference:
        # an OOD flag or an unknown expression in a variant context is still
        # an evidence gap for this guide.
        all_evidence = tuple(e for ev in by_context.values() for e in ev)

        result = gate(metrics, qc, all_evidence, bound, universe_gaps, spec)
        assessments.append(
            GuideAssessment(
                guide_id=guide_id,
                candidate=candidate,
                qc=qc,
                metrics=metrics,
                gate=result,
                evidence=reference,
            )
        )
    return assessments
