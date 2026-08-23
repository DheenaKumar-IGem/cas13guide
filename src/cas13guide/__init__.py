"""Cas13 target-RNA guide locator and scoring.

Architecture (see design doc v2):

    Layer 0   contract/    ResolvedDesignSpec -- the question, stated precisely
    Shared    services/    provenance, registry, calibration, context, indexes,
                           transcriptome, benchmark -- versioned, cross-cutting
    Pipeline  pipeline/    A candidates -> B qc -> C search -> D pair evidence
                           -> E aggregate -> F decide -> G output
    Support   metrics/     the six named metrics and the LCB/UCB bound algebra
              models/      typed objects that move between stages
              util/        sequence and duplex primitives

The load-bearing rule: efficacy, specificity, intent and uncertainty are never
collapsed into one score before the safety gate and the Pareto pass.
"""

__version__ = "0.1.0"
