"""Stage D -- pair-level evidence.

For every ``guide x locus x context`` triple: verify the alignment, ask the
activity model, run the answer through the calibration service, bound the
expression, and attach the versioned hazard weight.

Two rules from the design live here:

* accessibility is credited **only** when it comes from a source independent
  of the activity model (SHAPE/DMS-seq/icSHAPE), never from the thermodynamic
  fold the model already used as a feature;
* asking a model outside its declared domain is allowed, but the result is
  marked out-of-domain and the calibration service widens the interval, so no
  confident number is ever produced for an unvalidated case.
"""

from __future__ import annotations

from ..contract.spec import ResolvedDesignSpec
from ..models.evidence import PairEvidence, SearchHit
from ..services.calibration import CalibrationService
from ..services.context import ExpressionSource, expression_interval
from ..services.registry import ModelRegistry
from ..services.transcriptome import Transcriptome


def _stratum_for(spec: ResolvedDesignSpec, hit: SearchHit) -> str:
    """Calibration stratum for one hit.

    Keyed on the *error regime* (mismatch depth) rather than the intent class:
    what changes the model's error is how far the duplex sits from the dense
    part of the training screen, not whether the designer labelled the target
    REQUIRED or FORBIDDEN.
    """
    regime = CalibrationService.mismatch_regime(hit.duplex.n_mismatch)
    return CalibrationService.stratum_key(
        spec.profile.effector.value, spec.profile.assay.value, regime
    )


def evidence_for_hit(
    hit: SearchHit,
    spacer: str,
    spec: ResolvedDesignSpec,
    registry: ModelRegistry,
    calibration: CalibrationService,
    transcriptome: Transcriptome,
    expression: ExpressionSource,
    context: str = "reference",
) -> PairEvidence:
    locus = transcriptome.loci.get(hit.locus_id)

    accessibility_value = None
    if locus is not None:
        accessibility_value = registry.accessibility.accessibility(
            locus.sequence, hit.start, len(hit.target_site)
        )

    in_domain = registry.activity.domain.supports(
        spec.profile.effector, len(spacer), hit.duplex.n_mismatch
    )
    prediction = registry.activity.predict(spacer, hit.duplex, accessibility_value)

    stratum = _stratum_for(spec, hit)
    activity = calibration.interval(prediction, stratum, spec.thresholds.alpha)
    if not in_domain:
        # Model domain violation is a stronger statement than a missing
        # calibration stratum: refuse to narrow at all.
        activity = type(activity).unknown(stratum=f"{stratum}->model-out-of-domain")

    tpm = expression.expression_tpm(hit.locus_id)
    if tpm is None and locus is not None:
        tpm = locus.expression
    target = spec.contract.get(hit.target_id) if hit.target_id else None
    relative_uncertainty = target.expression_uncertainty if target else 0.25
    expression_iv = expression_interval(tpm, relative_uncertainty)

    hazard_weight = spec.hazard.weight_for(hit.target_class, hit.biotype)

    accessibility_iv = None
    if accessibility_value is not None:
        accessibility_iv = type(activity).exact(accessibility_value, stratum="accessibility")

    return PairEvidence(
        guide_id=hit.guide_id,
        locus_id=hit.locus_id,
        context=context,
        hit=hit,
        activity=activity,
        expression=expression_iv,
        hazard_weight=hazard_weight,
        accessibility=accessibility_iv,
        domain_ok=in_domain,
    )


def build_pair_evidence(
    hits_by_guide: dict[str, tuple[SearchHit, ...]],
    spacers: dict[str, str],
    spec: ResolvedDesignSpec,
    registry: ModelRegistry,
    calibration: CalibrationService,
    transcriptome: Transcriptome,
    expression: ExpressionSource,
    context: str = "reference",
) -> dict[str, tuple[PairEvidence, ...]]:
    out: dict[str, tuple[PairEvidence, ...]] = {}
    for guide_id, hits in hits_by_guide.items():
        spacer = spacers[guide_id]
        out[guide_id] = tuple(
            evidence_for_hit(
                hit, spacer, spec, registry, calibration, transcriptome, expression, context
            )
            for hit in hits
        )
    return out
