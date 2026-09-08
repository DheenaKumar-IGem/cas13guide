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

from ..contract.enums import TargetClass
from ..contract.spec import ResolvedDesignSpec
from ..metrics.bounds import Interval
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

    # Only for a site the design is *aiming at*. Every reader of this value is
    # guarded by ``is_intended`` -- ``intended_accessibility`` and the metric
    # vector both skip a non-intended pair, the off-target explorer has no
    # accessibility column, and a folded profile is not independent evidence so
    # it never reaches the activity model. Computing it for an off-target hit
    # was therefore a full ``pfl_fold_up`` over that locus's whole sequence,
    # thrown away.
    #
    # It is the dominant cost of any run with a real background corpus, because
    # that is exactly where the off-target hits are: measured on a 1.5 kb target
    # against 25 background loci, 7.85s -> 0.90s, an 8.7x difference for an
    # identical report.
    intended = hit.target_class in (TargetClass.REQUIRED, TargetClass.DESIRED)
    accessibility_value = None
    if locus is not None and intended:
        accessibility_value = registry.accessibility.accessibility(
            locus.sequence, hit.start, len(hit.target_site)
        )

    # A computed fold is not independent evidence: it is a prediction from the
    # same sequence the activity model already sees, and letting it into
    # predict() would have one computation vouch for another. It is still
    # recorded below, and it still ranks -- it just does not move the bound.
    independent = getattr(registry.accessibility, "is_independent_evidence", False)
    model_accessibility = accessibility_value if independent else None

    in_domain = registry.activity.domain.supports(
        spec.profile.effector, len(spacer), hit.duplex.n_mismatch
    )
    stratum = _stratum_for(spec, hit)

    if not in_domain:
        # Do not ask a model a question it has told us it cannot answer. The
        # domain check ran here and `predict()` was called anyway, its result
        # thrown away three lines later -- harmless for the built-in heuristic,
        # fatal for a trained one: `TrainedActivityModel.predict` delegates to
        # a FeatureSpec fixed at the screen's spacer length and raises
        # ValueError for any other, so fitting on a 22 nt screen and designing
        # at 23 nt -- both canonical for RfxCas13d -- crashed the run on a
        # prediction nothing was going to read.
        #
        # A domain violation is a stronger statement than a missing calibration
        # stratum: refuse to narrow at all.
        activity = Interval.unknown(stratum=f"{stratum}->model-out-of-domain")
    else:
        prediction = registry.activity.predict(spacer, hit.duplex, model_accessibility)
        activity = calibration.interval(prediction, stratum, spec.thresholds.alpha)

    tpm = expression.expression_tpm(hit.locus_id)
    if tpm is None and locus is not None:
        tpm = locus.expression
    target = spec.contract.get(hit.target_id) if hit.target_id else None
    relative_uncertainty = target.expression_uncertainty if target else 0.25
    expression_iv = expression_interval(tpm, relative_uncertainty)

    hazard_weight = spec.hazard.weight_for(hit.target_class, hit.biotype)

    accessibility_iv = None
    if accessibility_value is not None:
        accessibility_iv = Interval.exact(
            accessibility_value,
            stratum="accessibility" if independent else "accessibility/predicted",
        )

    # The same waste as accessibility above, in the field beside it, left
    # behind when that one was fixed. `duplex_dg_per_nt` has exactly one
    # reader -- `metrics/aggregate.intended_duplex_energy`, which skips a
    # non-intended pair -- and the off-target explorer has no dG column, so a
    # fold run for an off-target hit was computed and dropped.
    #
    # It grows with the background while the intended folds stay fixed:
    # measured on an 800 nt target against 80 background loci, 1290 discarded
    # folds against 785 read ones, 12% of wall clock for an identical report.
    energy = None
    if intended:
        energy = registry.energy.duplex_dg_per_nt(spacer, hit.target_site)

    return PairEvidence(
        guide_id=hit.guide_id,
        locus_id=hit.locus_id,
        context=context,
        hit=hit,
        activity=activity,
        expression=expression_iv,
        hazard_weight=hazard_weight,
        accessibility=accessibility_iv,
        duplex_dg_per_nt=energy,
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
