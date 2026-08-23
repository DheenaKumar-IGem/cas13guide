"""Shared fixtures for the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cas13guide.contract.enums import AssayType, Effector, Task, TargetClass, Universe
from cas13guide.contract.spec import (
    ContextScope,
    EffectorAssayProfile,
    ResolvedDesignSpec,
    TargetContract,
    TargetSpec,
)
from cas13guide.metrics.bounds import Interval
from cas13guide.models.evidence import Candidate, PairEvidence, QCResult, SearchHit
from cas13guide.services.indexes import ExhaustivenessBound
from cas13guide.util.duplex import profile_duplex
from cas13guide.util.seq import revcomp

FULL_UNIVERSES = frozenset(
    {Universe.CODING, Universe.NCRNA, Universe.PRE_MRNA, Universe.RRNA, Universe.MITO}
)

SEQ_A = "AUGGCACUGAAGUCCAUCGACUGGAACUUCCGAGUGCAUGGCAUUACCGAGUCCUAUGGCAAG"
SEQ_B = "AUGCCUGAUUACGGAUCCAUUGGCCUAAGCUUAGCCAUGGAUCCGUUAACGGAUCCAUUGGCA"


def make_spec(
    task: Task = Task.KNOCKDOWN,
    targets=None,
    universes: frozenset = FULL_UNIVERSES,
    assay: AssayType = AssayType.CELL_CULTURE_KNOCKDOWN,
    **threshold_overrides,
) -> ResolvedDesignSpec:
    if targets is None:
        targets = (
            TargetSpec(
                target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED, expression=100.0
            ),
        )
    spec = ResolvedDesignSpec(
        contract=TargetContract(task=task, targets=targets),
        profile=EffectorAssayProfile(effector=Effector.RFX_CAS13D, assay=assay),
        context=ContextScope(
            organism="Homo sapiens",
            genome_build="GRCh38",
            annotation_version="v44",
            declared_universes=universes,
        ),
    )
    if threshold_overrides:
        spec = spec.model_copy(
            update={"thresholds": spec.thresholds.model_copy(update=threshold_overrides)}
        )
    return spec


def make_hit(
    guide_id: str = "g1",
    locus_id: str = "L1",
    target_id: str | None = None,
    target_class: TargetClass | None = None,
    biotype: Universe = Universe.CODING,
    spacer: str = "GCAUUGCUGUGGAGCUGAAGUC",
    n_mismatch: int = 0,
) -> SearchHit:
    site = list(revcomp(spacer))
    # Introduce true (non-wobble) mismatches: G->C is never a wobble.
    for i in range(n_mismatch):
        site[i * 3] = "C" if site[i * 3] != "C" else "G"
    site_str = "".join(site)
    return SearchHit(
        guide_id=guide_id,
        locus_id=locus_id,
        target_site=site_str,
        start=0,
        duplex=profile_duplex(spacer, site_str),
        biotype=biotype,
        target_id=target_id,
        target_class=target_class,
    )


def make_evidence(
    activity_lower: float,
    activity_upper: float,
    *,
    guide_id: str = "g1",
    locus_id: str = "L1",
    target_id: str | None = None,
    target_class: TargetClass | None = None,
    expression: tuple[float, float] = (0.5, 0.5),
    hazard_weight: float = 1.0,
    ood: bool = False,
    domain_ok: bool = True,
    expression_imputed: bool = False,
    biotype: Universe = Universe.CODING,
) -> PairEvidence:
    return PairEvidence(
        guide_id=guide_id,
        locus_id=locus_id,
        context="reference",
        hit=make_hit(guide_id, locus_id, target_id, target_class, biotype),
        activity=Interval(
            point=(activity_lower + activity_upper) / 2,
            lower=activity_lower,
            upper=activity_upper,
            ood=ood,
        ),
        expression=Interval(
            point=expression[0],
            lower=expression[0],
            upper=expression[1],
            imputed=expression_imputed,
        ),
        hazard_weight=hazard_weight,
        domain_ok=domain_ok,
    )


def clean_qc(guide_id: str = "g1") -> QCResult:
    return QCResult(guide_id=guide_id, passed=True, manufacturability=1.0)


def exhaustive_bound(max_mismatches: int = 4, exhaustive: bool = True) -> ExhaustivenessBound:
    return ExhaustivenessBound(
        max_mismatches=max_mismatches,
        wobble_aware=True,
        universes=FULL_UNIVERSES,
        exhaustive=exhaustive,
        method="test",
    )


def make_candidate(guide_id: str = "g1", spacer: str = "GCAUUGCUGUGGAGCUGAAGUC") -> Candidate:
    return Candidate(guide_id=guide_id, spacer=spacer, source_target_id="t1", start=0)
