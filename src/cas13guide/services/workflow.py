"""One design run, from input to report, callable from anywhere.

The CLI and the web server are both thin wrappers over :func:`run_request`.
Keeping the whole flow here rather than inside an ``argparse`` handler is what
makes that possible: a request is a dataclass, not a namespace of CLI flags, so
a form POST and a command line reach the pipeline through the same code and can
never drift into behaving differently.

Everything that is *not* the pipeline lives here too -- choosing which models to
build, seeding a stand-in calibration, deciding which universes to declare --
because those are deployment decisions rather than science, and they are exactly
the decisions that go wrong quietly when each entry point makes them for itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..contract.enums import AssayType, Effector, TargetClass, Task, Universe
from ..contract.presets import designability, get_preset, largest_designable_spacer_length
from ..contract.spec import (
    ContextScope,
    EffectorAssayProfile,
    ResolvedDesignSpec,
    TargetContract,
    TargetSpec,
    Thresholds,
)
from .fasta import SequenceInputError, SequenceRecord, parse_sequence_input, read_fasta

#: Universes declared for a design run. Deliberately the full set: declaring
#: fewer would quietly narrow the universe-adequacy gate, which is the check
#: that stops a suspiciously clean off-target report from passing.
ALL_UNIVERSES: frozenset[Universe] = frozenset(Universe)

#: Said out loud on every run that uses the built-in calibration. Neither of
#: these is a number to put in a paper, and both look exactly like numbers that
#: are.
STANDING_CAVEATS: tuple[str, ...] = (
    "The bundled calibration is SYNTHETIC. Bounds here are exercisable, not "
    "earned -- supply real screen data via `cas13guide train` before relying "
    "on them.",
    "Predicted signal is RELATIVE and comparable only within this run: the "
    "kinetic constants shipped by default are placeholders, so no absolute "
    "fluorescence figure is offered.",
)


class DesignRequestError(ValueError):
    """The request cannot be turned into a design spec. Message is user-facing."""


@dataclass(slots=True)
class DesignRequest:
    """Everything one run needs. Defaults match the CLI's."""

    sequence_text: str | None = None
    fasta_path: str | Path | None = None
    preset: str | None = None
    variants_path: str | Path | None = None
    variants_text: str | None = None
    background_path: str | Path | None = None

    effector: str = "LwaCas13a"
    assay: str = "in_vitro_detection"
    task: str = "detection"
    spacer_length: int | None = None
    max_mismatches: int = 4
    min_accessibility: float = 0.0
    conserved_mismatches: int = 0
    panel_size: int = 3
    expression: float | None = None

    organism: str = "Homo sapiens"
    genome_build: str = "GRCh38"
    annotation: str = "v44"

    use_structure: bool = True
    use_signal: bool = True

    seed: int = 0
    calibration_n: int = 200
    calibration_noise: float = 0.10


@dataclass(slots=True)
class DesignOutcome:
    """The report plus everything a caller needs to present it honestly."""

    report: dict[str, Any]
    records: list[SequenceRecord]
    spec: ResolvedDesignSpec
    notes: list[str] = field(default_factory=list)
    #: Per-nucleotide unpaired probability of the primary target, when a
    #: structural model was available. Drives the report's accessibility track.
    profile: tuple[float, ...] = ()

    @property
    def ranked(self) -> list[dict[str, Any]]:
        return self.report.get("ranked", [])


def _read_records(request: DesignRequest) -> list[SequenceRecord]:
    if request.preset:
        preset = get_preset(request.preset)
        return [
            SequenceRecord(
                name=preset.preset_id,
                sequence=preset.sequence,
                description=preset.accession,
            )
        ]
    if request.fasta_path:
        return read_fasta(request.fasta_path)
    if request.sequence_text and request.sequence_text.strip():
        return parse_sequence_input(request.sequence_text)
    raise DesignRequestError(
        "No sequence supplied. Paste a sequence, upload a FASTA file, or pick a "
        "built-in target."
    )


def _resolve_spacer_length(
    request: DesignRequest, effector: Effector, notes: list[str]
) -> int | None:
    if request.spacer_length is not None:
        return request.spacer_length
    if not request.preset:
        return None                      # the ortholog's canonical length
    preset = get_preset(request.preset)
    length = largest_designable_spacer_length(preset, effector)
    notes.append(
        f"{preset.preset_id} is {preset.length} nt: "
        + designability(preset, effector, length).note
    )
    if length is None:
        raise DesignRequestError(
            f"{preset.preset_id} ({preset.length} nt) is too short for any "
            f"spacer {effector.value} tolerates. Design against an amplicon "
            "instead."
        )
    return length


def build_spec(request: DesignRequest, records: Sequence[SequenceRecord],
               notes: list[str]) -> ResolvedDesignSpec:
    try:
        effector = Effector(request.effector)
        assay = AssayType(request.assay)
        task = Task(request.task)
    except ValueError as exc:
        raise DesignRequestError(str(exc)) from exc

    spacer_length = _resolve_spacer_length(request, effector, notes)
    targets = tuple(
        TargetSpec(
            target_id=record.name,
            sequence=record.sequence,
            # The first record is what the design must hit; the rest are
            # desirable, not required, so one unusable extra record cannot
            # fail the whole run.
            target_class=TargetClass.REQUIRED if index == 0 else TargetClass.DESIRED,
            gene=record.name,
            biotype=Universe.NCRNA,
            expression=request.expression,
        )
        for index, record in enumerate(records)
    )
    try:
        return ResolvedDesignSpec(
            contract=TargetContract(task=task, targets=targets),
            profile=EffectorAssayProfile(
                effector=effector, assay=assay, spacer_length=spacer_length
            ),
            context=ContextScope(
                organism=request.organism,
                genome_build=request.genome_build,
                annotation_version=request.annotation,
                declared_universes=ALL_UNIVERSES,
            ),
            thresholds=Thresholds(
                required_search_mismatches=request.max_mismatches,
                min_accessibility=request.min_accessibility,
            ),
        )
    except Exception as exc:
        raise DesignRequestError(f"cannot build the design spec: {exc}") from exc


def build_registry(request: DesignRequest, notes: list[str]):
    """Assemble the model registry, degrading loudly rather than silently."""
    from .registry import HeuristicCas13Model, ModelRegistry, NullAccessibilityModel

    accessibility = NullAccessibilityModel()
    extra: dict[str, Any] = {}

    if request.use_structure:
        try:
            from .accessibility import ViennaRNAAccessibilityModel
            from .thermodynamics import ViennaRNADuplexEnergyModel

            accessibility = ViennaRNAAccessibilityModel()
            extra["energy"] = ViennaRNADuplexEnergyModel()
        except Exception as exc:
            notes.append(
                f"Structural scoring is OFF ({exc}). Accessibility and duplex "
                'energy will be blank. Install with: pip install "cas13guide[structure]"'
            )
    if request.use_signal:
        from .kinetics import CollateralKineticsModel

        extra["kinetics"] = CollateralKineticsModel()

    return ModelRegistry(HeuristicCas13Model(), accessibility, **extra)


def seed_calibration(request: DesignRequest, spec: ResolvedDesignSpec):
    """A stand-in calibration so bounds exist at all.

    Synthetic on purpose and flagged as such in :data:`STANDING_CAVEATS`. Group
    labels are supplied so the group-adequacy check is actually exercised rather
    than switching itself off.
    """
    from .calibration import CalibrationService

    rng = random.Random(request.seed)
    calibration = CalibrationService()
    for regime in ("mm0", "mm1-2", "mm3+"):
        key = CalibrationService.stratum_key(
            spec.profile.effector.value, spec.profile.assay.value, regime
        )
        for i in range(request.calibration_n):
            truth = rng.random()
            observed = min(1.0, max(0.0, truth + rng.gauss(0, request.calibration_noise)))
            calibration.add_observation(key, observed, truth, group=f"T{i % 12:02d}")
    return calibration


def run_request(request: DesignRequest) -> DesignOutcome:
    """Run one design end to end. Raises :class:`DesignRequestError` on bad input."""
    from ..pipeline.orchestrator import DesignContext, run_design
    from .indexes import ExhaustiveScanIndex
    from .transcriptome import Locus, Transcriptome

    notes: list[str] = list(STANDING_CAVEATS)
    try:
        records = _read_records(request)
    except SequenceInputError as exc:
        raise DesignRequestError(str(exc)) from exc

    spec = build_spec(request, records, notes)
    registry = build_registry(request, notes)

    world = Transcriptome(request.genome_build, request.annotation)
    for record in records:
        world.add(Locus(record.name, record.sequence, Universe.NCRNA,
                        gene=record.name, expression=request.expression))
    if request.background_path:
        for record in read_fasta(request.background_path):
            world.add(Locus(record.name, record.sequence, Universe.CODING,
                            gene=record.name, expression=None))

    contexts = [
        DesignContext(
            "reference", world, ExhaustiveScanIndex(world, ALL_UNIVERSES),
            {record.name: record.name for record in records},
        )
    ]

    variant_records: list[SequenceRecord] = []
    if request.variants_path:
        variant_records = read_fasta(request.variants_path)
    elif request.variants_text and request.variants_text.strip():
        variant_records = parse_sequence_input(request.variants_text, default_name="variant")
    if variant_records:
        from .conservation import summarize_conservation, variant_contexts

        combined = [records[0]] + variant_records
        summary = summarize_conservation(
            combined, spec.spacer_length, max_mismatches=request.conserved_mismatches
        )
        notes.append(f"Conservation across {len(variant_records)} variant(s): "
                     f"{summary.summary()}")
        contexts = variant_contexts(
            combined, ALL_UNIVERSES, target_id=records[0].name,
            biotype=Universe.NCRNA, expression=request.expression,
        )

    report = run_design(
        spec, contexts, registry, seed_calibration(request, spec),
        panel_size=request.panel_size,
    )
    data = report.to_dict()

    profile: tuple[float, ...] = ()
    source = getattr(registry.accessibility, "unpaired_profile", None)
    if source is not None and records:
        try:
            profile = tuple(source(records[0].sequence))
        except Exception:
            profile = ()

    return DesignOutcome(
        report=data, records=records, spec=spec, notes=notes, profile=profile
    )
