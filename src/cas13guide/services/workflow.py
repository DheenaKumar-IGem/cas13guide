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

#: The index *filter*: every universe is admitted, so no locus the user
#: supplied is excluded from the search. This is the job the name was right
#: for.
#:
#: It was also passed as ``declared_universes``, the caller's assertion of what
#: the corpus covers, with a comment saying declaring fewer "would quietly
#: narrow the universe-adequacy gate". The opposite was true. `universe_gaps`
#: computes ``(required | declared) - searched``; declaring everything and
#: searching a corpus of one ncrna locus gave ``ALL - ALL``, because the index
#: reported its filter as what it had searched. The gate returned no gap on
#: every run the CLI has ever done, and the report told the user off-targets
#: had been searched over rrna and mito -- the two highest hazard weights in
#: the package -- against sequences that were never loaded.
ALL_UNIVERSES: frozenset[Universe] = frozenset(Universe)

#: What the target record is, when the user has not said. Keyed on the assay
#: because that is the only evidence available: an in-vitro detection run is
#: designed against a supplied amplicon, an antiviral run against a viral
#: genome, a knockdown run against a cellular transcript. Every value here was
#: previously ``NCRNA`` regardless of assay, which is why an in-vitro run
#: covered neither universe its own assay is exposed to.
TARGET_BIOTYPE_BY_ASSAY: dict[AssayType, Universe] = {
    AssayType.IN_VITRO_DETECTION: Universe.SYNTHETIC,
    AssayType.ANTIVIRAL: Universe.VIRAL,
    AssayType.CELL_CULTURE_KNOCKDOWN: Universe.CODING,
    AssayType.IN_VIVO_KNOCKDOWN: Universe.CODING,
}

#: Kept as the module's public name for callers that imported it, but the
#: caveats a run actually carries are now *derived* from that run -- see
#: ``services/limitations.py``. A fixed banner has two failure modes and this
#: package would hit both: it keeps warning about placeholder constants after
#: you have supplied real ones, and it stays silent about a heuristic index
#: because nobody remembered to add the line.
STANDING_CAVEATS: tuple[str, ...] = ()


class DesignRequestError(ValueError):
    """The request cannot be turned into a design spec. Message is user-facing."""


@dataclass(slots=True)
class DesignRequest:
    """Everything one run needs. Defaults match the CLI's."""

    sequence_text: str | None = None
    fasta_path: str | Path | None = None
    #: An uploaded FASTA's *text*, as the browser supplies it. Distinct from
    #: ``sequence_text`` so the one-source guard can see it: the web form used
    #: to fold the two together, which put the upload beyond the guard's reach
    #: and silently dropped it whenever the textarea was also filled -- the
    #: precise behaviour the guard was added to stop, on the door it was added
    #: for. Text rather than a path, so the upload keeps going through
    #: ``parse_sequence_input`` and the two doors stay identical.
    fasta_text: str | None = None
    preset: str | None = None
    variants_path: str | Path | None = None
    variants_text: str | None = None
    background_path: str | Path | None = None

    effector: str = "LwaCas13a"
    assay: str = "in_vitro_detection"
    #: What the supplied sequences are. The universe-adequacy gate asks whether
    #: the corpus covers what the assay exposes the guide to, and it can only
    #: be answered by labelling the sequences -- a declaration cannot buy
    #: coverage, because the gate reads what the index holds.
    target_biotype: str | None = None
    background_biotype: str = "coding"
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
    #: Path to a JSON file of measured kinetic constants. Without one the model
    #: runs on placeholders and ``absolute_signal()`` refuses -- which was
    #: unreachable-by-design until now, because nothing plumbed a path through.
    kinetics_json: str | None = None
    #: Path to a JSON file holding a measured common-cause fraction for panel
    #: members (FR6.3). Without one the panel's detection probability is
    #: reported as the interval it honestly is, and no point estimate is
    #: offered -- see ``services/panel_confidence.py``.
    panel_dependence_json: str | None = None
    #: Use the Arc RNAtargeting CNN instead of the built-in heuristic. Needs
    #: onnxruntime and gpu-contrafold, RfxCas13d, and a 30 nt spacer -- the only
    #: configuration the network was trained on. Refuses loudly otherwise
    #: rather than scoring out-of-domain and calling it activity.
    use_arc: bool = False
    #: Path to a BLAST nucleotide database for off-target search (FR4, DR1/DR3).
    #: Without one the exhaustive wobble-aware scanner runs over whatever
    #: `--background` supplied, which is slower but reports exhaustive=True.
    blast_db: str | None = None
    #: What the caller asserts that database covers. The universe-adequacy gate
    #: reads this assertion and cannot check it, so an honest value matters.
    blast_universes: str = "coding"


@dataclass(slots=True)
class DesignOutcome:
    """The report plus everything a caller needs to present it honestly."""

    report: dict[str, Any]
    records: list[SequenceRecord]
    spec: ResolvedDesignSpec
    notes: list[str] = field(default_factory=list)
    #: What this run could not establish, derived from the models and bounds it
    #: actually used. See ``services/limitations.py``.
    limitations: tuple = ()
    #: Per-nucleotide unpaired probability of the primary target, when a
    #: structural model was available. Drives the report's accessibility track.
    profile: tuple[float, ...] = ()

    @property
    def ranked(self) -> list[dict[str, Any]]:
        return self.report.get("ranked", [])


def _read_records(request: DesignRequest) -> list[SequenceRecord]:
    # A run designs against one target source. This used to return on the first
    # of preset > fasta > sequence that was set, so supplying two meant the
    # other was read by nobody and mentioned by nobody -- not in the notes, the
    # shortfalls, or the export header. The web form offers all three at once,
    # and argparse puts them in one group with no exclusion, so it is reachable
    # from both doors. Refusing is the only honest answer: guessing which one
    # the user meant is how a report ends up being about a target they did not
    # think they were designing.
    supplied = [
        flag
        for flag, present in (
            ("--preset", bool(request.preset)),
            (
                "--fasta",
                bool(request.fasta_path)
                or bool(request.fasta_text and request.fasta_text.strip()),
            ),
            ("--sequence", bool(request.sequence_text and request.sequence_text.strip())),
        )
        if present
    ]
    if len(supplied) > 1:
        raise DesignRequestError(
            f"{', '.join(supplied)} were all supplied, and a run designs against "
            "one target source. Pick one -- otherwise the report would be about "
            "a target you may not have chosen, with nothing saying so."
        )
    if request.preset:
        try:
            preset = get_preset(request.preset)
        except KeyError as exc:
            # get_preset's KeyError is right for a library caller and wrong
            # here: run_request documents DesignRequestError, and cli._design
            # catches exactly that. `_read_fasta_or_fail` below was added to
            # close this hole for --background and --variants and left the
            # branch directly above it open, so dropping the "-5p" off a preset
            # id still produced a traceback where the same typo on --fasta
            # produced a located message. args[0] rather than str(exc), which
            # would wrap the whole hint in repr quotes.
            raise DesignRequestError(exc.args[0]) from exc
        return [
            SequenceRecord(
                name=preset.preset_id,
                sequence=preset.sequence,
                description=preset.accession,
            )
        ]
    if request.fasta_path:
        return read_fasta(request.fasta_path)
    if request.fasta_text and request.fasta_text.strip():
        return parse_sequence_input(request.fasta_text)
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


def _record_biotype(record, default: Universe, flag: str) -> Universe:
    """The universe one supplied record belongs to.

    ``biotype=`` in the FASTA description labels a single record, so one file
    can hold the several universes an assay is exposed to; ``default`` is what
    the caller said about the file as a whole. Without this a corpus could
    cover at most two universes -- the target's and the background's -- and
    cell_culture_knockdown is exposed to five, so the adequacy gate was
    unsatisfiable through this door however the user labelled the file.
    """
    for token in record.description.split():
        key, sep, value = token.partition("=")
        if sep and key.lower() == "biotype":
            try:
                return Universe(value.strip().lower())
            except ValueError:
                raise DesignRequestError(
                    f"{flag} record {record.name!r} carries "
                    f"biotype={value!r}, which is not a universe. Use one of: "
                    + ", ".join(sorted(u.value for u in Universe))
                ) from None
    return default


def _target_biotype(request: DesignRequest) -> Universe:
    """What the target records are, as the user said or as the run can tell."""
    if request.target_biotype:
        return Universe(request.target_biotype)
    if request.preset:
        # A preset ships its own biotype -- the shipped ones are miRNAs, and
        # ncrna is what they are. Letting the assay default overrule it made a
        # run disagree with `to_target_spec` about the same preset, so the
        # tests that use `to_target_spec` would have been describing a target
        # no run produces.
        try:
            return get_preset(request.preset).biotype
        except KeyError:
            # An unknown preset id has its own legible message further in.
            pass
    try:
        return TARGET_BIOTYPE_BY_ASSAY[AssayType(request.assay)]
    except ValueError:
        # An unknown assay string is rejected with a legible message further
        # in; returning a default here keeps that message the one the user
        # sees, rather than a KeyError from this helper.
        return Universe.NCRNA


def build_spec(request: DesignRequest, records: Sequence[SequenceRecord],
               notes: list[str]) -> ResolvedDesignSpec:
    try:
        effector = Effector(request.effector)
        assay = AssayType(request.assay)
        target_biotype = _target_biotype(request)
        # The corpus covers the background's universe too, when one is given.
        # An overclaim here can only widen the gap the gate reads -- it keys on
        # what the index holds -- so the safe direction is to name it.
        corpus = {target_biotype}
        if request.background_path:
            corpus.add(Universe(request.background_biotype))
        task = Task(request.task)
    except ValueError as exc:
        raise DesignRequestError(str(exc)) from exc

    spacer_length = _resolve_spacer_length(request, effector, notes)
    # Inside the guard below, not above it: TargetSpec validates too (expression
    # is ge=0), so `--expression -5` raised a raw pydantic ValidationError while
    # the adjacent `--spacer-length 30` rejection produced a clean message. One
    # function, two contracts.
    try:
        targets = tuple(
            TargetSpec(
                target_id=record.name,
                sequence=record.sequence,
                # The first record is what the design must hit; the rest are
                # desirable, not required, so one unusable extra record cannot
                # fail the whole run.
                target_class=(
                    TargetClass.REQUIRED if index == 0 else TargetClass.DESIRED
                ),
                gene=record.name,
                biotype=target_biotype,
                expression=request.expression,
            )
            for index, record in enumerate(records)
        )
        return ResolvedDesignSpec(
            contract=TargetContract(task=task, targets=targets),
            profile=EffectorAssayProfile(
                effector=effector, assay=assay, spacer_length=spacer_length
            ),
            context=ContextScope(
                organism=request.organism,
                genome_build=request.genome_build,
                annotation_version=request.annotation,
                # What the corpus actually holds, not what the search was
                # willing to admit. `build_spec` sees only the target records;
                # `run_request` widens this with the accepted --background
                # loci once it knows which survived.
                declared_universes=frozenset(corpus),
            ),
            thresholds=Thresholds(
                required_search_mismatches=request.max_mismatches,
                min_accessibility=request.min_accessibility,
            ),
        )
    except Exception as exc:
        raise DesignRequestError(f"cannot build the design spec: {exc}") from exc


def build_registry(request: DesignRequest, notes: list[str], spec=None, records=()):
    """Assemble the model registry, degrading loudly rather than silently."""
    from .registry import HeuristicCas13Model, ModelRegistry, NullAccessibilityModel

    accessibility = NullAccessibilityModel()
    extra: dict[str, Any] = {}

    if request.use_structure:
        try:
            from .accessibility import ViennaRNAAccessibilityModel
            from .thermodynamics import ViennaRNADuplexEnergyModel

            accessibility = ViennaRNAAccessibilityModel()   # type: ignore[assignment]
            extra["energy"] = ViennaRNADuplexEnergyModel()
        except Exception as exc:
            notes.append(
                f"Structural scoring is OFF ({exc}). Accessibility and duplex "
                'energy will be blank. Install with: pip install "cas13guide[structure]"'
            )
    if request.kinetics_json and not request.use_signal:
        # Two flags that contradict each other. The file was not read at all,
        # so a malformed one passed unnoticed -- a typo in a scripted pipeline
        # would surface only when somebody removed --no-signal, on a run that
        # had "worked" for weeks. Validate it either way, and say it went
        # unused rather than ignoring it in silence.
        from .kinetics import KineticConstants

        try:
            KineticConstants.from_json(request.kinetics_json)
        except (OSError, ValueError, TypeError) as exc:
            raise DesignRequestError(
                f"--kinetics-constants {request.kinetics_json}: {exc}"
            ) from exc
        notes.append(
            f"Kinetic constants were supplied ({request.kinetics_json}) and "
            "signal prediction is off, so they were NOT used and no signal "
            "was predicted. Drop --no-signal to use them."
        )

    if request.use_signal:
        from .kinetics import CollateralKineticsModel, KineticConstants

        if request.kinetics_json:
            try:
                constants = KineticConstants.from_json(request.kinetics_json)
            except (OSError, ValueError, TypeError) as exc:
                raise DesignRequestError(
                    f"--kinetics-constants {request.kinetics_json}: {exc}"
                ) from exc
            extra["kinetics"] = CollateralKineticsModel(constants)
            if constants.provisional:
                notes.append(
                    f"Kinetic constants loaded from {request.kinetics_json} are "
                    "still flagged provisional=true, so absolute signal stays "
                    "refused. Set provisional=false once the values are measured."
                )
        else:
            extra["kinetics"] = CollateralKineticsModel()

    activity = HeuristicCas13Model()
    if request.use_arc:
        activity = _build_arc_model(request, spec, records)
    return ModelRegistry(activity, accessibility, **extra)


def _reference_index(request: DesignRequest, world):
    """The off-target index for this run.

    Defaults to the exhaustive wobble-aware scan over whatever was loaded, which
    can honestly report ``exhaustive=True``. ``--blast-db`` swaps in ``blastn``,
    which is faster over a large database and is a seeded heuristic, so it
    reports ``exhaustive=False`` and the run says so.
    """
    from .indexes import ExhaustiveScanIndex

    if not request.blast_db:
        return ExhaustiveScanIndex(world, ALL_UNIVERSES)

    from .blast import BlastOffTargetIndex, BlastUnavailable

    try:
        universes = frozenset(
            Universe(name.strip())
            for name in request.blast_universes.split(",")
            if name.strip()
        )
    except ValueError as exc:
        raise DesignRequestError(
            f"--blast-universes: {exc}. Valid values: "
            + ", ".join(sorted(u.value for u in Universe))
        ) from exc
    if not universes:
        raise DesignRequestError(
            "--blast-universes cannot be empty: the universe-adequacy gate reads "
            "it as your assertion about what the database covers"
        )

    index = BlastOffTargetIndex(database=request.blast_db, universes=universes)
    try:
        index.ensure_available()
    except BlastUnavailable as exc:
        raise DesignRequestError(str(exc)) from exc
    # Unioned with the local scan, never substituted for it. The same index
    # answers both of the pipeline's questions -- where does this guide bind
    # off-target, and does it bind its intended target at all -- and a BLAST
    # database can only answer the first: it never sees this run's
    # Transcriptome, and its subject ids are accessions that `annotate_hit`
    # cannot match to the run's target ids. Substituting it left every guide
    # with no on-target evidence, so `--blast-db` FAILed every candidate:
    # measured 39/39 FAIL against 28 REVIEW for the identical design without
    # the flag. A flag that fails everything looks exactly like a target with
    # no usable guides.
    from .indexes import UnionSearchIndex

    return UnionSearchIndex([ExhaustiveScanIndex(world, ALL_UNIVERSES), index])


def _build_arc_model(request: DesignRequest, spec, records):
    """The Arc CNN, or a refusal that says which precondition failed.

    Every check here is a configuration the network was never trained on, and
    the failure mode for all of them is the same: a confident number about a
    thing the model has not seen. So they raise instead of degrading.
    """
    from ..contract.enums import Effector
    from .arc_model import GUIDE_LENGTH, arc_model_for

    if spec is not None:
        if spec.profile.effector is not Effector.RFX_CAS13D:
            raise DesignRequestError(
                f"--arc needs RfxCas13d; this run is {spec.profile.effector.value}. "
                "The network was trained on RfxCas13d alone and has nothing to "
                "say about another ortholog."
            )
        if spec.spacer_length != GUIDE_LENGTH:
            raise DesignRequestError(
                f"--arc needs a {GUIDE_LENGTH} nt spacer; this run uses "
                f"{spec.spacer_length}. The input tensor is a fixed "
                f"({GUIDE_LENGTH}, 4), so another length cannot be scored at all."
            )
    try:
        from .contrafold import ContrafoldMFESource
    except ImportError as exc:  # pragma: no cover
        raise DesignRequestError(str(exc)) from exc

    try:
        source = ContrafoldMFESource()
    except Exception as exc:
        raise DesignRequestError(
            f"--arc needs LinearFold-CONTRAfold folding scores: {exc}"
        ) from exc

    sequences = {r.name: r.sequence for r in records}
    if not sequences:
        raise DesignRequestError("--arc needs at least one target sequence")
    try:
        return arc_model_for(sequences, source)
    except Exception as exc:
        raise DesignRequestError(f"could not build the Arc model: {exc}") from exc


def _read_fasta_or_fail(path, flag: str) -> list[SequenceRecord]:
    """``read_fasta``, with the failure typed the way callers are told it is.

    ``run_request`` documents that it raises :class:`DesignRequestError` on bad
    input, and ``cli._design`` catches exactly that to print a located message
    instead of a traceback. ``_read_records`` upheld it for ``--fasta`` and the
    other file arguments did not, so the identical typo produced a clean
    message on one flag and a stack trace on the next.
    """
    try:
        return read_fasta(path)
    except SequenceInputError as exc:
        raise DesignRequestError(f"{flag} {path}: {exc}") from exc


def _panel_dependence(request: DesignRequest):
    """The measured common-cause fraction for FR6.3, if one was supplied.

    ``None`` is a first-class answer here rather than a default value: the
    panel confidence model reports an interval without it, and a number with
    it. Nothing in this package knows what beta is for your assay.
    """
    if not request.panel_dependence_json:
        return None
    from .panel_confidence import PanelDependence

    try:
        return PanelDependence.from_json(request.panel_dependence_json)
    except (OSError, ValueError, TypeError) as exc:
        raise DesignRequestError(
            f"--panel-dependence {request.panel_dependence_json}: {exc}"
        ) from exc


def seed_calibration(request: DesignRequest, spec: ResolvedDesignSpec):
    """A stand-in calibration so bounds exist at all.

    Synthetic on purpose and *marked* as such -- ``calibration.synthetic`` is
    set below, and ``services/limitations.py`` reads that flag to raise the
    CRITICAL synthetic-calibration limitation. This used to point at
    :data:`STANDING_CAVEATS`, which is the empty tuple: the pointer outlived
    the constant it named by the commit that emptied it. Group labels are
    supplied so the group-adequacy check is actually exercised rather than
    switching itself off.
    """
    from .calibration import CalibrationService

    rng = random.Random(request.seed)
    calibration = CalibrationService()
    # Marked rather than described. services/limitations.py reads this flag, so
    # the warning appears because the calibration *is* synthetic, not because a
    # caller remembered to add a banner -- and it disappears when it stops being.
    calibration.synthetic = True
    # The parameters that generate this set, attached to the object that
    # carries it. Seed and noise decide every conformal interval, and therefore
    # every metric read at a bound, every verdict and the panel -- and they
    # appeared in no recorded field: not the ledger, not the fingerprint (they
    # are request knobs, not spec), not the export headers. Two runs differing
    # only in seed disagreed on 50 verdicts while carrying byte-identical
    # provenance, which is the one thing provenance exists to make impossible.
    calibration.source = {
        "kind": "synthetic",
        "seed": request.seed,
        "n_per_stratum": request.calibration_n,
        "noise_sd": request.calibration_noise,
    }
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

    notes: list[str] = []
    try:
        records = _read_records(request)
    except SequenceInputError as exc:
        raise DesignRequestError(str(exc)) from exc

    spec = build_spec(request, records, notes)
    registry = build_registry(request, notes, spec=spec, records=records)

    world = Transcriptome(request.genome_build, request.annotation)
    target_biotype = _target_biotype(request)
    for record in records:
        world.add(Locus(record.name, record.sequence, target_biotype,
                        gene=record.name, expression=request.expression))
    background_loci: list[Locus] = []
    # Every --background record used to be CODING whatever it was, so the two
    # highest hazard weights in DEFAULT_HAZARD_BY_BIOTYPE -- rRNA at 2.0 and
    # mito at 1.5 -- were unreachable from the CLI: a user who supplied rRNA
    # as off-target evidence had it weighted 1.0, like any transcript.
    background_biotype = Universe(request.background_biotype)
    if request.background_path:
        for record in _read_fasta_or_fail(request.background_path, "--background"):
            background_loci.append(Locus(
                record.name, record.sequence,
                # Per record, so one pulled corpus can hold the rRNA, mito and
                # ncRNA the assay is exposed to rather than one universe at a
                # time. See `_record_biotype`.
                _record_biotype(record, background_biotype, "--background"),
                gene=record.name, expression=None))
    # Captured before the loop, because the loop grows `world`: testing
    # membership against it made the *second* of two background records sharing
    # a name take the target-collision branch, which silently dropped a locus
    # the user supplied as off-target evidence and then told them a target was
    # involved when none was. Duplicate names in a pulled background corpus are
    # the ordinary concatenated-file case `services/fasta.py` refuses to
    # deduplicate, so this is a routine input, and the note travels into every
    # export header -- a false statement outliving the session.
    target_names = {record.name for record in records}
    seen_background: set[str] = set()
    #: The background records this run actually accepted. `background_loci` is
    #: everything the file held, including the records rejected below -- and it
    #: was the list handed to `variant_contexts`, so a record dropped from the
    #: reference world for colliding with a target's name was still added to
    #: every *variant* world, where the name does not collide. The run printed
    #: "the background copy ignored" and then searched the target against it:
    #: adding `--variants` to an otherwise identical request took
    #: worst_off_target from 0.000 to 0.571, with the note still saying
    #: otherwise. The filtered list is what the contexts get.
    accepted_background: list[Locus] = []
    for locus in background_loci:
        if locus.locus_id in seen_background:
            notes.append(
                f"--background holds more than one record named "
                f"{locus.locus_id!r}. Only the first is searched -- the others "
                "are dropped, because a world keys loci by name and cannot tell "
                "them apart. Rename them so each is its own locus."
            )
            continue
        seen_background.add(locus.locus_id)
        if locus.locus_id in target_names:
            # Transcriptome.add overwrites by locus_id, so a background record
            # sharing a target's name replaced the target's own sequence,
            # biotype and expression -- and `locus_to_target` still pointed at
            # it, so the run searched the target against something that was not
            # it. Measured: PASS 97 -> 0 from adding the target's own record to
            # the background FASTA, with nothing said. The route is ordinary: a
            # background corpus pulled from the same annotation contains the
            # target's transcript id. `variant_contexts` already guards this;
            # the non-variant path did not, so the two disagreed.
            notes.append(
                f"--background record {locus.locus_id!r} shares a name with a "
                "target record. The target is kept and the background copy "
                "ignored -- rename it if they are meant to be different loci."
            )
            continue
        world.add(locus)
        accepted_background.append(locus)

    reference_index = _reference_index(request, world)
    contexts = [
        DesignContext(
            "reference", world, reference_index,
            {record.name: record.name for record in records},
        )
    ]

    variant_records: list[SequenceRecord] = []
    if request.variants_path:
        variant_records = _read_fasta_or_fail(request.variants_path, "--variants")
    elif request.variants_text and request.variants_text.strip():
        try:
            variant_records = parse_sequence_input(
                request.variants_text, default_name="variant"
            )
        except SequenceInputError as exc:
            raise DesignRequestError(f"variants: {exc}") from exc
    if variant_records:
        from .conservation import summarize_conservation, variant_contexts

        combined = [records[0]] + variant_records
        try:
            summary = summarize_conservation(
                combined, spec.spacer_length, max_mismatches=request.conserved_mismatches
            )
        except ValueError as exc:
            raise DesignRequestError(f"--variants: {exc}") from exc
        notes.append(f"Conservation across {len(variant_records)} variant(s): "
                     f"{summary.summary()}")

        # The variant contexts REPLACE the reference one, so whatever the
        # caller loaded has to travel with them. It used to not: each variant
        # world held that variant and nothing else, and the index was rechosen
        # locally -- so `--background` was parsed and dropped and `--blast-db`
        # was validated (ensure_available() and all) and never queried. Adding
        # `--variants` to an otherwise identical request turned an off-target
        # report into a vacuously clean one, which is the worst direction for
        # this particular number to be wrong in.
        def index_for(variant_world):
            # Rebuilt per world, because the variant sequence exists only
            # inside that world. Reusing the reference index looked safe -- a
            # BLAST database really is corpus-independent -- and was exactly
            # wrong for that reason: an index that cannot see the context's own
            # loci cannot find the variant, so robustness (a min of coverage
            # across contexts) collapsed to 0.000 for every guide.
            return _reference_index(request, variant_world)

        # Wrapped for the same reason summarize_conservation above is: both
        # refuse a variant set they cannot tell apart, and both are reached
        # with user-supplied FASTA, so the refusal has to arrive as the type
        # run_request documents rather than as a ValueError from two modules
        # down.
        try:
            contexts = variant_contexts(
                combined, ALL_UNIVERSES, target_id=records[0].name,
                # The same biotype the reference world files the target under.
                # Hardcoding NCRNA here while `_target_biotype` decided it
                # there gave the target two universes: adding --variants to an
                # otherwise identical cell_culture run took 223 PASS to 223
                # REVIEW, reporting `coding` as a universe the search did not
                # cover, because no variant world held a coding locus.
                biotype=target_biotype, expression=request.expression,
                background=accepted_background, extra_targets=records[1:],
                index_for=index_for,
                genome_build=request.genome_build, annotation=request.annotation,
            )
        except ValueError as exc:
            raise DesignRequestError(f"--variants: {exc}") from exc

    calibration_service = seed_calibration(request, spec)
    from .blast import BlastUnavailable
    from .kinetics import StiffKineticsError

    try:
        report = run_design(
            spec, contexts, registry, calibration_service,
            panel_size=request.panel_size,
            panel_dependence=_panel_dependence(request),
        )
    except StiffKineticsError as exc:
        # Raised from inside stage E, long after the JSON was validated, so it
        # escaped run_request untyped and the CLI printed a traceback. The
        # trigger is not exotic: kinetics.py names k_on = 1 /nM/s as an
        # ordinary measured association rate, and --kinetics-constants is
        # advertised as the flag that unlocks absolute signal -- so the flag
        # that is supposed to reward real constants punished them.
        raise DesignRequestError(
            f"--kinetics-constants {request.kinetics_json}: {exc}"
        ) from exc
    except BlastUnavailable as exc:
        # The same shape as the kinetics case above, one subsystem over, and
        # missed when that one was fixed. `_reference_index` guards blastn's
        # absence at *construction*; an ambiguity code in a subject is raised
        # at *search* time, outside that guard, so it escaped as a traceback
        # and exit 1 -- the status documented as "nothing survived the gate".
        # The database is third-party and the user cannot edit it, so this is
        # reachable on the module's intended use.
        raise DesignRequestError(str(exc)) from exc
    data = report.to_dict()
    weakest_bound = report.search_bound
    gaps = report.universe_gaps

    profile: tuple[float, ...] = ()
    source = getattr(registry.accessibility, "unpaired_profile", None)
    if source is not None and records:
        try:
            profile = tuple(source(records[0].sequence))
        except Exception:
            profile = ()

    from .limitations import as_lines, limitations_for_run

    found = limitations_for_run(
        spec,
        registry=registry,
        calibration=calibration_service,
        bound=weakest_bound,
        universe_gaps=gaps,
        self_identical_loci=report.self_identical_loci,
        panel_size=len(data.get("panel", {}).get("guides", ())),
        panel_confidence=report.panel_confidence,
        provenance=data.get("provenance"),
    )
    notes = as_lines(found) + notes

    # The terminal, the HTML report and the export headers all carried these.
    # The machine-readable report did not -- `report.json` had no limitations
    # key at all, so a pipeline reading it saw PASS verdicts and calibrated
    # intervals with nothing saying the calibration was synthetic and every
    # PASS therefore exercisable rather than earned. That is the one surface
    # where a caveat cannot be inferred from tone, and it was the only surface
    # without them.
    #
    # Structured rather than rendered: `notes` is the prose form and is already
    # in the exports. A caller switching on severity should not have to parse
    # "CRITICAL: " off the front of a sentence.
    data["limitations"] = [
        {
            "id": limitation.id,
            "severity": limitation.severity.value,
            "text": limitation.text,
            "source": limitation.source,
        }
        for limitation in found
    ]

    return DesignOutcome(
        report=data, records=records, spec=spec, notes=notes, profile=profile,
        limitations=found,
    )
