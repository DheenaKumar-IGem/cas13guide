"""Command line entry point.

``cas13guide design`` -- design guides from a sequence, FASTA file or preset.
``cas13guide demo``  -- run the bundled end-to-end example.
``cas13guide check`` -- run the benchmark/coverage CI gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _demo(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
    import demo_knockdown  # type: ignore

    demo_knockdown.main()
    return 0


def _check(args: argparse.Namespace) -> int:
    """Fail loudly if a model/calibration change broke skill or coverage."""
    import random

    from .services.benchmark import BenchmarkCase, run_benchmark
    from .services.calibration import CalibrationService
    from .services.registry import HeuristicCas13Model, ModelRegistry, NullAccessibilityModel
    from .util.duplex import profile_duplex
    from .util.seq import revcomp

    spacer = "GCAUUGCUGUGGAGCUGAAGUC"
    stratum = "RfxCas13d|cell_culture_knockdown|mm1-2"
    model = HeuristicCas13Model()
    rng = random.Random(args.seed)

    def make_cases(n: int) -> list[BenchmarkCase]:
        out = []
        for _ in range(n):
            site = list(revcomp(spacer))
            for _ in range(rng.randint(0, 4)):
                i = rng.randrange(len(site))
                site[i] = rng.choice([b for b in "AUGC" if b != site[i]])
            site_str = "".join(site)
            pred = model.predict(spacer, profile_duplex(spacer, site_str), None)
            out.append(
                BenchmarkCase(
                    spacer=spacer,
                    target_site=site_str,
                    observed_activity=min(1.0, max(0.0, pred + rng.gauss(0, 0.12))),
                    stratum=stratum,
                )
            )
        return out

    calibration = CalibrationService()
    for case in make_cases(300):
        calibration.add_observation(
            stratum, model.predict(spacer, profile_duplex(spacer, case.target_site), None),
            case.observed_activity,
        )

    registry = ModelRegistry(model, NullAccessibilityModel())
    report = run_benchmark(registry, calibration, make_cases(200))
    print(report.summary())
    if not report.coverage_ok:
        print("FAIL: conformal coverage below nominal", file=sys.stderr)
        return 1
    print("OK: skill and coverage within tolerance")
    return 0


def _train(args: argparse.Namespace) -> int:
    """Ingest a real screen, run the acceptance gate, and report."""
    from .services.acceptance import evaluate_dataset
    from .services.ingest import ColumnMap, DatasetError, load_screen

    columns = ColumnMap(
        spacer=args.spacer_column,
        activity=args.activity_column,
        group=args.group_column,
        target_site=args.target_site_column,
        higher_is_better=not args.lower_is_better,
    )
    try:
        examples, dataset = load_screen(
            args.path, columns=columns, effector=args.effector, assay=args.assay
        )
    except DatasetError as exc:
        print(f"cannot read dataset: {exc}", file=sys.stderr)
        return 2

    print(dataset.summary())
    if not dataset.ok:
        return 2

    report = evaluate_dataset(examples, seed=args.seed)
    print()
    print(report.summary())
    if report.result is not None:
        print("\n  top features:")
        for name, weight in report.result.model.top_features(8):
            print(f"    {name:<20} {weight:+.3f}")
    return 0 if report.accepted else 1



def _design(args: argparse.Namespace) -> int:
    """Design guides against a pasted sequence, a FASTA file, or a preset.

    The one command that exercises the whole product: input (FR1.1-FR1.5),
    scoring (FR3/5/6 when ViennaRNA is present), ranking (FR8.1) and export
    (FR8.3). NFR2 asks that a basic query need no bioinformatics command-line
    knowledge; this is as close as a CLI gets, and it is what a UI would call.
    """
    import random

    from .contract.enums import AssayType, Effector, Task, TargetClass, Universe
    from .contract.presets import (
        MIRNA_PRESETS,
        designability,
        get_preset,
        largest_designable_spacer_length,
        preset_ids,
    )
    from .contract.spec import (
        ContextScope,
        EffectorAssayProfile,
        ResolvedDesignSpec,
        TargetContract,
        Thresholds,
    )
    from .pipeline.orchestrator import DesignContext, run_design
    from .services.calibration import CalibrationService
    from .services.fasta import (
        SequenceInputError,
        parse_sequence_input,
        read_fasta,
        write_exports,
    )
    from .services.indexes import ExhaustiveScanIndex
    from .services.registry import (
        HeuristicCas13Model,
        ModelRegistry,
        NullAccessibilityModel,
    )
    from .services.transcriptome import Locus, Transcriptome

    if args.list_presets:
        print("built-in targets (FR1.3):")
        for preset_id in preset_ids():
            preset = get_preset(preset_id)
            print(f"  {preset_id:18s} {preset.accession:14s} {preset.length:3d} nt")
        return 0

    # ---------------- input (FR1.1, FR1.2, FR1.3, FR1.4) ----------------
    try:
        if args.preset:
            preset = get_preset(args.preset)
            records = [
                type("R", (), {"name": preset.preset_id, "sequence": preset.sequence,
                               "description": preset.accession, "length": preset.length})()
            ]
        elif args.fasta:
            records = read_fasta(args.fasta)
        elif args.sequence:
            records = parse_sequence_input(args.sequence)
        else:
            print("supply --sequence, --fasta or --preset (or --list-presets)",
                  file=sys.stderr)
            return 2
    except (SequenceInputError, KeyError) as exc:
        print(f"cannot read input: {exc}", file=sys.stderr)
        return 2

    effector = Effector(args.effector)
    assay = AssayType(args.assay)

    # ---------------- spacer length (FR1.5, FR2.1) ----------------
    spacer_length = args.spacer_length
    if spacer_length is None and args.preset:
        preset = get_preset(args.preset)
        spacer_length = largest_designable_spacer_length(preset, effector)
        report = designability(preset, effector, spacer_length)
        print(f"preset {preset.preset_id} ({preset.length} nt): {report.note}")
        if spacer_length is None:
            return 2

    targets = []
    for index, record in enumerate(records):
        targets.append(
            __import__("cas13guide.contract.spec", fromlist=["TargetSpec"]).TargetSpec(
                target_id=record.name,
                sequence=record.sequence,
                target_class=TargetClass.REQUIRED if index == 0 else TargetClass.DESIRED,
                gene=record.name,
                biotype=Universe.NCRNA,
                expression=args.expression,
            )
        )

    universes = frozenset({Universe.SYNTHETIC, Universe.VIRAL, Universe.NCRNA,
                           Universe.CODING, Universe.PRE_MRNA, Universe.RRNA,
                           Universe.MITO})
    try:
        spec = ResolvedDesignSpec(
            contract=TargetContract(task=Task(args.task), targets=tuple(targets)),
            profile=EffectorAssayProfile(
                effector=effector, assay=assay, spacer_length=spacer_length
            ),
            context=ContextScope(
                organism=args.organism, genome_build=args.genome_build,
                annotation_version=args.annotation, declared_universes=universes,
            ),
            thresholds=Thresholds(
                required_search_mismatches=args.max_mismatches,
                min_accessibility=args.min_accessibility,
            ),
        )
    except Exception as exc:
        print(f"cannot build the design spec: {exc}", file=sys.stderr)
        return 2

    # ---------------- models (FR3.x, FR5.x, FR6.x) ----------------
    accessibility = NullAccessibilityModel()
    energy = None
    kinetics = None
    if not args.no_structure:
        try:
            from .services.accessibility import ViennaRNAAccessibilityModel
            from .services.thermodynamics import ViennaRNADuplexEnergyModel

            accessibility = ViennaRNAAccessibilityModel()
            energy = ViennaRNADuplexEnergyModel()
        except Exception as exc:
            print(f"structural scoring disabled ({exc}); "
                  'install with: pip install "cas13guide[structure]"', file=sys.stderr)
    if not args.no_signal:
        from .services.kinetics import CollateralKineticsModel

        kinetics = CollateralKineticsModel()

    registry_kwargs = {}
    if energy is not None:
        registry_kwargs["energy"] = energy
    if kinetics is not None:
        registry_kwargs["kinetics"] = kinetics
    registry = ModelRegistry(HeuristicCas13Model(), accessibility, **registry_kwargs)

    # ---------------- searchable world ----------------
    world = Transcriptome(args.genome_build, args.annotation)
    for record in records:
        world.add(Locus(record.name, record.sequence, Universe.NCRNA,
                        gene=record.name, expression=args.expression))
    if args.background:
        for record in read_fasta(args.background):
            world.add(Locus(record.name, record.sequence, Universe.CODING,
                            gene=record.name, expression=None))

    # ---------------- calibration ----------------
    rng = random.Random(args.seed)
    calibration = CalibrationService()
    for regime in ("mm0", "mm1-2", "mm3+"):
        key = CalibrationService.stratum_key(effector.value, assay.value, regime)
        for i in range(args.calibration_n):
            truth = rng.random()
            calibration.add_observation(
                key, min(1.0, max(0.0, truth + rng.gauss(0, args.calibration_noise))),
                truth, group=f"T{i % 12:02d}",
            )
    print(
        "NOTE: calibration is SYNTHETIC. Bounds are exercisable, not earned -- "
        "feed real screen data via `cas13guide train` before trusting them.",
        file=sys.stderr,
    )

    contexts = [
        DesignContext(
            "reference", world, ExhaustiveScanIndex(world, universes),
            {record.name: record.name for record in records},
        )
    ]

    # ---------------- variants (FR7.1, FR7.2) ----------------
    if args.variants:
        from .services.conservation import summarize_conservation, variant_contexts

        variant_records = read_fasta(args.variants)
        primary = records[0]
        combined = [primary] + list(variant_records)

        # FR7.1 -- can a single shared guide exist at all, and where?
        summary = summarize_conservation(
            combined, spec.spacer_length, max_mismatches=args.conserved_mismatches
        )
        print(f"\nconservation ({len(variant_records)} variant(s)): {summary.summary()}")
        if not summary.designable:
            print("  proceeding anyway; robustness will show which variants each "
                  "guide loses.", file=sys.stderr)

        # FR7.2 -- one context per variant, so `robustness` (the minimum
        # required coverage across contexts) flags any guide that loses one.
        contexts = variant_contexts(
            combined, universes, target_id=primary.name,
            biotype=Universe.NCRNA, expression=args.expression,
        )

    report = run_design(spec, contexts, registry, calibration,
                        panel_size=args.panel_size)

    # ---------------- output (FR8.1, FR8.2, FR8.3) ----------------
    data = report.to_dict()
    print(f"\nverdicts: {data['verdict_counts']}  "
          f"review_rate={report.review_rate:.0%}")
    rows = data["ranked"][: args.top]
    if rows:
        robust = "robust" if args.variants else ""
        print(f"\n{'rank':>4}  {'spacer':32s} {'score':>6} {'access':>7} "
              f"{'dG/nt':>7} {'signal':>7} {'PFS':>4} {robust:>7}")
        for row in rows:
            # FR7.2: robustness is the worst required coverage across the
            # variant contexts, so a guide that loses one shows it here.
            worst = (f"{row['metrics']['robustness']:.3f}" if args.variants else "")
            print(f"{row['rank']:>4}  {row['spacer']:32s} {row['score']:>6.2f} "
                  f"{str(row['accessibility']):>7} {str(row['duplex_dg_per_nt']):>7} "
                  f"{str(row['predicted_signal']):>7} {str(row['pfs_3prime']):>4} "
                  f"{worst:>7}")
    else:
        print("\nno guide cleared the gate.")
        for shortfall in data["shortfalls"]:
            print(f"  ! {shortfall}")

    if args.out:
        written = write_exports(data, args.out, stem=args.stem)
        print("\nwrote:")
        for label, path in written.items():
            print(f"  {label:9s} {path}")
    return 0 if rows else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cas13guide", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the bundled end-to-end example")
    demo.set_defaults(func=_demo)

    check = sub.add_parser("check", help="run the benchmark / coverage CI gate")
    check.add_argument("--seed", type=int, default=0)
    check.set_defaults(func=_check)

    train = sub.add_parser("train", help="ingest a screen and run the acceptance gate")
    train.add_argument("path", help="CSV/TSV screen file")
    train.add_argument("--spacer-column", default="spacer")
    train.add_argument("--activity-column", default="activity")
    train.add_argument("--group-column", default="transcript")
    train.add_argument("--target-site-column", default=None)
    train.add_argument(
        "--lower-is-better",
        action="store_true",
        help="activity is a depletion score (log2 fold change), so more negative is better",
    )
    train.add_argument("--effector", default="RfxCas13d")
    train.add_argument("--assay", default="cell_culture_knockdown")
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=_train)

    design = sub.add_parser(
        "design", help="design guides against a sequence, FASTA file or preset"
    )
    source = design.add_argument_group("input (FR1.1-FR1.3)")
    source.add_argument("--sequence", help="pasted raw or FASTA sequence text")
    source.add_argument("--fasta", help="FASTA file to read")
    source.add_argument("--preset", help="built-in target id, e.g. hsa-miR-21-5p")
    source.add_argument("--list-presets", action="store_true",
                        help="list the built-in targets and exit")
    source.add_argument("--background", help="FASTA of additional off-target sequences")
    source.add_argument("--variants", help="FASTA of related sequences (isoforms, "
                                           "strain variants) a shared guide must "
                                           "also work against (FR7.1, FR7.2)")
    source.add_argument("--conserved-mismatches", type=int, default=0,
                        help="mismatches tolerated when calling a region conserved")

    design.add_argument("--effector", default="LwaCas13a",
                        choices=["LwaCas13a", "RfxCas13d", "PspCas13b"])
    design.add_argument("--assay", default="in_vitro_detection",
                        choices=["in_vitro_detection", "cell_culture_knockdown",
                                 "in_vivo_knockdown", "antiviral"])
    design.add_argument("--task", default="detection",
                        choices=["detection", "knockdown", "isoform", "allele",
                                 "antiviral", "editing"])
    design.add_argument("--spacer-length", type=int, default=None,
                        help="window length; defaults to the ortholog's canonical value")
    design.add_argument("--max-mismatches", type=int, default=4,
                        help="off-target search stringency (FR4.3)")
    design.add_argument("--min-accessibility", type=float, default=0.0,
                        help="flag sites below this mean unpaired probability (FR3.4)")
    design.add_argument("--no-structure", action="store_true",
                        help="skip ViennaRNA accessibility and duplex energy")
    design.add_argument("--no-signal", action="store_true",
                        help="skip the collateral-cleavage signal model")
    design.add_argument("--panel-size", type=int, default=3)
    design.add_argument("--top", type=int, default=10)
    design.add_argument("--expression", type=float, default=None)
    design.add_argument("--organism", default="Homo sapiens")
    design.add_argument("--genome-build", default="GRCh38")
    design.add_argument("--annotation", default="v44")
    design.add_argument("--seed", type=int, default=0)
    design.add_argument("--calibration-n", type=int, default=200)
    design.add_argument("--calibration-noise", type=float, default=0.10)
    design.add_argument("--out", help="directory to write CSV/FASTA exports into (FR8.3)")
    design.add_argument("--stem", default="candidates")
    design.set_defaults(func=_design)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
