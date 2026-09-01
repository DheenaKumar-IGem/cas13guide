"""Command line entry point.

``cas13guide design`` -- design guides from a sequence, FASTA file or preset.
``cas13guide serve``  -- the same thing, in a browser.
``cas13guide demo``  -- run the bundled end-to-end example.
``cas13guide check`` -- run the benchmark/coverage CI gate.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from pathlib import Path


def _demo(args: argparse.Namespace) -> int:
    # Imported from the package, not fished out of an ``examples/`` directory
    # two levels up. That path exists only in a source checkout and does not
    # ship in the wheel, so for anyone who pip-installed this package the
    # subcommand advertised in ``--help`` died with ModuleNotFoundError and
    # exit 1. Found by installing the wheel into a clean interpreter and
    # running the documented commands from outside the source tree.
    from .demo import main as run_demo

    run_demo()
    return 0


def _check(args: argparse.Namespace) -> int:
    """Fail loudly if a calibration change broke conformal coverage.

    Skill (spearman, mae) is measured and printed, and deliberately not gated
    here: there is no stored baseline to compare against, and a gate with
    nothing on the other side of it is a green light rather than a check.
    ``services/benchmark.assert_no_regression`` is the gate for that, and it
    needs a baseline the caller supplies.
    """
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
    # Names the gate that actually ran. Spearman and MAE are computed and
    # printed above, and nothing compares them to a baseline --
    # `benchmark.assert_no_regression` exists for that and this command does
    # not call it -- so claiming skill was "within tolerance" certified a check
    # that never happened.
    print("OK: conformal coverage within tolerance (skill is reported, not gated)")
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

    A thin wrapper over ``services.workflow.run_request``: the CLI and the web
    server share that one code path so a form POST and a command line cannot
    drift into behaving differently.
    """
    from .contract.presets import get_preset, preset_ids
    from .services.fasta import write_exports
    from .services.workflow import DesignRequest, DesignRequestError, run_request

    if args.list_presets:
        print("built-in targets (FR1.3):")
        for preset_id in preset_ids():
            preset = get_preset(preset_id)
            print(f"  {preset_id:18s} {preset.accession:14s} {preset.length:3d} nt")
        return 0

    # Checked before the pipeline runs, so a typo costs no compute -- and
    # because both failures used to arrive as a pathlib traceback after the
    # work was already done.
    if args.out:
        destination = pathlib.Path(args.out)
        if destination.exists() and not destination.is_dir():
            print(
                f"cannot write the exports: --out must name a directory; "
                f"{destination} is a file",
                file=sys.stderr,
            )
            return 2
    # `--stem` names a file, not a path. "../escaped" wrote all four exports
    # outside the directory the user named, which is not what a stem is for
    # and not something a user asking for a filename prefix can expect.
    if pathlib.Path(args.stem).name != args.stem or args.stem in ("", ".", ".."):
        print(
            f"cannot write the exports: --stem must be a plain file name, not a "
            f"path; got {args.stem!r}",
            file=sys.stderr,
        )
        return 2

    request = DesignRequest(
        sequence_text=args.sequence,
        fasta_path=args.fasta,
        preset=args.preset,
        variants_path=args.variants,
        background_path=args.background,
        effector=args.effector,
        assay=args.assay,
        task=args.task,
        spacer_length=args.spacer_length,
        max_mismatches=args.max_mismatches,
        min_accessibility=args.min_accessibility,
        conserved_mismatches=args.conserved_mismatches,
        panel_size=args.panel_size,
        expression=args.expression,
        organism=args.organism,
        genome_build=args.genome_build,
        annotation=args.annotation,
        use_structure=not args.no_structure,
        use_signal=not args.no_signal,
        seed=args.seed,
        calibration_n=args.calibration_n,
        calibration_noise=args.calibration_noise,
        kinetics_json=args.kinetics_constants,
        panel_dependence_json=args.panel_dependence,
        use_arc=args.arc,
        blast_db=args.blast_db,
        blast_universes=args.blast_universes,
    )
    try:
        outcome = run_request(request)
    except DesignRequestError as exc:
        print(f"cannot run this design: {exc}", file=sys.stderr)
        return 2

    for note in outcome.notes:
        print(note, file=sys.stderr)

    data = outcome.report
    print()
    print(f"verdicts: {data['verdict_counts']}  "
          f"review_rate={data['review_rate']:.0%}")

    all_rows = data["ranked"]
    # Default to one row per distinct site. The raw ranking is dominated by
    # single-base shifts of the same window -- see f_decide.distinct_site_ids.
    shown = all_rows if args.all_sites else [r for r in all_rows if r["distinct_site"]]
    rows = shown[: args.top]
    collapsed = len(all_rows) - len(shown)
    if collapsed and not args.all_sites:
        print(
            f"showing one guide per distinct site; {collapsed} shifted "
            "copies of a listed site are hidden (--all-sites to show them)",
            file=sys.stderr,
        )
    if rows:
        robust = "robust" if args.variants else ""
        print()
        print(f"{'rank':>4}  {'spacer':32s} {'score':>6} {'access':>7} "
              f"{'dG/nt':>7} {'signal':>7} {'PFS':>4} {robust:>7}")
        for row in rows:
            worst = (f"{row['metrics']['robustness']:.3f}" if args.variants else "")
            print(f"{row['rank']:>4}  {row['spacer']:32s} {row['score']:>6.2f} "
                  f"{str(row['accessibility']):>7} {str(row['duplex_dg_per_nt']):>7} "
                  f"{str(row['predicted_signal']):>7} {str(row['pfs_3prime']):>4} "
                  f"{worst:>7}")
    else:
        print()
        print("no guide cleared the gate.")

    # Shortfalls print whether or not the table is empty. They used to print
    # only when nothing ranked, which is the case where a reader already knows
    # something went wrong. The dangerous case is the other one: 264 ranked
    # rows, an empty panel, and a terminal that mentions neither -- the run
    # recommends nothing and looks like a success. The report and the export
    # headers always carried these; only the terminal did not.
    panel = data.get("panel") or {}
    messages = list(panel.get("dropped", ())) + list(data.get("shortfalls", ()))
    if rows and not panel.get("guides"):
        messages.insert(
            0,
            "no panel: guides ranked but none PASSed the gate, and the panel "
            "is PASS-only. The rows above are candidates to triage, not a "
            "recommendation.",
        )
    for message in messages:
        print(f"  ! {message}")

    # FR6.3. Printed after the table because it is a statement about the
    # panel the run recommends, not about the rows a human is triaging -- and
    # printed as an interval because that is what it is.
    confidence = (data.get("panel") or {}).get("confidence")
    if confidence and confidence["targets"]:
        print()
        print("panel detection probability (FR6.3):")
        for target in confidence["targets"]:
            point = (
                "" if target["point"] is None
                else f"  point={target['point']:.3f}"
            )
            thin = "  [thin calibration]" if target["thin"] else ""
            print(
                f"  {target['target_id']:<16} "
                f"[{target['lower']:.3f}, {target['upper']:.3f}]"
                f"{point}{thin}"
            )
        print("  lower = members fail together, upper = members fail "
              "independently")

    if args.out:
        # Every path problem here used to arrive as a pathlib traceback and
        # exit 1 -- the same code `design` returns for the legitimate "no
        # guide cleared the gate" outcome, so a wrapper script could not tell
        # "nothing ranked" from "could not write the report". The pipeline's
        # own failures have had a located message and exit 2 all along; the
        # filesystem's did not.
        try:
            from .services.report_html import AccessibilityTrack, track_rows, write_report

            tracks = []
            if outcome.profile:
                tracks = [
                    AccessibilityTrack(
                        name=f"{outcome.records[0].name} "
                             f"({len(outcome.records[0].sequence)} nt)",
                        profile=outcome.profile,
                        # Only guides tiled from the record this track depicts.
                        # `start` is an offset on the guide's OWN source target, and
                        # a multi-record FASTA makes records 1+ DESIRED targets that
                        # are tiled too -- so an unfiltered top-N drew guides from a
                        # paralog at their own coordinates on the primary's profile:
                        # marks shifted by the other record's offset, and marks past
                        # the end of the transcript silently clipped out of the
                        # picture. Measured: all five default marks foreign, one of
                        # them 6 nt beyond a 150 nt track.
                        guides=[
                            (r.get("start", 0), outcome.spec.spacer_length,
                             f"#{r['rank']} {r['guide_id']}")
                            for r in track_rows(data["ranked"], outcome.records[0].name,
                                                args.track_guides)
                        ],
                    )
                ]
            html_path = write_report(
                data,
                pathlib.Path(args.out) / f"{args.stem}.html",
                title=f"Cas13 guide design - {outcome.records[0].name}",
                subtitle=(
                    f"{request.effector} / {request.assay} / "
                    f"{outcome.spec.spacer_length} nt spacer  |  "
                    f"spec {outcome.spec.fingerprint()}"
                ),
                tracks=tracks,
                notes=outcome.notes,
            )
            subtitle = (
                f"{request.effector} / {request.assay} / "
                f"{outcome.spec.spacer_length} nt spacer"
            )
            written = write_exports(
                data, args.out, stem=args.stem,
                subtitle=subtitle, limitations=outcome.notes,
            )
            written["html"] = html_path
            print()
            print("wrote:")
            for label, path in written.items():
                print(f"  {label:9s} {path}")
        except OSError as exc:
            target = getattr(exc, "filename", None) or args.out
            print(
                f"cannot write the exports: {exc.strerror or exc}: {target} "
                "(from --out/--stem)",
                file=sys.stderr,
            )
            return 2
    return 0 if rows else 1


def _serve(args: argparse.Namespace) -> int:
    """Serve the browser UI (FR1.2, NFR2)."""
    from .services.webapp import serve

    return serve(host=args.host, port=args.port, open_browser=not args.no_browser)


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
    design.add_argument(
        "--kinetics-constants", metavar="PATH",
        help="JSON of measured kinetic constants (FR6.1). Without one the "
             "signal model runs on placeholders and absolute predictions "
             "stay refused; see KineticConstants.to_json for the schema")
    design.add_argument(
        "--blast-db", metavar="PATH",
        help="BLAST nucleotide database for off-target search (FR4, DR1/DR3). "
             "Without one, --background is scanned exhaustively instead, which "
             "is slower but can report exhaustive=True")
    design.add_argument(
        "--blast-universes", default="coding", metavar="A,B",
        help="what you assert --blast-db covers; the universe-adequacy gate "
             "reads this and cannot verify it (default: coding)")
    design.add_argument(
        "--arc", action="store_true",
        help="score activity with the Arc RNAtargeting CNN instead of the "
             "built-in heuristic. Needs RfxCas13d and --spacer-length 30, and "
             'pip install "cas13guide[arc,contrafold]"')
    design.add_argument("--no-signal", action="store_true",
                        help="skip the collateral-cleavage signal model")
    design.add_argument("--panel-size", type=int, default=3)
    design.add_argument(
        "--panel-dependence", metavar="FILE.json",
        help="measured common-cause fraction for panel members (FR6.3). "
             "Without one the panel's detection probability is reported as an "
             "interval and no point estimate is offered",
    )
    design.add_argument("--top", type=int, default=10)
    design.add_argument("--all-sites", action="store_true",
                        help="include guides that are shifted copies of a "
                             "higher-ranked site (hidden by default)")
    design.add_argument("--expression", type=float, default=None)
    design.add_argument("--organism", default="Homo sapiens")
    design.add_argument("--genome-build", default="GRCh38")
    design.add_argument("--annotation", default="v44")
    design.add_argument("--seed", type=int, default=0)
    design.add_argument("--calibration-n", type=int, default=200)
    design.add_argument("--calibration-noise", type=float, default=0.10)
    design.add_argument("--out", help="directory to write CSV/FASTA exports into (FR8.3)")
    design.add_argument("--stem", default="candidates")
    design.add_argument("--track-guides", type=int, default=5,
                        help="how many top guides to mark on the accessibility track")
    design.set_defaults(func=_design)

    serve = sub.add_parser("serve", help="serve the browser UI (FR1.2, NFR2)")
    serve.add_argument("--host", default="127.0.0.1",
                       help="bind address; anything but localhost is exposed to "
                            "the network and this is a development server")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--no-browser", action="store_true",
                       help="do not open a browser window")
    serve.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
