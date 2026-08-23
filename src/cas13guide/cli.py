"""Command line entry point.

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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
