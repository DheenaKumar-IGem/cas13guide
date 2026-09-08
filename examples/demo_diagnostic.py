"""Diagnostic guide design against a miRNA preset (FR1.3, FR3.x, FR5.x, FR6.x).

Runs the full detection stack -- structural accessibility from ViennaRNA,
guide:target duplex free energy, and the collateral-cleavage signal model --
against one of the built-in miRNA targets, and shows the two things that make
this a *diagnostic* designer rather than a knockdown one:

* predicted signal is only an objective when collateral cleavage is the assay's
  readout; in a knockdown assay it is a hazard and does not rank;
* the mature miRNA itself is too short to tile, so the amplicon path is the one
  that produces a usable panel.

Requires ViennaRNA::

    pip install -e ".[structure]"
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cas13guide.contract.enums import AssayType, Effector, Task, Universe
from cas13guide.contract.presets import (
    MIRNA_PRESETS,
    PRESET_SOURCE,
    amplicon_target,
    designability,
    get_preset,
    largest_designable_spacer_length,
)
from cas13guide.contract.spec import (
    ContextScope,
    EffectorAssayProfile,
    ResolvedDesignSpec,
    TargetContract,
    Thresholds,
)
from cas13guide.pipeline.orchestrator import DesignContext, run_design
from cas13guide.services.limitations import as_lines
from cas13guide.services.calibration import CalibrationService
from cas13guide.services.indexes import ExhaustiveScanIndex
from cas13guide.services.kinetics import CollateralKineticsModel
from cas13guide.services.registry import HeuristicCas13Model, ModelRegistry
from cas13guide.services.transcriptome import Locus, Transcriptome
from cas13guide.services.workflow import build_synthetic_calibration

RNG = random.Random(5)
UNIVERSES = frozenset({Universe.SYNTHETIC, Universe.VIRAL, Universe.NCRNA})


def seed_calibration(spec: ResolvedDesignSpec) -> CalibrationService:
    """A stand-in, from the one builder that makes them.

    This was a third hand-rolled copy. It kept the group labels the design
    path supplies but not the `synthetic` flag or the `source` mapping, so
    `services/limitations.py` could not raise the CRITICAL saying every
    interval here rests on generated data, and the provenance ledger recorded
    this calibration as "unstated". Neither showed, because the script printed
    no limitations and no ledger.

    It also drew from RNG, the module-level generator this file uses for its
    sequences, so the seed was not on its own enough to rebuild it.
    """
    return build_synthetic_calibration(
        spec.profile.effector.value, spec.profile.assay.value,
        seed=5, n_per_stratum=200, noise_sd=0.1)


def main() -> None:
    try:
        from cas13guide.services.accessibility import ViennaRNAAccessibilityModel
        from cas13guide.services.thermodynamics import ViennaRNADuplexEnergyModel
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"ViennaRNA is required for this demo: {exc}")
        print('Install it with:  pip install -e ".[structure]"')
        raise SystemExit(1)

    preset = get_preset("hsa-miR-21-5p")
    print(f"target   : {preset.preset_id}  {preset.accession}  ({preset.length} nt)")
    verified = PRESET_SOURCE.get("verified", "unknown")
    print(f"source   : {PRESET_SOURCE['database']} ({PRESET_SOURCE['url']})")
    print(f"verified : {verified}")

    # ---- 1. why the mature strand is not the thing to design against ----
    print("\n--- geometry ---")
    print(" ", designability(preset, Effector.LWA_CAS13A).note)
    best = largest_designable_spacer_length(preset, Effector.LWA_CAS13A)
    print(f"  largest spacer that fits the mature strand: {best} nt "
          f"-> {designability(preset, Effector.LWA_CAS13A, best).windows} window(s)")

    # ---- 2. the amplicon a real assay would actually detect ----
    target = amplicon_target(
        preset,
        "".join(RNG.choice("AUGC") for _ in range(70)),
        "".join(RNG.choice("AUGC") for _ in range(70)),
    )
    print(f"\n  amplicon target: {len(target.sequence)} nt "
          f"-> designs normally at the canonical 28 nt")

    spec = ResolvedDesignSpec(
        contract=TargetContract(task=Task.DETECTION, targets=(target,)),
        profile=EffectorAssayProfile(
            effector=Effector.LWA_CAS13A, assay=AssayType.IN_VITRO_DETECTION
        ),
        context=ContextScope(
            organism="Homo sapiens", genome_build="GRCh38",
            annotation_version="v44", declared_universes=UNIVERSES,
        ),
        thresholds=Thresholds(
            min_accessibility=0.35,
            duplex_dg_per_nt_min=-2.2,
            duplex_dg_per_nt_max=-1.2,
        ),
    )

    world = Transcriptome("GRCh38", "v44")
    world.add(Locus("L1", target.sequence, Universe.NCRNA, gene=preset.display_name,
                    expression=50.0))
    world.add(Locus("BG", "".join(RNG.choice("AUGC") for _ in range(3000)),
                    Universe.SYNTHETIC, expression=5.0))
    context = DesignContext(
        "reference", world, ExhaustiveScanIndex(world, UNIVERSES), {"L1": target.target_id}
    )

    registry = ModelRegistry(
        HeuristicCas13Model(),
        ViennaRNAAccessibilityModel(),
        ViennaRNADuplexEnergyModel(),
        CollateralKineticsModel(),
    )
    print("\n--- models ---")
    for name, version in registry.versions().items():
        print(f"  {name:14s} {version}")

    started = time.perf_counter()
    report = run_design(spec, [context], registry, seed_calibration(spec), panel_size=3)
    elapsed = time.perf_counter() - started

    print(f"\n--- result ({elapsed:.2f}s; NFR1 budget is 60s) ---")
    print(f"  {report.verdict_counts}  review_rate={report.review_rate:.0%}")

    rows = report.to_dict()["ranked"]
    print(f"\n{'rank':>4}  {'spacer':30s} {'access':>7} {'dG/nt':>7} {'signal':>7} {'PFS':>4}")
    for row in rows[:8]:
        print(f"{row['rank']:>4}  {row['spacer']:30s} "
              f"{row['accessibility']:>7} {row['duplex_dg_per_nt']:>7} "
              f"{row['predicted_signal']:>7} {str(row['pfs_3prime']):>4}")

    signals = [r["predicted_signal"] for r in rows if r["predicted_signal"] is not None]
    if signals:
        print(f"\n  predicted signal spans {min(signals):.4f}-{max(signals):.4f} "
              f"across {len(set(signals))} distinct values")
    print("  NOTE: signal is RELATIVE. The kinetic constants shipped here are "
          "placeholders,\n        so absolute_signal() deliberately refuses until "
          "measured values are supplied.")

    print("\n--- panel ---")
    print(f"  {[g.guide_id for g in report.panel.guides]}")
    print(f"  {report.panel.guarantee}")

    # This script printed a ranking, per-guide collateral signal and a panel,
    # and not one word about what none of it establishes -- a grep of its
    # whole output for "critical", "warning" or "limitation" returned zero.
    # That is the finding the README records as fixed for `cas13guide demo`,
    # in the sibling nobody re-ran it against. The report carries them, so
    # this is a read.
    if report.limitations:
        print("\n--- what this run does not establish ---")
        for line in as_lines(report.limitations):
            print(f"  {line}")


if __name__ == "__main__":
    main()
