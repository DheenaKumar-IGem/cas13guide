"""Train a model from a screen, then design guides with it.

This closes the loop the README's "known limitation" describes: the reference
heuristic saturates, so every perfect-match guide lands on the same lower
bound and the ranking degenerates to tie-breaks. Swapping in a *fitted* model
changes nothing downstream -- it enters through the same ``ActivityModel``
seam -- and the ranking becomes informative.
"""

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
from cas13guide.pipeline.orchestrator import DesignContext, run_design
from cas13guide.services.calibration import CalibrationService
from cas13guide.services.registry import (
    HeuristicCas13Model,
    ModelRegistry,
    NullAccessibilityModel,
)
from cas13guide.services.training import compare_splitting, train_and_calibrate
from cas13guide.services.transcriptome import Locus, Transcriptome
from cas13guide.simulate import simulate_screen

UNIVERSES = frozenset(
    {Universe.CODING, Universe.NCRNA, Universe.PRE_MRNA, Universe.RRNA, Universe.MITO}
)


def build_world(target_sequence: str) -> tuple[Transcriptome, dict[str, str]]:
    import random

    rng = random.Random(4)

    def rna(n):
        return "".join(rng.choice("AUGC") for _ in range(n))

    t = Transcriptome("GRCh38", "v44")
    t.add(Locus("TGT1", target_sequence, Universe.CODING, gene="TGT", expression=120.0))
    t.add(Locus("LNC1", rna(400), Universe.NCRNA, gene="LINC01", expression=None))
    t.add(Locus("RRNA1", rna(500), Universe.RRNA, gene="RN18S", expression=9000.0))
    t.add(Locus("MT1", rna(300), Universe.MITO, gene="MT-CO1", expression=2500.0))
    t.add(Locus("PRE1", rna(600), Universe.PRE_MRNA, gene="TGT", expression=8.0))
    return t, {"TGT1": "t_tgt"}


# Discrimination is measured across every assessed guide, not just the ranked
# ones: the ranked list holds gate survivors, so a model that fails everything
# would show "zero distinct scores" for reasons unrelated to discrimination.


def main() -> None:
    # ---- 1. a screen, and an honest look at how to split it ----
    # 40 transcripts so the calibration split holds enough independent groups
    # for the conformal bounds to be presented as certified.
    examples, _ = simulate_screen(
        seed=21, step=4, signal_scale=10, transcript_effect_sd=0.8,
        noise_sd=0.2, n_transcripts=40, transcript_length=220,
    )
    print(f"simulated screen : {len(examples)} rows over "
          f"{len({e.group for e in examples})} transcripts\n")

    comparison = compare_splitting(examples, k=5, seed=1)
    print(comparison.summary(), "\n")

    # ---- 2. train -> calibrate -> evaluate, all grouped by transcript ----
    result = train_and_calibrate(examples, alpha=1.0, seed=5)
    print(f"split sizes      : {result.split_sizes}")
    print(f"held-out         : {result.benchmark.summary()}")
    print(f"coverage holds   : {result.benchmark.coverage_ok}")
    print("top features     :")
    for name, weight in result.model.top_features(6):
        print(f"                   {name:<20} {weight:+.3f}")
    print()

    # ---- 3. design with the trained model ----
    target = next(e for e in examples).target_site  # any real 22-mer context
    sequence = "".join(e.target_site for e in examples[:6])  # a 132nt transcript
    world, locus_to_target = build_world(sequence)

    spec = ResolvedDesignSpec(
        contract=TargetContract(
            task=Task.KNOCKDOWN,
            targets=(
                TargetSpec(
                    target_id="t_tgt", sequence=sequence,
                    target_class=TargetClass.REQUIRED, gene="TGT", expression=120.0,
                ),
            ),
        ),
        profile=EffectorAssayProfile(
            effector=Effector.RFX_CAS13D, assay=AssayType.CELL_CULTURE_KNOCKDOWN
        ),
        context=ContextScope(
            organism="Homo sapiens", genome_build="GRCh38",
            annotation_version="v44", cell_or_tissue="HEK293T",
            declared_universes=UNIVERSES,
        ),
    )

    def design(model, calibration, label):
        contexts = [DesignContext.auto("reference", world, UNIVERSES, locus_to_target)]
        report = run_design(
            spec, contexts, ModelRegistry(model, NullAccessibilityModel()),
            calibration, panel_size=3,
        )
        m = report.metric_summary
        print(f"{label:<18} PASS={report.verdict_counts['PASS']:>3} "
              f"REVIEW={report.verdict_counts['REVIEW']:>3} "
              f"FAIL={report.verdict_counts['FAIL']:>3} | "
              f"distinct coverage={m.distinct_coverage_values:>4}/{m.n_assessed} | "
              f"spread={m.coverage_spread:.4f}")
        return report

    print("--- ranking informativeness ---")

    # the heuristic, calibrated the same way, for a like-for-like comparison
    heuristic = HeuristicCas13Model()
    heuristic_cal = CalibrationService()
    for e in examples:
        heuristic_cal.add_observation(
            e.stratum(),
            heuristic.predict(e.spacer, e.duplex(), e.accessibility),
            e.observed_activity,
            group=e.group,
        )
    design(heuristic, heuristic_cal, "heuristic model")
    trained_report = design(result.model, result.calibration, "trained model")

    print("\n--- top ranked (trained) ---")
    for r in trained_report.ranked[:5]:
        m = r.assessment.metrics
        print(f"  #{r.rank} {r.assessment.guide_id} score={r.score:.3f} "
              f"cov={m.required_coverage:.3f} worst_off={m.worst_off_target:.3f}")

    # FR6.3. Worth showing here rather than in the built-in-model demos: the
    # success posterior is counted out of the calibration set, so it only says
    # anything once the calibration is a real screen rather than a stand-in.
    confidence = trained_report.panel_confidence
    if confidence and confidence.targets:
        print("\n--- panel detection probability (FR6.3) ---")
        for target in confidence.targets:
            members = ", ".join(f"{m.guide_id}={m.mean:.3f}" for m in target.members)
            print(f"  {target.target_id}: [{target.lower:.3f}, {target.upper:.3f}]"
                  f"  from {len(target.members)} member(s): {members}")
        print("  The floor is the best member; the ceiling assumes they fail")
        print("  independently, which guides on one transcript do not. No point")
        print("  estimate is offered without a measured common-cause fraction.")


if __name__ == "__main__":
    main()
