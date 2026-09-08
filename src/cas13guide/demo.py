"""The bundled end-to-end example, as a module rather than a script.

``cas13guide demo`` used to reach for ``examples/demo_knockdown.py`` by walking
two directories up from ``cli.py`` and putting the result on ``sys.path``. That
works in a source checkout and nowhere else: ``examples/`` is not part of the
distribution, so for anyone who ``pip install``ed the package the subcommand --
advertised in ``--help`` and in ``cli``'s own docstring -- died with
``ModuleNotFoundError: No module named 'demo_knockdown'`` and exit 1. Found by
building a wheel, installing it into a clean interpreter and running the
documented commands from outside the source tree, which nothing had done.

So the demo lives in the package, ships in the wheel, and works wherever the
package does. ``examples/demo_knockdown.py`` stays as the script it always was
and calls straight into this.

It builds a small but structurally honest world -- an intended transcript, a
paralog sharing a near-cognate site, an unmeasured lncRNA, rRNA and mito decoys,
a pre-mRNA, and a variant haplotype context -- then runs the full pipeline and
prints the report.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .contract.enums import AssayType, Effector, Task, TargetClass, Universe
from .contract.spec import (
    ContextScope,
    EffectorAssayProfile,
    ResolvedDesignSpec,
    TargetContract,
    TargetSpec,
)
from .pipeline.g_output import DesignReport
from .pipeline.orchestrator import DesignContext, run_design
from .services.calibration import CalibrationService
from .services.registry import (
    HeuristicCas13Model,
    ModelRegistry,
    NullAccessibilityModel,
)
from .services.transcriptome import Locus, Transcriptome

RNG = random.Random(11)


def random_rna(n: int) -> str:
    return "".join(RNG.choice("AUGC") for _ in range(n))


TARGET = (
    "AUGGCACUGAAGUCCAUCGACUGGAACUUCCGAGUGCAUGGCAUUACCGAGUCCUAUGGCAAGUUCGAC"
    "CUGAUCGAGAAGUGCCUGGAAUCCAUGGGCUACCGAGAUCUGGCCAAGUUCGGCAUCUAA"
)


def build_world() -> tuple[Transcriptome, Transcriptome, dict[str, str]]:
    """Reference transcriptome, a variant-haplotype context, and locus->target."""
    reference = Transcriptome("GRCh38", "v44")
    reference.add(Locus("TGT1", TARGET, Universe.CODING, gene="TGT", expression=120.0))

    # A paralog sharing a near-cognate site (3 mismatches from one target site).
    site = TARGET[30:52]
    paralog_site = list(site)
    for pos in (4, 11, 18):
        paralog_site[pos] = RNG.choice([b for b in "AUGC" if b != paralog_site[pos]])
    paralog = random_rna(120) + "".join(paralog_site) + random_rna(120)
    reference.add(Locus("PAR1", paralog, Universe.CODING, gene="TGTL", expression=45.0))

    # An unmeasured lncRNA: "not detected" is not "absent".
    reference.add(Locus("LNC1", random_rna(400), Universe.NCRNA, gene="LINC01", expression=None))
    # Universes the assay is exposed to and that must therefore be searched.
    reference.add(Locus("RRNA1", random_rna(600), Universe.RRNA, gene="RN18S", expression=9000.0))
    reference.add(Locus("MT1", random_rna(300), Universe.MITO, gene="MT-CO1", expression=2500.0))
    reference.add(Locus("PRE1", random_rna(700), Universe.PRE_MRNA, gene="TGT", expression=8.0))

    # A variant haplotype: same world, one SNV inside the target transcript.
    variant = Transcriptome("GRCh38", "v44+hap2")
    for locus in reference.loci.values():
        variant.add(locus)
    mutated = list(TARGET)
    mutated[60] = "G" if mutated[60] != "G" else "A"
    variant.add(
        Locus("TGT1", "".join(mutated), Universe.CODING, gene="TGT", expression=120.0)
    )
    return reference, variant, {"TGT1": "t_tgt"}


#: What generates the stand-in below. These reach the ledger and the
#: limitation through the builder, which records them itself rather than
#: leaving a caller to keep a parallel dict in step.
_SYNTHETIC = {"seed": 11, "n_per_stratum": 120, "noise_sd": 0.13}


def seed_calibration(spec: ResolvedDesignSpec) -> CalibrationService:
    """Stand in for held-out tiling-screen data.

    A thin caller now. This used to be its own copy of the design path's
    seeding, and it drifted from it three times -- most recently by omitting
    the group label, which left every stratum with zero independent units, so
    every interval came back out-of-domain and every guide landed R3
    `calibration_unsupported`. The demo could not reach a PASS, its PASS-only
    panel was necessarily empty, and the README quoted that shortfall as the
    demo's transcript.
    """
    from .services.workflow import build_synthetic_calibration

    return build_synthetic_calibration(
        spec.profile.effector.value, spec.profile.assay.value, **_SYNTHETIC)


def main() -> DesignReport:
    reference, variant, locus_to_target = build_world()

    contract = TargetContract(
        task=Task.KNOCKDOWN,
        targets=(
            TargetSpec(
                target_id="t_tgt",
                sequence=TARGET,
                target_class=TargetClass.REQUIRED,
                gene="TGT",
                transcript="TGT1",
                expression=120.0,
            ),
        ),
    )
    profile = EffectorAssayProfile(
        effector=Effector.RFX_CAS13D, assay=AssayType.CELL_CULTURE_KNOCKDOWN
    )
    context = ContextScope(
        organism="Homo sapiens",
        genome_build="GRCh38",
        annotation_version="v44",
        cell_or_tissue="HEK293T",
        # Declaring every universe the assay is exposed to; drop one and the
        # universe-adequacy gate FAILs the whole run by design.
        declared_universes=frozenset(
            {Universe.CODING, Universe.NCRNA, Universe.PRE_MRNA, Universe.RRNA, Universe.MITO}
        ),
    )
    spec = ResolvedDesignSpec(contract=contract, profile=profile, context=context)

    universes = context.declared_universes
    # Let the system pick the index. At this toy scale it will choose the
    # linear scanner; on a real transcriptome it switches to the FM-index.
    contexts = [
        DesignContext.auto("reference", reference, universes, locus_to_target),
        DesignContext.auto("hap2", variant, universes, locus_to_target),
    ]
    registry = ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel())
    calibration = seed_calibration(spec)

    print(f"spec fingerprint : {spec.fingerprint()}")
    print(f"spacer length    : {spec.spacer_length}")
    print(f"index            : {contexts[0].index.version}")
    print(f"                   {contexts[0].index_rationale}\n")

    report = run_design(
        spec, contexts, registry, calibration, panel_size=3,
        on_event=lambda m: print("  " + m),
    )

    print("\n--- verdicts ---")
    print(report.verdict_counts, f"review_rate={report.review_rate:.2%}")

    # `report.ranked` is every gate survivor, PASS and REVIEW alike. The header
    # used to say "PASS only" two lines under a line reading "0 PASS of 108" --
    # asserting, for every row it listed, a property the run had just reported
    # as false for all of them. The CLI already carries the triage note; the
    # demo is the first thing a new user runs, so it needs it more.
    print("\n--- top ranked (gate survivors: PASS and REVIEW) ---")
    if not report.panel.guides:
        print(
            "  ! no panel: guides ranked but none PASSed the gate, and the "
            "panel is PASS-only. The rows below are candidates to triage, "
            "not a recommendation."
        )
    for r in report.ranked[:5]:
        m = r.assessment.metrics
        print(
            f"  #{r.rank} {r.assessment.guide_id} {r.assessment.spacer} "
            f"score={r.score:.2f} cov={m.required_coverage:.3f} "
            f"worst_off={m.worst_off_target:.3f} burden={m.cumulative_burden:.3f} "
            f"robust={m.robustness:.3f}"
        )

    print("\n--- panel ---")
    print(f"  guides   : {[g.guide_id for g in report.panel.guides]}")
    print(f"  coverage : { {k: round(v,4) for k,v in report.panel.coverage.items()} }")
    print(f"  guarantee: {report.panel.guarantee}")

    print("\n--- controls ---")
    for c in report.controls:
        print(f"  {c.guide_id} {c.spacer} gc={c.gc:.2f} verified_no_hits={c.verified_no_hits}")

    print("\n--- top off-targets ---")
    for row in report.off_target_rows[:5]:
        print(
            f"  {row['guide_id']} -> {row['locus']} ({row['gene']},{row['biotype']}) "
            f"{row['mismatches']}mm hazard={row['hazard']:.4f} "
            f"expr_unknown={row['expression_unknown']}"
        )

    print("\n--- validation plan ---")
    for line in report.validation_plan:
        print(f"  - {line}")

    if report.shortfalls:
        print("\n--- shortfalls ---")
        for s in report.shortfalls:
            print(f"  ! {s}")

    # The demo printed a panel, a ranking and a validation plan, and not one
    # word about what none of it establishes -- no mention that the calibration
    # is synthetic and every PASS therefore exercisable rather than earned,
    # that the activity model is a saturating heuristic, or that collateral is
    # not predicted at all. `design` prints all four. This is the first command
    # anybody runs, and a tool whose argument is that it says what it could not
    # establish cannot have its showcase say nothing.
    #
    # Read off the report rather than reassembled here. This call used to be
    # made twice, once here and once in the design path, and the two had
    # drifted: only that one passed `self_identical_loci`, so this surface
    # could not raise the off-target-may-be-the-target CRITICAL at all.
    from .services.limitations import as_lines

    if report.limitations:
        print("\n--- what this run does not establish ---")
        for line in as_lines(report.limitations):
            print(f"  {line}")

    data = report.to_dict()

    destination = Path("report.json")
    destination.write_text(json.dumps(data, indent=2, default=str))
    # Resolved, because "wrote report.json" does not say *where* when the
    # command can now be run from any directory on any machine.
    print(f"\nwrote {destination.resolve()}")
    return report


if __name__ == "__main__":
    main()
