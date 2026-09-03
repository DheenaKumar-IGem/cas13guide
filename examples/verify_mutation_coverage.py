"""Check the claim that the suite is mutation-checked, rather than asserting it.

README.md says: "The suite is mutation-checked: the invariants the README claims
are each pinned by a test that fails when the behaviour is reverted." That is a
claim about the tests, and this package's standing rule is that a claim is worth
exactly what the procedure that reproduces it is worth -- which is why
``presets.verify_against_mirbase()`` exists rather than a hard-coded checksum.
This is the same thing for the test suite.

Each mutation below is a small **semantic** change to a load-bearing line: not a
syntax error, not a deleted function, but code that still runs and still looks
plausible. A mutation the suite catches is an invariant genuinely pinned. A
mutation that leaves the suite green is a hole -- the behaviour it changes is
not pinned by anything, and could be reverted by a refactor without a word.

Two things make the result trustworthy rather than flattering:

* **the mutations are subtle.** Disabling a whole check is caught by any
  competent suite and proves little. Half of these move a boundary by one step,
  swap a bound for its opposite, take the right operation on the wrong
  variable, or drop a finite-sample correction -- the changes a tired reviewer
  waves through;
* **a survivor is confirmed against the FULL suite.** Each mutation is screened
  against the handful of modules that should catch it, and anything that
  survives is re-run against all 1204 tests before being called a hole. So a
  "hole" is never an artefact of which modules were chosen. Two of the 38 are
  caught *only* by the full suite, which is why that second stage is not
  optional.

Run it::

    python examples/verify_mutation_coverage.py            # all mutations
    python examples/verify_mutation_coverage.py conformal  # matching subset

It copies the tree to a temporary directory and mutates the copy, so the
repository is never modified. Expect it to take a while: a surviving mutation
costs a full suite run.

Last measured: **38 mutations, 38 caught, 0 holes.**
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: ``(id, path relative to the repo, the code as written, the mutation, modules
#: that ought to catch it)``. The last field is a screening hint only -- a
#: survivor is always confirmed against the whole suite.
MUTATIONS: list[tuple[str, str, str, str, list[str]]] = [
    # ---- metrics/bounds.py: the asymmetric bound algebra ----
    ("bounds-benefit-reads-upper", "src/cas13guide/metrics/bounds.py",
     '    """Read a benefit at its lower confidence bound."""\n    return interval.lower',
     '    """Read a benefit at its lower confidence bound."""\n    return interval.upper',
     ["test_metrics.py", "test_gate.py", "test_decision_order.py"]),
    ("bounds-hazard-reads-point", "src/cas13guide/metrics/bounds.py",
     "        out *= iv.upper", "        out *= iv.point",
     ["test_metrics.py", "test_gate.py", "test_context_evidence.py"]),
    # ---- metrics/aggregate.py: the six named metrics ----
    ("aggregate-coverage-takes-max", "src/cas13guide/metrics/aggregate.py",
     "    return min(best.get(t.target_id, NO_EVIDENCE) for t in required)",
     "    return max(best.get(t.target_id, NO_EVIDENCE) for t in required)",
     ["test_metrics.py", "test_aggregate_stage.py", "test_gate.py"]),
    ("aggregate-margin-takes-min-off", "src/cas13guide/metrics/aggregate.py",
     "    worst_off = max(off_upper, default=0.0)",
     "    worst_off = min(off_upper, default=0.0)",
     ["test_mutation_holes.py", "test_metrics.py", "test_bugfix_regressions.py"]),
    ("aggregate-desired-drops-weight", "src/cas13guide/metrics/aggregate.py",
     "    return sum(t.weight * best.get(t.target_id, NO_EVIDENCE) for t in desired)",
     "    return sum(best.get(t.target_id, NO_EVIDENCE) for t in desired)",
     ["test_mutation_holes.py", "test_metrics.py", "test_aggregate_stage.py"]),
    # ---- pipeline/f_decide.py: the safety gate ----
    ("gate-coverage-boundary-flips", "src/cas13guide/pipeline/f_decide.py",
     "    if metrics.required_coverage < t.required_coverage_min:",
     "    if metrics.required_coverage < t.required_coverage_min - 0.05:",
     ["test_mutation_holes.py", "test_gate.py", "test_decision_order.py"]),
    ("gate-drops-forbidden-hit", "src/cas13guide/pipeline/f_decide.py",
     "    if forbidden:\n        fail.append(ReasonCode.CREDIBLE_FORBIDDEN_HIT)",
     "    if False and forbidden:\n        fail.append(ReasonCode.CREDIBLE_FORBIDDEN_HIT)",
     ["test_gate.py", "test_decision_order.py"]),
    ("gate-drops-universe-adequacy", "src/cas13guide/pipeline/f_decide.py",
     "    if universe_gaps:\n        fail.append(ReasonCode.UNIVERSE_INADEQUATE)",
     "    if False and universe_gaps:\n        fail.append(ReasonCode.UNIVERSE_INADEQUATE)",
     ["test_gate.py", "test_search.py"]),
    ("gate-hazard-boundary-flips", "src/cas13guide/pipeline/f_decide.py",
     "    if metrics.worst_off_target > t.worst_off_target_max:",
     "    if metrics.worst_off_target > t.worst_off_target_max * 2:",
     ["test_gate.py", "test_decision_order.py"]),
    ("gate-drops-qc-fail", "src/cas13guide/pipeline/f_decide.py",
     "    if not qc.passed:\n        fail.append(ReasonCode.DETERMINISTIC_INCOMPATIBLE)",
     "    if False and not qc.passed:\n        fail.append(ReasonCode.DETERMINISTIC_INCOMPATIBLE)",
     ["test_gate.py", "test_candidates_qc.py", "test_bugfix_regressions_round5.py"]),
    ("distinct-sites-uses-strict-gt", "src/cas13guide/pipeline/f_decide.py",
     "            abs(start - other) < min_spacing",
     "            abs(start - other) < min_spacing - 1",
     ["test_decision_order.py", "test_bugfix_regressions_round4.py",
      "test_bugfix_regressions_round5.py"]),
    # Followed its code: the rule moved into GuideAssessment.binding_sites when
    # the panel selector and the ranked report were made to read one definition
    # of a binding site. The anchor stayed behind in f_decide and the mutation
    # silently stopped running.
    ("distinct-sites-best-becomes-worst", "src/cas13guide/models/evidence.py",
     "            if target_id not in best or lower > best[target_id]:",
     "            if target_id not in best or lower < best[target_id]:",
     ["test_decision_order.py", "test_bugfix_regressions_round5.py",
      "test_panel_site_consistency.py", "test_bugfix_hunt_round13.py"]),
    # ---- pipeline/b_qc.py ----
    ("qc-drops-ambiguity-check", "src/cas13guide/pipeline/b_qc.py",
     '    if any(base not in "ACGU" for base in spacer):',
     '    if False and any(base not in "ACGU" for base in spacer):',
     ["test_candidates_qc.py", "test_bugfix_regressions_round5.py"]),
    ("qc-drops-pfs-incompatible", "src/cas13guide/pipeline/b_qc.py",
     "    if not params.pfs.check(candidate.five_prime_flank, candidate.three_prime_flank):",
     "    if False and not params.pfs.check(candidate.five_prime_flank, "
     "candidate.three_prime_flank):",
     ["test_candidates_qc.py", "test_effector_constants.py"]),
    # ---- pipeline/optimizer.py ----
    ("panel-drops-spacing", "src/cas13guide/pipeline/optimizer.py",
     "            if target_id in other.sites and abs(start - other.sites[target_id]) "
     "< min_spacing:",
     "            if False and target_id in other.sites and abs(start - "
     "other.sites[target_id]) < min_spacing:",
     ["test_optimizer.py", "test_panel_site_consistency.py", "test_panel_guarantee.py"]),
    ("panel-profile-reads-upper", "src/cas13guide/pipeline/optimizer.py",
     "            if benefit_lower(e.activity) > coverage.get(tid, 0.0):",
     "            if e.activity.upper > coverage.get(tid, 0.0):",
     ["test_mutation_holes.py", "test_optimizer.py", "test_panel_guarantee.py", "test_bugfix_regressions_round5.py"]),
    ("panel-drops-diversity", "src/cas13guide/pipeline/optimizer.py",
     "        if len(candidate.off_loci & other.off_loci) > max_shared_off:",
     "        if False and len(candidate.off_loci & other.off_loci) > max_shared_off:",
     ["test_optimizer.py", "test_panel_guarantee.py"]),
    # ---- pipeline/a_candidates.py: the budget split ----
    ("budget-share-off-by-one", "src/cas13guide/pipeline/a_candidates.py",
     "        allowance = max(1, share + (1 if position < remainder else 0))",
     "        allowance = max(1, share)",
     ["test_mutation_holes.py", "test_candidates_qc.py", "test_bugfix_regressions_round4.py"]),
    ("budget-keeps-last-window-off", "src/cas13guide/pipeline/a_candidates.py",
     "                    round(i * (offered - 1) / (allowance - 1))",
     "                    round(i * offered / (allowance - 1))",
     ["test_candidates_qc.py", "test_bugfix_regressions_round4.py"]),
    # ---- pipeline/d_pair_evidence.py ----
    ("domain-violation-still-narrows", "src/cas13guide/pipeline/d_pair_evidence.py",
     '        activity = Interval.unknown(stratum=f"{stratum}->model-out-of-domain")',
     "        activity = activity",
     ["test_gate.py", "test_bugfix_regressions.py", "test_presets.py"]),
    # ---- services/calibration.py: the conformal quantile itself ----
    ("conformal-rank-off-by-one", "src/cas13guide/services/calibration.py",
     "        rank = math.ceil((n + 1) * (1.0 - alpha))",
     "        rank = math.ceil(n * (1.0 - alpha))",
     ["test_calibration.py", "test_benchmark.py"]),
    ("conformal-drops-finite-sample-check", "src/cas13guide/services/calibration.py",
     "        if rank > n:\n            return scores[-1], False",
     "        if rank > n:\n            return scores[-1], True",
     ["test_calibration.py"]),
    ("conformal-group-adequacy-off-by-one", "src/cas13guide/services/calibration.py",
     "        return stratum.n_groups >= self.min_stratum_groups",
     "        return stratum.n_groups >= self.min_stratum_groups - 1",
     ["test_mutation_holes.py", "test_calibration.py"]),
    ("coverage-slack-doubled", "src/cas13guide/services/calibration.py",
     "COVERAGE_SLACK = 0.05", "COVERAGE_SLACK = 0.10",
     ["test_mutation_holes.py", "test_calibration.py", "test_acceptance.py", "test_bugfix_regressions_round8.py"]),
    # ---- services/indexes.py: the search bound algebra ----
    ("union-dedup-key-drops-start", "src/cas13guide/services/indexes.py",
     "                key = (hit.locus_id, hit.start)", "                key = (hit.locus_id,)",
     ["test_bugfix_regressions_round2.py", "test_search.py"]),
    ("weakest-bound-becomes-latest", "src/cas13guide/services/indexes.py",
     "    if candidate.exhaustive != current.exhaustive:\n        return not candidate.exhaustive",
     "    if candidate.exhaustive != current.exhaustive:\n        return candidate.exhaustive",
     ["test_search.py", "test_gate.py", "test_index_selection.py"]),
    # ---- services/panel_confidence.py (FR6.3) ----
    ("panel-floor-becomes-mean", "src/cas13guide/services/panel_confidence.py",
     "                    lower=max(means),", "                    lower=sum(means) / len(means),",
     ["test_mutation_holes.py", "test_panel_confidence.py", "test_bugfix_regressions.py"]),
    ("panel-success-threshold-halved", "src/cas13guide/services/panel_confidence.py",
     "        self.threshold = float(spec.thresholds.required_coverage_min)",
     "        self.threshold = float(spec.thresholds.required_coverage_min) / 2",
     ["test_panel_confidence.py"]),
    ("beta-quantile-loses-precision", "src/cas13guide/services/panel_confidence.py",
     "    for _ in range(100):", "    for _ in range(8):",
     ["test_panel_confidence.py"]),
    # ---- services/kinetics.py: the stability guard ----
    ("kinetics-stability-target-relaxed", "src/cas13guide/services/kinetics.py",
     "STABILITY_TARGET = 2.0", "STABILITY_TARGET = 200.0",
     ["test_kinetics_mass_balance.py", "test_bugfix_regressions.py"]),
    # ---- services/conservation.py ----
    ("conservation-worst-becomes-best", "src/cas13guide/services/conservation.py",
     "        return max(self.mismatches_by_variant.values(), default=0)",
     "        return min(self.mismatches_by_variant.values(), default=0)",
     ["test_conservation.py"]),
    # ---- util/duplex.py: the wobble rule ----
    # Drops G-U (written here as the A->G substitution) and keeps C-U, rather
    # than renaming the constant. The rename was consistent inside this module,
    # so it changed nothing here and only ever failed because two other modules
    # import the name: it was an ImportError wearing a semantic mutation's
    # clothes. This removes one real pair and leaves valid, running code.
    ("wobble-rule-dropped", "src/cas13guide/util/duplex.py",
     'frozenset({("C", "U"), ("A", "G")})', 'frozenset({("C", "U")})',
     ["test_duplex.py", "test_search.py"]),
    # ---- round thirteen's fixes, which were the newest code here and so
    # ---- the least settled. None of the mutations above reach any of them,
    # ---- and four of these six restore the defect exactly as it shipped.
    ("expression-ignorance-is-not-a-bound", "src/cas13guide/services/context.py",
     'return Interval.unknown(lo=0.0, hi=1.0, stratum="expression/unmeasured")',
     'return Interval.unknown(lo=0.0, hi=abundance_factor(1000.0), '
     'stratum="expression/unmeasured")',
     ["test_bugfix_hunt_round13.py"]),
    ("pfs-unread-base-is-disfavoured", "src/cas13guide/contract/effectors.py",
     'UNREADABLE_FLANK: frozenset[str] = frozenset({"N"})',
     "UNREADABLE_FLANK: frozenset[str] = frozenset()",
     ["test_bugfix_hunt_round13.py"]),
    # An off-by-one at the far end of the second strand: the helix stops one
    # base short whenever it would reach y[0]. In bounds, still runs, and the
    # kind of boundary a tired reviewer waves through.
    #
    # It replaced a first attempt that swapped `y[j - length]` for
    # `y[j + length]` to make the scan pair *parallel*. That reads like a
    # sharper mutation and is a worse one: the loop only bounds-checks the
    # descending index, so it raises IndexError instead of returning a
    # plausible wrong number. A mutation that crashes tests that the suite
    # notices a crash, which is the same weakness this file just removed from
    # `wobble-rule-dropped`.
    ("cross-complement-stops-one-base-short", "src/cas13guide/pipeline/b_qc.py",
     "while (i + length) < len(x) and (j - length) >= 0:",
     "while (i + length) < len(x) and (j - length) > 0:",
     ["test_bugfix_hunt_round13.py"]),
    ("dr-threshold-left-at-its-watson-crick-value", "src/cas13guide/pipeline/b_qc.py",
     "MAX_DR_STEM = 9", "MAX_DR_STEM = 6",
     ["test_bugfix_hunt_round13.py"]),
    ("binding-sites-forgets-desired-targets", "src/cas13guide/models/evidence.py",
     "if not evidence.is_intended or not target_id:",
     "if evidence.target_class is not TargetClass.REQUIRED or not target_id:",
     ["test_bugfix_hunt_round13.py", "test_panel_site_consistency.py"]),
    # Subtler than hashing the wrong fields: the digest stays a content hash and
    # keeps every byte that matters, but two loads of one corpus in different
    # orders stop agreeing -- so the identity changes for no reason anyone can
    # see, which is worse than no identity at all.
    ("world-digest-depends-on-load-order",
     "src/cas13guide/services/transcriptome.py",
     "for locus in sorted(self.loci.values(), key=lambda lo: lo.locus_id)",
     "for locus in self.loci.values()",
     ["test_bugfix_hunt_round13.py"]),
]


def _run(root: Path, modules: list[str] | None) -> tuple[bool, str]:
    command = [sys.executable, "-m", "pytest", "-q", "-x", "--no-header",
               "-p", "no:cacheprovider"]
    command += [f"tests/{name}" for name in modules] if modules else ["tests"]
    done = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=3600)
    return done.returncode == 0, (done.stdout.strip().splitlines() or [""])[-1]


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else None
    holes: list[str] = []
    broken: list[str] = []
    checked = 0

    with tempfile.TemporaryDirectory(prefix="cas13guide-mutation-") as scratch:
        root = Path(scratch) / "tree"
        root.mkdir()
        # Whatever the suite reads has to be here, not just what the mutations
        # edit. `tests/test_documents_agree.py` compares the README and the
        # landing page against the package, so a copy holding only src and
        # tests could not collect it -- the sweep aborted on its own baseline
        # check until this list grew. Mutations only ever touch `src`, so the
        # rest is inert ballast that keeps the suite runnable.
        for item in ("src", "tests", "examples", "docs", "pyproject.toml",
                     "README.md"):
            source = REPO / item
            if not source.exists():
                continue
            if source.is_dir():
                shutil.copytree(source, root / item)
            else:
                shutil.copy2(source, root / item)

        # A mutation is "caught" when the suite goes from green to red. Nothing
        # checked that it started green, and starting red makes every mutation
        # look caught -- including the ones that are holes. This instrument
        # found that out by being run that way: two document-consistency tests
        # were failing at the time, and the full-suite stage would have called
        # any survivor "caught" by tests that were already red.
        #
        # It costs one suite run to rule out, and a whole sweep is worthless
        # without it.
        # Unconditional, including for a filtered subset run: a subset started
        # from red is exactly as misleading as a whole sweep started from red,
        # and the cost is one suite run against an instrument whose only value
        # is being trustworthy.
        print("checking the unmutated tree first ...")
        green, tail = _run(root, None)
        if not green:
            print()
            print("[ABORT] the tree does not pass its own suite unmutated.")
            print("Every mutation would be reported caught, by tests that")
            print("were already failing. Fix the baseline, then re-run.")
            print(f"  {tail}")
            return 2
        print("baseline green.")
        print()

        for name, relative, original_code, mutated_code, modules in MUTATIONS:
            if wanted and wanted not in name:
                continue
            path = root / relative
            before = path.read_text(encoding="utf-8")
            # An anchor must match exactly once, and failing that is a failure
            # rather than a skip. Both halves were learned the hard way.
            #
            # Present-but-not-unique: `wobble-rule-dropped` anchored on the bare
            # name `WOBBLE_SUBSTITUTIONS`, which occurs twice in duplex.py --
            # the definition and its one use. `str.replace` renamed both, which
            # is a *consistent rename*, so within that module the mutation did
            # nothing at all. It only ever reported "caught" because two other
            # modules import that name and the suite died of an ImportError. It
            # had been testing that a broken import breaks the suite, under a
            # README line claiming it mutates the wobble rule.
            #
            # Absent: `distinct-sites-best-becomes-worst` lost its anchor when
            # the site rule moved into `GuideAssessment.binding_sites`. It was
            # skipped in silence, `checked` never counted it, and the run still
            # printed "Every mutation was caught" and exited 0. An instrument
            # whose job is finding unpinned behaviour must not quietly stop
            # testing something and call that success.
            occurrences = before.count(original_code)
            if occurrences != 1:
                reason = ("anchor no longer present -- has the code moved?"
                          if occurrences == 0
                          else f"anchor matches {occurrences} places, so the "
                               "mutation is ambiguous")
                print(f"[BROKEN] {name}: {reason}")
                broken.append(f"{name} ({reason})")
                continue
            checked += 1
            path.write_text(before.replace(original_code, mutated_code), encoding="utf-8")
            try:
                screened, _ = _run(root, modules)
                if not screened:
                    print(f"[caught] {name}")
                    continue
                # Screened green: only the full suite can call this a hole.
                whole, tail = _run(root, None)
                if whole:
                    holes.append(name)
                    print(f"[HOLE  ] {name} -- full suite still green: {tail}")
                else:
                    print(f"[caught] {name} (by the full suite, not the screening modules)")
            finally:
                path.write_text(before, encoding="utf-8")

    print()
    print(f"{checked} of {len(MUTATIONS)} mutation(s) applied, {len(holes)} survived.")
    if broken:
        print()
        print("Anchors that no longer identify one place -- these mutations did")
        print("NOT run, so nothing below is evidence about what they guard:")
        for item in broken:
            print(f"  {item}")
    if holes:
        print("Unpinned behaviour -- each of these can be reverted without a test noticing:")
        for name in holes:
            print(f"  {name}")
        return 1
    if broken:
        return 1
    print("Every mutation was caught: the invariants above are pinned, not merely asserted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
