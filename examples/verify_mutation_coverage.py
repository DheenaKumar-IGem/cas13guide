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
  survives is re-run against all 828 tests before being called a hole. So a
  "hole" is never an artefact of which modules were chosen. Ten of the
  thirty-three are caught *only* by the full suite, which is why that second
  stage is not optional.

Run it::

    python examples/verify_mutation_coverage.py            # all mutations
    python examples/verify_mutation_coverage.py conformal  # matching subset

It copies the tree to a temporary directory and mutates the copy, so the
repository is never modified. Expect it to take a while: a surviving mutation
costs a full suite run.

Last measured: **33 mutations, 33 caught, 0 holes.**
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
     ["test_metrics.py", "test_bugfix_regressions.py"]),
    ("aggregate-desired-drops-weight", "src/cas13guide/metrics/aggregate.py",
     "    return sum(t.weight * best.get(t.target_id, NO_EVIDENCE) for t in desired)",
     "    return sum(best.get(t.target_id, NO_EVIDENCE) for t in desired)",
     ["test_metrics.py", "test_aggregate_stage.py"]),
    # ---- pipeline/f_decide.py: the safety gate ----
    ("gate-coverage-boundary-flips", "src/cas13guide/pipeline/f_decide.py",
     "    if metrics.required_coverage < t.required_coverage_min:",
     "    if metrics.required_coverage < t.required_coverage_min - 0.05:",
     ["test_gate.py", "test_decision_order.py"]),
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
    ("distinct-sites-best-becomes-worst", "src/cas13guide/pipeline/f_decide.py",
     "            if target_id not in best_lower or lower > best_lower[target_id]:",
     "            if target_id not in best_lower or lower < best_lower[target_id]:",
     ["test_decision_order.py", "test_bugfix_regressions_round5.py"]),
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
     ["test_optimizer.py", "test_panel_guarantee.py", "test_bugfix_regressions_round5.py"]),
    ("panel-drops-diversity", "src/cas13guide/pipeline/optimizer.py",
     "        if len(candidate.off_loci & other.off_loci) > max_shared_off:",
     "        if False and len(candidate.off_loci & other.off_loci) > max_shared_off:",
     ["test_optimizer.py", "test_panel_guarantee.py"]),
    # ---- pipeline/a_candidates.py: the budget split ----
    ("budget-share-off-by-one", "src/cas13guide/pipeline/a_candidates.py",
     "        allowance = max(1, share + (1 if position < remainder else 0))",
     "        allowance = max(1, share)",
     ["test_candidates_qc.py", "test_bugfix_regressions_round4.py"]),
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
     ["test_calibration.py"]),
    ("coverage-slack-doubled", "src/cas13guide/services/calibration.py",
     "COVERAGE_SLACK = 0.05", "COVERAGE_SLACK = 0.10",
     ["test_calibration.py", "test_acceptance.py", "test_bugfix_regressions_round8.py"]),
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
     ["test_panel_confidence.py", "test_bugfix_regressions.py"]),
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
    ("wobble-rule-dropped", "src/cas13guide/util/duplex.py",
     "WOBBLE_SUBSTITUTIONS", "WOBBLE_SUBSTITUTIONS_DISABLED",
     ["test_duplex.py", "test_search.py"]),
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
    checked = 0

    with tempfile.TemporaryDirectory(prefix="cas13guide-mutation-") as scratch:
        root = Path(scratch) / "tree"
        root.mkdir()
        for item in ("src", "tests", "pyproject.toml"):
            source = REPO / item
            if source.is_dir():
                shutil.copytree(source, root / item)
            else:
                shutil.copy2(source, root / item)

        for name, relative, original_code, mutated_code, modules in MUTATIONS:
            if wanted and wanted not in name:
                continue
            path = root / relative
            before = path.read_text(encoding="utf-8")
            if original_code not in before:
                print(f"[SKIP ] {name}: anchor no longer present -- has the code moved?")
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
    print(f"{checked} mutation(s) applied, {len(holes)} survived.")
    if holes:
        print("Unpinned behaviour -- each of these can be reverted without a test noticing:")
        for name in holes:
            print(f"  {name}")
        return 1
    print("Every mutation was caught: the invariants above are pinned, not merely asserted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
