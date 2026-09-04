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
  survives is re-run against all 1241 tests before being called a hole. So a
  "hole" is never an artefact of which modules were chosen. None of the 79 are
  caught *only* by the full suite today -- each hint names the module measured
  to catch it -- but the stage stays, because that is what makes a survivor a
  statement about the suite rather than about a chosen subset.

Run it::

    python examples/verify_mutation_coverage.py            # all mutations
    python examples/verify_mutation_coverage.py conformal  # matching subset

It copies the tree to a temporary directory and mutates the copy, so the
repository is never modified. About five minutes as it stands; a surviving
mutation costs a whole extra suite run, so a hole is expensive and a hint that
names the wrong module is expensive too.

Last measured: **79 mutations, 79 caught, 0 holes.**
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: ``(id, path relative to the repo, the code as written, the mutation, modules
#: that ought to catch it)``. The last field is a screening hint only -- a
#: survivor is always confirmed against the whole suite.
#:
#: The first module listed is the one *measured* to catch it, where that was
#: worth finding out; the rest are the ones a reader would expect to. They are
#: not always the same, and the difference is worth seeing. A halved panel
#: success threshold is caught by ``test_bugfix_hunt_round11.py`` and not by
#: ``test_panel_confidence.py``; the off-target dedup key is caught by
#: ``test_installed_package.py``, which runs the example scripts end to end,
#: and not by ``test_search.py``. Where a behaviour is pinned only by a module
#: that looks unrelated to it, pruning that module would quietly unpin it.
#:
#: Naming the real catcher also costs the sweep six fewer whole-suite runs,
#: which is most of its wall clock -- an instrument nobody runs documents drift
#: instead of preventing it.
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
     ["test_installed_package.py", "test_bugfix_regressions_round2.py", "test_search.py"]),
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
     ["test_bugfix_hunt_round11.py", "test_panel_confidence.py"]),
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
    # ---- Probes into surface no mutation had ever touched. Eight of the ten
    # ---- mutations whose verdicts were unverifiable turned out to be holes,
    # ---- so the modules nothing has ever mutated are where the next ones are.
    #
    # services/limitations.py is the product's central promise -- the guide
    # designer that tells you when it does not know -- and had never been
    # mutated at all. Demotion rather than deletion: a dropped limitation is
    # caught by anything that greps for the text, while a CRITICAL quietly
    # rendered as a warning still appears on every surface in the same words.
    # Only the severity mark and the ordering change, which is exactly how a
    # caveat stops being read.
    ("critical-calibration-warning-demoted",
     "src/cas13guide/services/limitations.py",
     'Severity.CRITICAL,\n                "Conformal bounds were calibrated',
     'Severity.WARNING,\n                "Conformal bounds were calibrated',
     ["test_mutation_holes.py", "test_caveats_reach_every_surface.py", "test_limitations.py",
      "test_bugfix_regressions.py"]),
    ("collateral-warning-goes-to-the-wrong-runs",
     "src/cas13guide/services/limitations.py",
     "    if spec.profile.collateral_is_hazard:",
     "    if not spec.profile.collateral_is_hazard:",
     ["test_caveats_reach_every_surface.py", "test_limitations.py"]),
    # services/fasta.py: the header a CSV carries out of the session.
    ("export-header-drops-the-world", "src/cas13guide/services/fasta.py",
     'if kind in ("model", "calibration", "search", "transcriptome"):',
     'if kind in ("model", "calibration", "search"):',
     ["test_bugfix_hunt_round13.py"]),
    # pipeline/g_output.py: the exported tables.
    ("rejection-rows-group-the-survivors", "src/cas13guide/pipeline/g_output.py",
     "        if a.gate.verdict is not Verdict.FAIL:",
     "        if a.gate.verdict is Verdict.FAIL:",
     ["test_rejection_reporting.py", "test_bugfix_regressions.py"]),
    # services/training.py: the grouped split, which is the whole reason this
    # module claims its cross-validation means anything. An off-by-one in the
    # train complement leaves the held-out groups *in* the training set: the
    # fold still has a train and a test list of plausible sizes, every number
    # it reports goes up, and the estimate is of a question nobody asked.
    ("group-kfold-trains-on-its-own-test-set",
     "src/cas13guide/services/training.py",
     "        train = [i for i, g in enumerate(groups) if assignment[g] != fold]",
     "        train = [i for i, g in enumerate(groups) if assignment[g] != fold + 1]",
     ["test_training.py", "test_trained_end_to_end.py"]),
    # services/report_html.py: the severity mark on the served report. The
    # limitation text is unchanged and still first on the page; only the rule
    # that colours it critical stops matching, so the caveat renders as an
    # ordinary note. Same shape as the demotion above, one surface further out.
    ("html-report-stops-marking-critical-notes",
     "src/cas13guide/services/report_html.py",
     'if n.upper().startswith("CRITICAL")',
     'if n.upper().startswith("WARNING")',
     ["test_mutation_holes.py", "test_caveats_reach_every_surface.py", "test_report_html.py"]),
    # services/ingest.py: the duplicate-column refusal round twelve added,
    # because csv.DictReader keeps the last column of a repeated name and a
    # stray second `activity` inverts every training label in silence.
    ("ingest-allows-a-duplicate-column", "src/cas13guide/services/ingest.py",
     "if header.count(name) > 1}",
     "if header.count(name) > 2}",
     ["test_bugfix_hunt_round12.py", "test_ingest.py"]),
    # services/workflow.py: the duplicate-name guard on --background. Comparing
    # the id against the wrong collection is always False, so the guard stops
    # dropping repeats -- and a world keys loci by name, so the *last* copy
    # silently overwrites the first. The note still says only the first was
    # searched. Round four fixed this; nothing has mutated the module since.
    ("background-dedup-tests-the-wrong-collection",
     "src/cas13guide/services/workflow.py",
     "        if locus.locus_id in seen_background:",
     "        if locus.locus_id in accepted_background:",
     ["test_bugfix_regressions_round5.py", "test_bugfix_regressions_round4.py", "test_the_input_doors_agree.py"]),
    # services/fasta.py: where a record's name comes from. Splitting the header
    # on the wrong character renames every locus containing that character --
    # ">ENST00000_1" becomes "ENST00000" -- so provenance, off-target rows and
    # the target map all name something that was never in the file. Nothing
    # crashes; the identifiers are simply not the ones the user supplied.
    ("fasta-splits-the-header-on-the-wrong-character",
     "src/cas13guide/services/fasta.py",
     'name, _, description = header.partition(" ")',
     'name, _, description = header.partition("_")',
     ["test_the_input_doors_agree.py", "test_bugfix_regressions.py"]),
    # pipeline/orchestrator.py: which context is *the* reference. Coordinates,
    # the off-target explorer and the ranked report all read the reference
    # context; the metrics are aggregated across all of them. Picking a variant
    # instead still produces a full run with plausible numbers, and every
    # position it reports is a position on a haplotype the user did not ask
    # about. Only reachable with --variants, which is exactly where the two
    # earlier context defects lived.
    ("orchestrator-picks-a-variant-as-the-reference",
     "src/cas13guide/pipeline/orchestrator.py",
     'reference = next((c for c in contexts if c.name == "reference"), contexts[0])',
     'reference = next((c for c in contexts if c.name != "reference"), contexts[0])',
     ["test_mutation_holes.py", "test_context_evidence.py", "test_bugfix_regressions_round4.py"]),
    # contract/spec.py: the gate's own floor. Every threshold here is a default
    # a caller may override, so nothing about the *value* is sacred -- but the
    # shipped default is what almost every run is judged against, and lowering
    # it by a fifth passes guides the gate is written to stop.
    ("spec-lowers-the-shipped-coverage-floor",
     "src/cas13guide/contract/spec.py",
     "    required_coverage_min: Fraction = 0.50",
     "    required_coverage_min: Fraction = 0.40",
     ["test_gate.py", "test_mutation_holes.py", "test_decision_order.py"]),
    # services/blast.py: the honesty flag on a seeded search. BLAST cannot say
    # it looked everywhere -- a site whose differences leave no exact word
    # intact is never found -- and the gate turns exhaustive=False into an R2
    # REVIEW. Claiming True lets guides PASS on a search that did not cover
    # what a PASS implies it covered, and nothing else about the run changes.
    ("blast-claims-it-searched-everywhere",
     "src/cas13guide/services/blast.py",
     "            exhaustive=False,",
     "            exhaustive=True,",
     ["test_blast.py", "test_search.py", "test_gate.py"]),
    # cli.py: "no guide cleared the gate" against "a display filter emptied the
    # list". `rows` has been through the distinct-site filter and --top; only
    # `all_rows` says what the run actually produced. Reading the display list
    # prints something false about the run, next to exports carrying every one
    # of those guides -- which is the defect the comment beside it records.
    ("cli-reports-the-display-filter-as-an-empty-run",
     "src/cas13guide/cli.py",
     "    elif all_rows:", "    elif rows:",
     ["test_mutation_holes.py", "test_subcommands_end_to_end.py", "test_module_entry_point.py"]),
    # The exit-code contracts, which only a subprocess can observe -- and so
    # were unreachable by this sweep until it started putting the mutated copy
    # on PYTHONPATH for child processes. A status code is the one part of a CLI
    # that another program reads, and a wrapper cannot tell a lie in it from
    # the truth.
    ("cli-train-always-reports-success", "src/cas13guide/cli.py",
     "    return 0 if report.accepted else 1",
     "    return 0",
     ["test_mutation_holes.py", "test_subcommands_end_to_end.py", "test_acceptance.py"]),
    ("cli-design-hides-that-nothing-ranked", "src/cas13guide/cli.py",
     "    return 0 if all_rows else 1",
     "    return 0",
     ["test_mutation_holes.py", "test_subcommands_end_to_end.py", "test_module_entry_point.py"]),
    # services/acceptance.py: the gate that decides whether a fitted model is
    # certifiable at all. Never mutated.
    ("acceptance-certifies-on-any-one-criterion",
     "src/cas13guide/services/acceptance.py",
     "        return bool(self.criteria) and all(c.passed for c in self.criteria)",
     "        return bool(self.criteria) and any(c.passed for c in self.criteria)",
     ["test_acceptance.py", "test_trained_end_to_end.py"]),
    # The same line's other half, and the reason it is written that way: an
    # empty criteria list makes `all()` True, so a report that evaluated
    # nothing would certify. That is the vacuous pass this repository has now
    # found in four separate instruments, guarded here in the package itself --
    # and a guard nothing pins is a guard that can be dropped by a refactor.
    ("acceptance-certifies-an-empty-report",
     "src/cas13guide/services/acceptance.py",
     "        return bool(self.criteria) and all(c.passed for c in self.criteria)",
     "        return all(c.passed for c in self.criteria)",
     ["test_mutation_holes.py", "test_acceptance.py"]),
    # services/provenance.py: the ledger digest is the identity two runs are
    # compared by. Unsorted keys make it depend on dict insertion order, so two
    # runs with identical content can disagree -- an identity that changes for
    # no reason is worse than none, which is the lesson the world digest in
    # transcriptome.py already carries.
    # services/fmindex.py: 292 lines, never mutated. The second boundary
    # defence, which rejects a walk result that would run off the end of its
    # locus. Loosening it invents an off-target spanning two transcripts --
    # a site that exists in the concatenated text and in no molecule.
    ("fmindex-accepts-a-hit-across-a-locus-boundary",
     "src/cas13guide/services/fmindex.py",
     "        if local < 0 or local + length > span.length:",
     "        if local < 0 or local + length > span.length + 1:",
     ["test_mutation_holes.py", "test_fmindex.py", "test_search.py", "test_properties.py"]),
    # Tried and removed: models/features.py's `max(seed_lo + 1, ...)`, which
    # keeps the seed window non-empty. With SEED_SPAN = (0.35, 0.75) the guard
    # only does anything at a spacer length of 1 -- at 2 the window is already
    # [0, 1) -- and the effector contract admits 22 to 30. So the mutation is
    # unobservable, the suite stays green, and the sweep calls it a hole when
    # what it has found is a line defending against an input that cannot
    # arrive. A probe that cannot fail is not a probe, and recording that here
    # is cheaper than someone rediscovering it.
    # pipeline/e_aggregate.py: which context's evidence an assessment carries.
    # `is None`, not `or` -- an empty reference tuple is falsy but is still the
    # reference, and falling through to a variant means the ranked row and the
    # off-target explorer describe a context the metrics were not computed
    # from. The comment beside it says so; nothing pinned it.
    ("aggregate-empty-reference-falls-through-to-a-variant",
     "src/cas13guide/pipeline/e_aggregate.py",
     "        if reference is None:",
     "        if not reference:",
     ["test_aggregate_stage.py", "test_context_evidence.py"]),
    # services/accessibility.py: whether a fold may tighten a calibrated bound.
    # Accessibility is independent evidence only when it comes from a source
    # the activity model did not already see -- SHAPE, DMS-seq, icSHAPE -- and
    # never from the thermodynamic fold, which the model used as a feature.
    # Flipping the default lets a run narrow its own interval using a number it
    # already counted, which is the one direction a bound must never move.
    ("accessibility-claims-to-be-independent-evidence",
     "src/cas13guide/services/accessibility.py",
     "    is_independent_evidence: bool = False",
     "    is_independent_evidence: bool = True",
     ["test_structure_and_signal.py", "test_bound_asymmetry.py"]),
    # pipeline/c_search.py: the universe-adequacy gate reads what the index
    # *proved* it searched, not what the spec declared. Trusting the
    # declaration was a measured hole -- the same spec and transcriptome went
    # from 42 FAIL to 42 PASS with zero off-target rows purely by narrowing the
    # index, because rRNA hits at 5000 TPM simply ceased to exist. This
    # restores it.
    ("search-gaps-trust-the-declaration-not-the-index",
     "src/cas13guide/pipeline/c_search.py",
     "    return frozenset((required | spec.context.declared_universes) - bound.universes)",
     "    return frozenset(required - spec.context.declared_universes)",
     ["test_search.py", "test_gate.py", "test_bugfix_regressions.py"]),
    # services/webapp.py: the page renders user-supplied sequence names. They
    # are escaped on the way in and this header is the belt to that pair of
    # braces, so loosening it removes a defence that only matters when the
    # other one has already failed.
    ("webapp-content-security-policy-allows-anything",
     "src/cas13guide/services/webapp.py",
     "default-src 'none'",
     "default-src *",
     ["test_webapp.py"]),
    # services/registry.py: the domain boundary. A model is asked only about
    # inputs it was validated on; beyond that the pipeline marks the answer
    # out-of-domain and refuses to narrow the interval. Loosening the mismatch
    # bound by one asks a trained model about a hit it never saw and then
    # believes the reply, which is the failure the whole domain check exists
    # to prevent.
    ("registry-domain-extends-past-what-the-model-saw",
     "src/cas13guide/services/registry.py",
     "            and n_mismatch <= self.max_mismatches",
     "            and n_mismatch <= self.max_mismatches + 1",
     ["test_arc_model.py", "test_trained_end_to_end.py", "test_bugfix_regressions.py", "test_gate.py"]),
    # contract/presets.py: whether the bundled miRNA presets have been checked
    # against a downloaded miRBase. The flag decides which of two limitations a
    # preset run states -- "these were verified against release N" or "these
    # have not been checked" -- and getting it backwards tells a user the
    # opposite of the truth about where their target sequence came from.
    ("presets-claim-the-wrong-verification-state",
     "src/cas13guide/contract/presets.py",
     "PRESETS_VERIFIED = True",
     "PRESETS_VERIFIED = False",
     ["test_limitations.py", "test_unproduced_limitations.py", "test_presets.py"]),
    # pipeline/g_output.py: the off-target explorer. This is the table a
    # validation plan points at -- "sequence the top off-target loci from the
    # explorer table" -- so its order is not cosmetic. Sorting ascending puts
    # the mildest hits at the top and `rows[:limit]` then keeps the safest,
    # discarding exactly the rows the table exists to show.
    ("off-target-table-lists-the-mildest-first",
     "src/cas13guide/pipeline/g_output.py",
     '    rows = sorted(worst_row.values(), key=lambda r: r["hazard"], reverse=True)',
     '    rows = sorted(worst_row.values(), key=lambda r: r["hazard"])',
     ["test_mutation_holes.py", "test_rejection_reporting.py", "test_bugfix_regressions.py", "test_gate.py"]),
    # The same table's other rule: one row per locus, keeping the *worst* hit
    # on it. Keeping the mildest instead reports a real off-target at a hazard
    # it does not have, under a column a reader ranks by.
    ("off-target-row-keeps-the-mildest-hit",
     "src/cas13guide/pipeline/g_output.py",
     '            if previous is None or row["hazard"] > previous["hazard"]:',
     '            if previous is None or row["hazard"] < previous["hazard"]:',
     ["test_mutation_holes.py", "test_rejection_reporting.py", "test_bugfix_regressions.py"]),
    # pipeline/f_decide.py: the REVIEW tier a guide is filed under when more
    # than one thing is borderline. It takes the *most* severe of them, because
    # the tier tells a reader how much of a human this needs. Taking the
    # mildest files a guide that wants a full review as one wanting a glance,
    # while every reason code beside it stays correct and complete -- the same
    # shape as the off-target table: the page is consistent and the emphasis
    # is inverted.
    ("review-tier-reports-the-mildest-concern",
     "src/cas13guide/pipeline/f_decide.py",
     "        tier = max((r[0] for r in review), key=lambda x: SEVERITY[x.value])",
     "        tier = min((r[0] for r in review), key=lambda x: SEVERITY[x.value])",
     ["test_gate.py", "test_decision_order.py", "test_bugfix_regressions.py"]),
    # The ranking itself, which is the headline output: the ordered list a
    # reader takes the top of. Inverting it puts the worst guides at rank 1
    # with every score on the page unchanged.
    ("ranking-puts-the-worst-guides-first",
     "src/cas13guide/pipeline/f_decide.py",
     "    scored.sort(key=lambda row: row[0], reverse=True)",
     "    scored.sort(key=lambda row: row[0])",
     ["test_decision_order.py", "test_gate.py"]),
    # pipeline/orchestrator.py: the search-escalation budget. Every loop here
    # carries one, and on exhaustion the run returns its best partial answer
    # with an explicit shortfall reason rather than spinning. Moving the
    # boundary by one lets a run take an escalation it was budgeted to refuse,
    # so the shortfall it should have recorded is never recorded -- and a
    # partial answer that does not say it is partial is the failure the budget
    # exists to make visible.
    ("orchestrator-takes-one-escalation-past-its-budget",
     "src/cas13guide/pipeline/orchestrator.py",
     "            if telemetry.search_escalations < spec.budget.max_search_escalations:",
     "            if telemetry.search_escalations <= spec.budget.max_search_escalations:",
     ["test_orchestrator.py", "test_subcommands_end_to_end.py", "test_bugfix_regressions.py", "test_gate.py"]),
    # services/workflow.py: a pasted sequence that is only whitespace. The
    # guard makes it fall through to "No sequence supplied. Paste a sequence,
    # upload a FASTA file, or pick a built-in target" -- an error a user can
    # act on. Without it the whitespace is treated as input and the reader gets
    # a parse failure about a sequence they did not knowingly supply.
    ("workflow-treats-whitespace-as-a-pasted-sequence",
     "src/cas13guide/services/workflow.py",
     "    if request.sequence_text and request.sequence_text.strip():",
     "    if request.sequence_text is not None:",
     ["test_mutation_holes.py", "test_the_input_doors_agree.py", "test_webapp.py", "test_subcommands_end_to_end.py"]),
    # The third text door, guarded the same way and found by looking for
    # siblings of the one that turned out to be unpinned. A whitespace
    # --variants box would become a variant world of nothing, and a run that
    # believes it has variant contexts computes robustness across them --
    # a number about a comparison that never happened.
    ("variants-door-treats-whitespace-as-variants",
     "src/cas13guide/services/workflow.py",
     "    elif request.variants_text and request.variants_text.strip():",
     "    elif request.variants_text is not None:",
     ["test_the_input_doors_agree.py", "test_context_evidence.py",
      "test_subcommands_end_to_end.py"]),
    # services/fasta.py: the parser's own empty check, one layer below those
    # doors. It is the message a user sees when nothing above caught it, and
    # it names both ways in: paste a sequence, or a FASTA record starting ">".
    ("fasta-parses-an-empty-input-instead-of-refusing",
     "src/cas13guide/services/fasta.py",
     "    if not text or not text.strip():",
     "    if text is None:",
     ["test_the_input_doors_agree.py", "test_bugfix_regressions.py"]),
    # A class not probed before: an arithmetic constant, rather than a bound
    # direction or a boundary. PRIOR_A is the Jeffreys prior on a binomial
    # rate, chosen over Beta(1, 1) because a uniform prior on a rate is not
    # uninformative under reparameterisation. Swapping it to 1.0 moves every
    # posterior and every credible interval this package reports, by a little,
    # in a way no shape or range check can see -- the numbers stay in [0, 1]
    # and stay ordered.
    ("panel-prior-becomes-uniform", "src/cas13guide/services/panel_confidence.py",
     "PRIOR_A = 0.5", "PRIOR_A = 1.0",
     ["test_panel_confidence.py", "test_bugfix_hunt_round11.py"]),
    # And the reported precision, which is not the same thing as the value. The
    # exported GC is rounded to four places; at one, distinct guides collapse
    # into ties and the column a reader sorts by stops separating them, while
    # every underlying number is unchanged.
    ("export-rounds-gc-to-one-place", "src/cas13guide/pipeline/g_output.py",
     '"gc": round(r.assessment.qc.gc, 4),',
     '"gc": round(r.assessment.qc.gc, 1),',
     ["test_rejection_reporting.py", "test_bugfix_regressions.py"]),
    # services/benchmark.py: the same all([]) hazard as the acceptance gate,
    # on the property both the `check` CI gate and the "conformal coverage"
    # acceptance criterion read. Found by grepping the package for every all()
    # over a collection that can be empty, after the acceptance one turned up.
    ("benchmark-certifies-coverage-it-never-measured",
     "src/cas13guide/services/benchmark.py",
     "        return bool(self.coverage) and all(c.ok for c in self.coverage)",
     "        return all(c.ok for c in self.coverage)",
     ["test_mutation_holes.py", "test_benchmark.py", "test_acceptance.py"]),
    ("provenance-digest-depends-on-key-order",
     "src/cas13guide/services/provenance.py",
     "        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)",
     "        payload = json.dumps(self.to_dict(), sort_keys=False, default=str)",
     ["test_mutation_holes.py", "test_reproducible_across_processes.py", "test_bugfix_regressions.py"]),
    (
        "scan-throughput-remeasured-twice-as-slow",
        "src/cas13guide/services/index_selection.py",
        "SCAN_SECONDS_PER_MB_PER_QUERY = 0.032",
        "SCAN_SECONDS_PER_MB_PER_QUERY = 0.064",
        ["test_index_selection.py"],
    ),
    (
        "terminal-mismatch-weight-raised",
        "src/cas13guide/util/duplex.py",
        "TERMINAL_WEIGHT = 0.25",
        "TERMINAL_WEIGHT = 0.30",
        ["test_mutation_holes.py"],
    ),
    (
        "expression-saturation-shifted",
        "src/cas13guide/services/context.py",
        "SATURATION_K = 50.0",
        "SATURATION_K = 60.0",
        ["test_mutation_holes.py"],
    ),
    (
        "homopolymer-limit-relaxed-by-one",
        "src/cas13guide/pipeline/b_qc.py",
        "MAX_HOMOPOLYMER = 4",
        "MAX_HOMOPOLYMER = 5",
        ["test_mutation_holes.py"],
    ),
    (
        "disfavoured-pfs-penalty-softened",
        "src/cas13guide/contract/effectors.py",
        "DISFAVOURED_PFS_FACTOR = 0.55",
        "DISFAVOURED_PFS_FACTOR = 0.60",
        ["test_mutation_holes.py"],
    ),
]


def _run(root: Path, modules: list[str] | None) -> tuple[bool, str]:
    command = [sys.executable, "-m", "pytest", "-q", "-x", "--no-header",
               "-p", "no:cacheprovider"]
    command += [f"tests/{name}" for name in modules] if modules else ["tests"]
    # PYTHONPATH, or the sweep cannot see half of what it thinks it tests.
    #
    # cas13guide is installed editable, and an editable install points at the
    # *real* src/. `tests/helpers.py` puts the copy's src first on sys.path, so
    # tests running in-process do import the mutated code -- but any test that
    # shells out (`python -m cas13guide.cli ...`) starts a fresh interpreter,
    # which resolves the package through the editable install and imports the
    # unmutated original. Every subprocess-based test therefore contributed
    # nothing to mutation coverage, silently: it passed under the mutation
    # because it never ran the mutation.
    #
    # Measured on a mutation in `cli.py` that a subprocess-driven test catches
    # in the real tree and the sweep called a hole. PYTHONPATH takes precedence
    # over the editable install's finder, and children inherit it, so this puts
    # the copy in front for the whole process tree.
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    try:
        done = subprocess.run(command, cwd=root, env=environment,
                              capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        # A mutation that hangs is caught -- an infinite loop is a failure the
        # suite would report if it ever finished. Reported as such rather than
        # raised, because an unhandled TimeoutExpired ends the sweep and loses
        # every verdict after it.
        return False, "TIMED OUT after 3600s"
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
                # Not __pycache__. Python invalidates a .pyc on source mtime
                # and size, and writing a mutation changes the mtime, so a
                # stale cache should never be used -- but "should" is the word
                # that starts every entry in this file. Copying compiled
                # bytecode into a tree whose whole purpose is to have its
                # source edited is a hazard with no upside, and it makes the
                # copy slower.
                shutil.copytree(source, root / item,
                                ignore=shutil.ignore_patterns("__pycache__"))
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
            # A screening module that does not exist is worse than a wrong
            # one. `pytest tests/nope.py` exits 4, the sweep reads any non-zero
            # exit as "the suite went red", and the mutation is reported caught
            # without ever having been judged. A stale hint would therefore
            # convert a mutation into a guaranteed false pass, silently and
            # permanently. Found by typing two module names that did not exist;
            # the other 45 hints were checked and all resolve.
            absent = [m for m in modules if not (root / "tests" / m).is_file()]
            if absent:
                reason = f"screening module(s) not found: {', '.join(absent)}"
                print(f"[BROKEN] {name}: {reason}")
                broken.append(f"{name} ({reason})")
                continue

            if original_code == mutated_code:
                reason = "the mutation is identical to the code it replaces"
                print(f"[BROKEN] {name}: {reason}")
                broken.append(f"{name} ({reason})")
                continue

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
        print("These mutations did NOT run -- an anchor that no longer")
        print("identifies exactly one place, a screening module that is not")
        print("there, or a mutation identical to the code it replaces.")
        print("Nothing above is evidence about what they guard:")
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
