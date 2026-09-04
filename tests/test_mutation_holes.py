"""Eight behaviours nothing pinned, found once the sweep could report a hole.

``examples/verify_mutation_coverage.py`` copies the tree to a scratch directory
and mutates the copy. It copied ``src``, ``tests`` and ``pyproject.toml`` --
but eight tests read ``README.md`` or ``examples/``, so the copied tree failed
six tests with two collection errors **before any mutation was applied**.

That is not a cosmetic problem. The sweep's logic is: screen a mutation against
the modules that ought to catch it, and if it survives, confirm against the
whole suite -- green means hole, red means caught. With the baseline
permanently red, the whole-suite run was never green, so ``holes`` could never
be non-empty. "0 survived" was true by construction rather than by
measurement, and the sweep was structurally incapable of reporting a hole.

Ten mutations took that path. Once the copy was fixed, eight of them turned out
to be genuine holes -- each a behaviour that can be reverted without a single
test noticing. They are pinned here.

Most are the same failure of imagination: a fixture with one item, where ``max``
and ``min`` agree; a weight of 1.0, where dropping the weight changes nothing;
a threshold tested at a value far from the boundary that moved. The tests below
were written to be *sensitive to the specific change*, which is what the
screening modules were not.

A ninth came from the first probe into ground the sweep had never touched.
``services/limitations.py`` -- the module carrying this tool's central promise,
that it tells you when it does not know -- had never been mutated at all, and
its most important caveat could be demoted from CRITICAL to WARNING with the
whole suite green. Every test asserted the text reached its surface; none
asserted the severity it reached them at. That is the last class here, and the
reason to keep pointing mutations at modules nothing has mutated yet.
"""

import random
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass
from cas13guide.contract.spec import TargetSpec
from cas13guide.models.evidence import MetricVector

#: Three targets that share no 22-mer, so tiling them produces distinct
#: spacers. `design_candidates` de-duplicates by spacer, so three copies of one
#: sequence would collapse into one target's worth of candidates and the budget
#: split under test would never be exercised.
DISTINCT_SEQUENCES = (
    "AUGGCACUGAAGUCCAUCGACUGGAACUUCCGAGUGCAUGGCAUUACCGAGUCCUAUGGCAAGUUCGAC",
    "CCUUAGGACGUUACCAGGUCAUUGCCAAGGUCUAGCAUCCGGAUUACGGCAUCAAGUCCGGAUUACGCA",
    "GAUCCAUGGUUCGAUCCAAGGCUUAGCCAUUGGACCUUAGGCAUUCCGAUUGGCCAAGUUACCGGAUCU",
)


class TestDiscriminationReadsTheWorstOffTarget(unittest.TestCase):
    """``max(off_upper)`` could become ``min(off_upper)`` unnoticed.

    The margin is on-target benefit divided by the worst forbidden hit. Taking
    the *least* bad one instead divides by a smaller number, so the guide reads
    as more discriminating than it is -- the direction a specificity metric must
    never move.

    The screening modules missed it because their fixtures carry one forbidden
    target, and over one element ``max`` and ``min`` are the same function.
    """

    def _spec(self):
        return make_spec(  # noqa: F405
            task="allele",
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A,  # noqa: F405
                           target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="f1", sequence=SEQ_B,  # noqa: F405
                           target_class=TargetClass.FORBIDDEN),
                TargetSpec(target_id="f2", sequence=SEQ_B,  # noqa: F405
                           target_class=TargetClass.FORBIDDEN),
            ),
        )

    def _evidence(self):
        return (
            make_evidence(0.80, 0.90, locus_id="L1", target_id="t1",  # noqa: F405
                          target_class=TargetClass.REQUIRED),
            make_evidence(0.05, 0.10, locus_id="Lf1", target_id="f1",  # noqa: F405
                          target_class=TargetClass.FORBIDDEN),
            make_evidence(0.30, 0.60, locus_id="Lf2", target_id="f2",  # noqa: F405
                          target_class=TargetClass.FORBIDDEN),
        )

    def test_the_margin_divides_by_the_worst_forbidden_hit(self):
        from cas13guide.metrics.aggregate import discrimination_margin

        spec = self._spec()
        if not spec.requires_discrimination:
            self.skipTest("fixture must be a discrimination task")
        margin = discrimination_margin(self._evidence(), spec)
        self.assertAlmostEqual(margin, 0.80 / 0.60, places=9)

    def test_and_not_by_the_mildest_one(self):
        """Stated separately because the first assertion alone would pass if
        the two forbidden uppers were ever made equal by a later edit."""
        from cas13guide.metrics.aggregate import discrimination_margin

        spec = self._spec()
        if not spec.requires_discrimination:
            self.skipTest("fixture must be a discrimination task")
        margin = discrimination_margin(self._evidence(), spec)
        self.assertLess(margin, 0.80 / 0.10)


class TestDesiredUtilityIsWeighted(unittest.TestCase):
    """``t.weight * best`` could lose its weight unnoticed.

    Every fixture used weight 1.0, where multiplying by the weight and not
    multiplying by it are the same arithmetic. A contract that says one desired
    target matters twice as much as another was silently not being honoured.
    """

    def _spec(self, first: float, second: float):
        return make_spec(  # noqa: F405
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A,  # noqa: F405
                           target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="d1", sequence=SEQ_B,  # noqa: F405
                           target_class=TargetClass.DESIRED, weight=first),
                TargetSpec(target_id="d2", sequence=SEQ_B,  # noqa: F405
                           target_class=TargetClass.DESIRED, weight=second),
            ),
        )

    def _evidence(self):
        return (
            make_evidence(0.50, 0.60, locus_id="Ld1", target_id="d1",  # noqa: F405
                          target_class=TargetClass.DESIRED),
            make_evidence(0.20, 0.30, locus_id="Ld2", target_id="d2",  # noqa: F405
                          target_class=TargetClass.DESIRED),
        )

    def test_utility_scales_with_the_declared_weight(self):
        from cas13guide.metrics.aggregate import desired_utility

        evidence = self._evidence()
        self.assertAlmostEqual(
            desired_utility(evidence, self._spec(3.0, 1.0)),
            3.0 * 0.50 + 1.0 * 0.20, places=9)

    def test_reweighting_the_contract_changes_the_utility(self):
        """The property that makes the weight load-bearing rather than
        decorative: two contracts differing only in weights must not score the
        same panel identically."""
        from cas13guide.metrics.aggregate import desired_utility

        evidence = self._evidence()
        self.assertNotAlmostEqual(
            desired_utility(evidence, self._spec(3.0, 1.0)),
            desired_utility(evidence, self._spec(1.0, 3.0)))


class TestTheCoverageGateFiresAtItsStatedThreshold(unittest.TestCase):
    """The boundary could move by 0.05 without a test noticing.

    Every gate test sat well clear of the threshold, so a guide judged against
    ``required_coverage_min - 0.05`` still landed in the bucket each fixture
    expected. A safety gate whose boundary is not tested at the boundary is not
    tested at all.
    """

    def _verdict_at(self, coverage: float):
        from cas13guide.pipeline.f_decide import gate

        spec = make_spec()  # noqa: F405
        metrics = MetricVector(  # noqa: F405
            required_coverage=coverage,
            desired_utility=0.0,
            worst_off_target=0.0,
            cumulative_burden=0.0,
            robustness=coverage,
            manufacturability=1.0,
        )
        result = gate(metrics, clean_qc(), (), exhaustive_bound(),  # noqa: F405
                      frozenset(), spec)
        return result.verdict, result.reasons

    def test_just_under_the_floor_is_not_a_pass(self):
        from cas13guide.contract.enums import Verdict

        floor = make_spec().thresholds.required_coverage_min  # noqa: F405
        verdict, _reasons = self._verdict_at(floor - 0.03)
        self.assertIsNot(verdict, Verdict.PASS,
                         f"coverage {floor - 0.03:.3f} is under the floor {floor}")

    def test_the_verdict_changes_across_the_stated_floor(self):
        """The boundary itself. The mutation moves the line, so only a pair
        straddling the line can see it move -- 0.03 under the floor is inside
        the 0.05 the mutation shifts it by."""
        floor = make_spec().thresholds.required_coverage_min  # noqa: F405
        under, _ = self._verdict_at(floor - 0.03)
        at, _ = self._verdict_at(floor)
        self.assertNotEqual(under, at)

    def test_the_reason_names_the_coverage_shortfall(self):
        """So the boundary is pinned by *why* it fired, not only by which
        bucket the guide landed in."""
        from cas13guide.contract.enums import ReasonCode

        floor = make_spec().thresholds.required_coverage_min  # noqa: F405
        _verdict, reasons = self._verdict_at(floor - 0.03)
        self.assertIn(ReasonCode.REQUIRED_TARGET_MISSED, reasons)


class TestPanelCoverageReadsTheLowerBound(unittest.TestCase):
    """``benefit_lower(e.activity)`` could become ``e.activity.upper``.

    The optimizer maximises expected covered targets, and every objective in
    this package is read at the bound unfavourable to the guide. Comparing on
    the *upper* bound instead lets a wide, doubtful hit displace a narrow
    confident one, and the value stored is still the loser's lower bound -- so
    a guide's coverage silently drops to that of its least certain site.

    It needs **two** hits on one target to see at all. With one row the
    comparison is against 0.0 and both bounds clear it, which is why every
    existing fixture missed it: they carry a single required hit per guide.
    """

    def _assessment(self, rows):
        """``rows``: (lower, upper) per hit, all on target ``t1``."""
        from dataclasses import replace

        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.models.evidence import Candidate, GateResult, GuideAssessment

        evidence = []
        for index, (lower, upper) in enumerate(rows):
            one = make_evidence(lower, upper, guide_id="g1",  # noqa: F405
                                locus_id="t1", target_id="t1",
                                target_class=TargetClass.REQUIRED)
            evidence.append(replace(one, hit=replace(one.hit, start=10 + 40 * index)))
        best = max(lower for lower, _ in rows)
        return GuideAssessment(
            guide_id="g1",
            candidate=Candidate(guide_id="g1", spacer="GCAUUGCUGUGGAGCUGAAGUC",
                                source_target_id="t1", start=10),
            qc=clean_qc("g1"),  # noqa: F405
            metrics=MetricVector(best, 0.0, 0.0, 0.0, best, 1.0),
            gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
            evidence=tuple(evidence),
        )

    #: A confident middling hit and a doubtful optimistic one. The first wins on
    #: the lower bound, the second wins on the upper.
    ROWS = [(0.60, 0.65), (0.40, 0.95)]

    def test_coverage_is_the_best_lower_bound(self):
        from cas13guide.pipeline.optimizer import _profile

        profile = _profile(self._assessment(self.ROWS))
        self.assertAlmostEqual(profile.coverage["t1"], 0.60, places=9)

    def test_a_doubtful_hit_does_not_displace_a_confident_one(self):
        """The same fact stated as the property it protects: adding a hit whose
        lower bound is worse must never lower a guide's coverage."""
        from cas13guide.pipeline.optimizer import _profile

        alone = _profile(self._assessment([self.ROWS[0]])).coverage["t1"]
        with_extra = _profile(self._assessment(self.ROWS)).coverage["t1"]
        self.assertGreaterEqual(with_extra, alone)

    def test_order_does_not_decide_it(self):
        """A max is order-independent; a comparison on the wrong bound is not."""
        from cas13guide.pipeline.optimizer import _profile

        forward = _profile(self._assessment(self.ROWS)).coverage["t1"]
        reverse = _profile(self._assessment(list(reversed(self.ROWS)))).coverage["t1"]
        self.assertAlmostEqual(forward, reverse, places=9)


class TestTheCandidateBudgetIsSpentInFull(unittest.TestCase):
    """The remainder of an uneven split could be dropped unnoticed.

    ``divmod(budget, n_targets)`` leaves a remainder that is handed to the
    earliest targets. Dropping it silently shrinks the search: with a budget of
    ten over three targets the run considers nine candidates and reports the
    budget as ten.
    """

    def _spec(self, budget: int, n_targets: int):
        spec = make_spec(  # noqa: F405
            targets=tuple(
                TargetSpec(target_id=f"t{i}", sequence=DISTINCT_SEQUENCES[i],
                           target_class=TargetClass.REQUIRED)
                for i in range(n_targets)
            ),
        )
        # The budget lives on `spec.budget`, not among the thresholds that
        # `make_spec(**overrides)` forwards.
        return spec.model_copy(update={
            "budget": spec.budget.model_copy(update={"max_candidates": budget})})

    def test_an_uneven_split_still_uses_every_slot(self):
        from cas13guide.pipeline.a_candidates import design_candidates

        # 10 over 3 targets: 4 + 3 + 3. Dropping the remainder gives 3 + 3 + 3.
        spec = self._spec(budget=10, n_targets=3)
        candidates = design_candidates(spec)
        self.assertEqual(len(candidates), 10)

    def test_an_even_split_is_unchanged(self):
        """The remainder path must not disturb the case that has none."""
        from cas13guide.pipeline.a_candidates import design_candidates

        self.assertEqual(len(design_candidates(self._spec(budget=9, n_targets=3))), 9)


class TestGroupAdequacyFiresAtItsStatedMinimum(unittest.TestCase):
    """``n_groups >= min`` could become ``>= min - 1``.

    A stratum one group short of the minimum would certify. The threshold
    exists because a quantile over too few independent transcripts is too
    variable to trust, and "one fewer than too few" is still too few.
    """

    def _service_with(self, n_groups: int):
        from cas13guide.services.calibration import CalibrationService

        service = CalibrationService()
        key = CalibrationService.stratum_key(
            "LwaCas13a", "in_vitro_detection", "mm0")
        for index in range(n_groups):
            for repeat in range(4):
                service.add_observation(key, 0.5 + 0.01 * repeat, 0.5,
                                        group=f"T{index:02d}")
        return service, key

    def test_one_group_short_of_the_minimum_is_not_adequate(self):
        from cas13guide.services.calibration import MIN_STRATUM_GROUPS

        service, key = self._service_with(MIN_STRATUM_GROUPS - 1)
        stratum = service._strata[key]
        self.assertFalse(service._groups_adequate(stratum))

    def test_exactly_the_minimum_is_adequate(self):
        """Both sides of the boundary, so the test sees the line move in
        either direction."""
        from cas13guide.services.calibration import MIN_STRATUM_GROUPS

        service, key = self._service_with(MIN_STRATUM_GROUPS)
        stratum = service._strata[key]
        self.assertTrue(service._groups_adequate(stratum))


class TestCoverageSlackIsTheStatedTolerance(unittest.TestCase):
    """``COVERAGE_SLACK`` could double without a test noticing.

    It is the tolerance the CI gate allows below nominal coverage. Doubling it
    accepts a stratum covering 0.83 against a nominal 0.90 -- and the module's
    own comment records that this constant is printed beside the verdict, so a
    silent change makes the printed threshold a lie as well.
    """

    def _report(self, empirical: float):
        from cas13guide.services.calibration import CoverageReport

        return CoverageReport(stratum="s", n=100, nominal=0.90,
                              empirical=empirical)

    def test_a_stratum_seven_points_under_nominal_is_not_ok(self):
        self.assertFalse(self._report(0.83).ok)

    def test_a_stratum_inside_the_stated_slack_is_ok(self):
        self.assertTrue(self._report(0.86).ok)

    def test_the_constant_is_the_one_the_boundary_uses(self):
        """Ties the two together, so moving the constant moves the boundary
        rather than leaving a second number behind."""
        from cas13guide.services.calibration import COVERAGE_SLACK

        just_inside = self._report(0.90 - COVERAGE_SLACK + 1e-9)
        just_outside = self._report(0.90 - COVERAGE_SLACK - 1e-9)
        self.assertTrue(just_inside.ok)
        self.assertFalse(just_outside.ok)


class TestThePanelFloorIsItsBestMember(unittest.TestCase):
    """``max(means)`` could become the mean of the means.

    The Fréchet floor is the comonotone case: a panel is worth at least its
    best member, because perfect dependence is the worst the marginals allow.
    Averaging instead reports a panel as *weaker* than a guide it contains,
    which is not a bound at all -- dropping the other members would raise the
    number.
    """

    def _model_and_panel(self, first: float, second: float):
        from cas13guide.pipeline.optimizer import Panel
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.panel_confidence import BayesianPanelConfidence
        from test_panel_confidence import guide

        # Two prediction clusters with opposite success rates, so the two
        # members get genuinely different posteriors. With one cluster every
        # member shares a mean, and max and mean agree -- which is how this
        # went unpinned in the first place.
        calibration = CalibrationService()
        key = CalibrationService.stratum_key(
            "RfxCas13d", "cell_culture_knockdown", "mm0")
        for index in range(30):
            calibration.add_observation(key, 0.95, 0.9 if index % 10 else 0.1,
                                        group=f"H{index % 10:02d}")
        for index in range(30):
            calibration.add_observation(key, 0.30, 0.9 if index % 10 == 0 else 0.1,
                                        group=f"L{index % 10:02d}")
        spec = make_spec()  # noqa: F405
        panel = Panel(guides=(guide("gA", {"t1": first}), guide("gB", {"t1": second})),
                      coverage={}, objective=0.0, guarantee="test")
        return BayesianPanelConfidence(calibration, spec), panel

    def test_the_floor_is_the_best_member_not_their_average(self):
        model, panel = self._model_and_panel(0.95, 0.30)
        target = model.for_panel(panel).by_target().get("t1")
        if target is None or len(target.members) < 2:
            self.skipTest("fixture must place two members on one target")
        means = sorted(member.mean for member in target.members)
        self.assertAlmostEqual(target.lower, means[-1], places=9)
        self.assertGreater(means[-1], sum(means) / len(means),
                           "fixture must give the members different means")

    def test_a_panel_is_never_worth_less_than_a_guide_it_contains(self):
        """The invariant in the form that makes it a bound: adding a weaker
        member must not lower the floor."""
        model, panel = self._model_and_panel(0.95, 0.30)
        target = model.for_panel(panel).by_target().get("t1")
        if target is None or len(target.members) < 2:
            self.skipTest("fixture must place two members on one target")
        for member in target.members:
            with self.subTest(guide=member.guide_id):
                self.assertGreaterEqual(target.lower, member.mean)


class TestTheCriticalCaveatsStayCritical(unittest.TestCase):
    """A CRITICAL demoted to a warning still says every word it said before.

    Found by the first probe into ``services/limitations.py``, which nothing
    had ever mutated -- and which carries the product's central promise: the
    guide designer that tells you when it does not know. Demoting
    ``synthetic-calibration`` from CRITICAL to WARNING left the whole suite
    green. Every test asserted the *text* reached its surface; none asserted
    what severity it reached them at.

    That is the difference between a caveat a reader must act on and one they
    may note in passing. "Every interval, and therefore every PASS, is
    exercisable rather than earned" is not a warning, and the severity mark is
    the only thing on the page that says so -- it also sets the ordering, so a
    demotion moves it below the notes a reader is meant to skim past.

    ``test_limitations.py`` already pins the severity of two limitations. These
    are the rest.
    """

    #: Every limitation that must reach a reader as CRITICAL. Stated as a set
    #: rather than one assertion each, so a *promotion* is caught too: a run
    #: that starts calling an ordinary caveat critical is its own defect, and
    #: it devalues the mark for the four that have earned it.
    MUST_BE_CRITICAL = frozenset({
        "synthetic-calibration",
        "heuristic-activity-model",
        "collateral-not-predicted",
        "search-scope-is-what-you-loaded",
    })

    def _found(self):
        from cas13guide.contract.enums import AssayType, Task
        from cas13guide.services.limitations import limitations_for_run
        from test_limitations import _Model, _Registry, spec_for

        class MarkedSynthetic:
            """A calibration that says its data was generated, not measured."""

            synthetic = True

        # The bound is what makes `search-scope-is-what-you-loaded` apply: a
        # scope caveat is only stated when a search actually ran.
        return limitations_for_run(
            spec_for(assay=AssayType.CELL_CULTURE_KNOCKDOWN, task=Task.KNOCKDOWN),
            registry=_Registry(activity=_Model("model/heuristic-cas13/v1")),
            calibration=MarkedSynthetic(),
            bound=exhaustive_bound(4),  # noqa: F405
        )

    def test_every_caveat_that_must_be_critical_is_critical(self):
        from cas13guide.services.limitations import Severity

        by_id = {item.id: item for item in self._found()}
        for identifier in sorted(self.MUST_BE_CRITICAL):
            with self.subTest(limitation=identifier):
                if identifier not in by_id:
                    self.skipTest(f"{identifier} does not apply to this fixture")
                self.assertIs(by_id[identifier].severity, Severity.CRITICAL)

    def test_the_synthetic_calibration_caveat_specifically(self):
        """Named on its own because it is the one that was demotable, and the
        one whose subject is that no PASS on this tool has been earned."""
        from cas13guide.services.limitations import Severity

        by_id = {item.id: item for item in self._found()}
        self.assertIn("synthetic-calibration", by_id)
        self.assertIs(by_id["synthetic-calibration"].severity, Severity.CRITICAL)

    def test_nothing_else_claims_that_severity(self):
        """The other direction. A mark that everything carries marks nothing."""
        from cas13guide.services.limitations import Severity

        critical = {item.id for item in self._found()
                    if item.severity is Severity.CRITICAL}
        self.assertTrue(critical <= self.MUST_BE_CRITICAL,
                        f"unexpected CRITICAL: {sorted(critical - self.MUST_BE_CRITICAL)}")

    def test_critical_caveats_sort_ahead_of_the_rest(self):
        """Severity is not only a label: it orders the list, so a reader who
        stops after one line has read the one that matters."""
        from cas13guide.services.limitations import Severity

        found = list(self._found())
        rank = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}
        positions = [rank[item.severity] for item in found]
        self.assertEqual(positions, sorted(positions),
                         "limitations are not ordered by severity")


class TestTheHtmlReportMarksItsCriticalNotes(unittest.TestCase):
    """The tenth hole, and the ninth one surface further out.

    Fixing the severity at the source pinned what ``limitations_for_run``
    returns. It did not pin what the served page does with it. The report marks
    a note critical by testing its text::

        f'<div class="note{" crit" if n.upper().startswith("CRITICAL") else ""}">'

    Change that prefix and every CRITICAL renders as an ordinary note: same
    words, same position, no mark. The suite stayed green, because the tests
    that read the page look for the *text* -- which is exactly the assumption
    that let the source-level demotion through.

    Worth stating as its own class rather than folding into the one above: a
    caveat has to survive every surface it crosses, and each surface is a
    separate place for it to lose its severity.
    """

    CRITICAL_NOTE = (
        "CRITICAL: Conformal bounds were calibrated on SYNTHETIC data. Every "
        "interval, and therefore every PASS, is exercisable rather than earned."
    )
    ORDINARY_NOTE = "INFO: Every score on this page is a prediction."

    def _page(self):
        from cas13guide.services.report_html import render_report

        return render_report({"verdict_counts": {}, "ranked": []},
                             title="T", subtitle="S",
                             notes=[self.CRITICAL_NOTE, self.ORDINARY_NOTE])

    def test_a_critical_note_carries_the_critical_class(self):
        import re

        page = self._page()
        marked = re.findall(r'<div class="note crit">(.*?)</div>', page, re.S)
        self.assertTrue(marked, "no note rendered with the critical mark")
        self.assertTrue(any("SYNTHETIC" in block for block in marked),
                        "the CRITICAL note is not the one carrying the mark")

    def test_an_ordinary_note_does_not(self):
        """Both directions. A mark every note carries marks nothing."""
        import re

        page = self._page()
        marked = re.findall(r'<div class="note crit">(.*?)</div>', page, re.S)
        self.assertFalse(any("prediction" in block for block in marked),
                         "an INFO note was rendered as critical")

    def test_both_notes_still_reach_the_page(self):
        """The mark is the thing under test, not whether the text arrives --
        and a test that only checked arrival is what missed this."""
        page = self._page()
        self.assertIn("SYNTHETIC", page)
        self.assertIn("prediction", page)


class TestTheReferenceContextIsTheReferenceOne(unittest.TestCase):
    """An eleventh hole, in ``pipeline/orchestrator.py``.

    One line decides which context is *the* reference::

        reference = next((c for c in contexts if c.name == "reference"), contexts[0])

    Invert that test and a variant is chosen, with the whole suite green.

    Working out what that actually breaks was the interesting part, and it is
    less than it first looks. Each context searches its own index either way --
    ``ctx_outcome = outcome if ctx is reference else run_search(..., ctx.index)``
    -- so the reference variable only decides which context *reuses* the
    already-computed outcome, and the per-context evidence is identical. A test
    written against the evidence sees nothing, which is what the first attempt
    here did.

    What it does decide is read from the reference's transcriptome directly:
    the exon-junction map handed to tiling, the envelope escalation, and the
    world the non-targeting controls are screened against. The junction map is
    the cleanest of the three to observe, and it is the one whose absence is
    silent -- ``spans_junction`` was False on every guide in every real run
    until that wire was added, so a run reading it from the wrong world looks
    exactly like the bug that already happened once.

    The fixture puts the variant *first*, because the fallback is
    ``contexts[0]`` and a reference that happens to be first would pass under
    either version of the line.
    """

    def _spans_junction(self) -> int:
        from cas13guide.contract.enums import Universe
        from cas13guide.pipeline.orchestrator import DesignContext, run_design
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.indexes import ExhaustiveScanIndex
        from cas13guide.services.registry import (
            HeuristicCas13Model,
            ModelRegistry,
            NullAccessibilityModel,
        )
        from cas13guide.services.transcriptome import Locus, Transcriptome

        spec = make_spec()  # noqa: F405
        universes = spec.context.declared_universes

        def context(name: str, junctions: tuple) -> DesignContext:
            world = Transcriptome("GRCh38", "v44")
            world.add(Locus("L_T1", SEQ_A, Universe.CODING,  # noqa: F405
                            gene="TGT", expression=100.0, junctions=junctions))
            return DesignContext(name, world,
                                 ExhaustiveScanIndex(world, universes),
                                 {"L_T1": "t1"})

        rng = random.Random(5)
        calibration = CalibrationService()
        for regime in ("mm0", "mm1-2", "mm3+"):
            key = CalibrationService.stratum_key(
                spec.profile.effector.value, spec.profile.assay.value, regime)
            for index in range(150):
                truth = rng.random()
                calibration.add_observation(
                    key, min(1.0, max(0.0, truth + rng.gauss(0, 0.1))), truth,
                    group=f"T{index % 12:02d}")

        # Only the reference world knows where the exon junction is.
        variant = context("v1", ())
        reference = context("reference", (30,))
        report = run_design(
            spec, [variant, reference],          # deliberately not first
            ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel()),
            calibration, panel_size=2,
        )
        return sum(1 for entry in report.ranked
                   if entry.assessment.candidate.spans_junction)

    def test_the_junction_map_comes_from_the_reference_world(self):
        self.assertGreater(
            self._spans_junction(), 0,
            "no guide was marked as spanning the reference world's junction")


class TestAnEmptyTableIsNotAnEmptyRun(unittest.TestCase):
    """A twelfth hole, in ``cli.py``, guarding a fix nothing had pinned.

    ``rows`` is the *display* list: it has been through the distinct-site
    filter and ``--top``. ``all_rows`` is what the run produced. The branch
    that distinguishes them reads ``elif all_rows:``, and reading ``rows``
    instead prints "no guide cleared the gate" whenever a display limit
    empties the table -- next to exports carrying every one of those guides.

    The comment beside that line records this as a defect that was already
    found and fixed once. Nothing pinned the fix, so it could have been
    reverted by any refactor that mistook the two names for each other.

    Driven through the terminal rather than by calling the function, because
    the defect is in what a user reads.
    """

    TARGET = (
        "GGACAUCGAAUCGCGGAACUUGGCAUCAAGCUAGCUAGGCAUCGAUCGAUCAGCUAGCAUCGAUCGGAUCC"
        "AUGGAUUCCAGGCACUUCUCGAUCAUAGGACUCGGUAAUGCCAUGAGUCGAUGGACUUCAGUGCCAUCGAU"
    )

    def _run(self, *extra):
        import pathlib as _pathlib
        import subprocess
        import sys
        import tempfile

        root = _pathlib.Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            fasta = _pathlib.Path(directory) / "t.fa"
            newline = chr(10)
            fasta.write_text(f">T1{newline}{self.TARGET}{newline}",
                             encoding="utf-8")
            done = subprocess.run(
                [sys.executable, "-m", "cas13guide.cli", "design",
                 "--fasta", str(fasta), "--task", "knockdown",
                 "--assay", "cell_culture_knockdown", "--no-signal",
                 "--out", str(_pathlib.Path(directory) / "out"), "--stem", "d",
                 *extra],
                cwd=root, capture_output=True, text=True, timeout=900,
            )
        return done.stdout

    def test_a_display_limit_does_not_report_an_empty_run(self):
        """--top 0 empties the table while the run ranked guides."""
        out = self._run("--top", "0")
        self.assertIn("none displayed", out)
        self.assertNotIn("no guide cleared the gate", out)

    def test_it_counts_what_the_run_produced_not_what_was_shown(self):
        """`all_rows`, not `shown`: the count must not move with --all-sites,
        a display flag, in the one sentence that says what the run produced."""
        import re

        plain = re.search(r"(\d+) guide\(s\) ranked, none displayed",
                          self._run("--top", "0"))
        every = re.search(r"(\d+) guide\(s\) ranked, none displayed",
                          self._run("--top", "0", "--all-sites"))
        self.assertIsNotNone(plain)
        self.assertIsNotNone(every)
        self.assertEqual(plain.group(1), every.group(1))

    def test_a_run_that_shows_its_table_says_neither(self):
        """The complement, on a fixture that is known to rank guides: when the
        table prints, neither empty-run message may appear. Without this the
        first two tests would pass on a build that printed the "none
        displayed" line unconditionally."""
        out = self._run("--top", "5")
        self.assertIn("rank", out)
        self.assertNotIn("none displayed", out)
        self.assertNotIn("no guide cleared the gate", out)


class TestTheExitCodesMeanWhatTheySay(unittest.TestCase):
    """The thirteenth and fourteenth holes, and the ones that prove the fix
    above was worth making.

    A status code is the part of a CLI another program reads, and a wrapper
    cannot tell a lie in it from the truth. Both contracts were unpinned:
    ``train`` could always report success while the acceptance gate rejected
    the dataset, and ``design`` could always report success while nothing
    ranked.

    They were also *unreachable* by the mutation sweep until it started putting
    the mutated copy on PYTHONPATH: an exit code is only observable through a
    subprocess, and a subprocess was importing the installed package rather
    than the mutation. So these two holes existed behind a blind spot, which is
    the argument for having closed it.

    The existing CLI tests assert 0 for success and 2 for a usage or input
    error. Nothing asserted 1 -- the status that says the run completed and its
    answer was "no".
    """

    NOTHING_RANKS = "AAAAAAUUUUUU" * 12   # every window fails QC on homopolymers

    def _cli(self, *argv, expect_files: bool = True):
        import pathlib as _pathlib
        import subprocess
        import sys
        import tempfile

        root = _pathlib.Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as directory:
            work = _pathlib.Path(directory)
            done = subprocess.run(
                [sys.executable, "-m", "cas13guide.cli", *argv,
                 *(("--out", str(work / "out"), "--stem", "d")
                   if expect_files else ())],
                cwd=root, capture_output=True, text=True, timeout=900,
            )
        return done

    def test_design_reports_one_when_nothing_ranked(self):
        import pathlib as _pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            fasta = _pathlib.Path(directory) / "t.fa"
            newline = chr(10)
            fasta.write_text(f">lowcomplex{newline}{self.NOTHING_RANKS}{newline}",
                             encoding="utf-8")
            done = self._cli("design", "--fasta", str(fasta), "--task", "knockdown",
                             "--assay", "cell_culture_knockdown", "--no-signal")
        self.assertEqual(done.returncode, 1, done.stdout[-600:])

    def test_design_reports_zero_when_guides_ranked(self):
        """Both sides, or "always 1" would pass as readily as "always 0"."""
        import pathlib as _pathlib
        import tempfile

        target = (
            "GGACAUCGAAUCGCGGAACUUGGCAUCAAGCUAGCUAGGCAUCGAUCGAUCAGCUAGCAUCGAUCGGAUCC"
            "AUGGAUUCCAGGCACUUCUCGAUCAUAGGACUCGGUAAUGCCAUGAGUCGAUGGACUUCAGUGCCAUCGAU"
        )
        with tempfile.TemporaryDirectory() as directory:
            fasta = _pathlib.Path(directory) / "t.fa"
            newline = chr(10)
            fasta.write_text(f">T1{newline}{target}{newline}", encoding="utf-8")
            done = self._cli("design", "--fasta", str(fasta), "--task", "knockdown",
                             "--assay", "cell_culture_knockdown", "--no-signal")
        self.assertEqual(done.returncode, 0, done.stdout[-600:])

    def _noise_screen(self, path):
        import random as _random

        rng = _random.Random(7)
        rows = ["spacer,activity,transcript"]
        for index in range(400):
            spacer = "".join(rng.choice("ACGU") for _ in range(22))
            rows.append(f"{spacer},{rng.random():.4f},T{index % 20:02d}")
        path.write_text(chr(10).join(rows) + chr(10), encoding="utf-8")
        return path

    def test_train_reports_one_when_the_gate_rejects_the_dataset(self):
        """Labels that are noise: the dataset loads, and the acceptance gate
        refuses to certify a model fitted on it. Exit 1 is what a CI job reads
        to know that happened -- and 2 already means something else, a dataset
        that could not be read at all."""
        import pathlib as _pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            screen = self._noise_screen(_pathlib.Path(directory) / "noise.csv")
            done = self._cli("train", str(screen), expect_files=False)
        self.assertEqual(done.returncode, 1, done.stdout[-600:])

    def test_and_one_is_distinct_from_the_unreadable_status(self):
        """The distinction round twelve added: a malformed file exits 2, a
        rejected dataset exits 1. Collapsing them makes a CI job unable to tell
        "your data is bad" from "your data did not support a model"."""
        import pathlib as _pathlib
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            missing = _pathlib.Path(directory) / "not-here.csv"
            done = self._cli("train", str(missing), expect_files=False)
        self.assertEqual(done.returncode, 2, done.stdout[-400:])


if __name__ == "__main__":
    unittest.main()
