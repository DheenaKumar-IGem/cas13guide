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
import pathlib
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.effectors import EFFECTORS
from cas13guide.contract.enums import Effector, QCFlag, TargetClass
from cas13guide.contract.spec import TargetSpec
from cas13guide.models.evidence import MetricVector
from cas13guide.pipeline.b_qc import qc_candidate
from cas13guide.services.context import abundance_factor
from cas13guide.util.duplex import SEED_SPAN, seed_weight

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


class TestAnEmptyReportDoesNotCertify(unittest.TestCase):
    """The fifteenth hole, and the one that names the pattern.

    ``AcceptanceReport.accepted`` reads::

        return bool(self.criteria) and all(c.passed for c in self.criteria)

    The ``bool(self.criteria)`` is there because ``all([])`` is True: a report
    that evaluated nothing would otherwise certify. That is the vacuous pass
    this repository has now found in four separate instruments -- an empty case
    list clearing a tolerance, an empty vocabulary reporting 0/0, an empty
    inventory reporting no dead values, a document dropping out of a
    comparison. Here it is guarded in the package itself.

    And nothing pinned the guard. Dropping it left the suite green, so a
    refactor tidying the expression could have removed the one clause standing
    between "we checked six things and they passed" and "we checked nothing".
    """

    def _report(self, criteria):
        from cas13guide.services.acceptance import AcceptanceReport

        return AcceptanceReport(criteria=tuple(criteria))

    def _criterion(self, passed: bool):
        from cas13guide.services.acceptance import Criterion

        return Criterion("rows", passed, 100.0, 50.0)

    def test_a_report_with_no_criteria_is_not_accepted(self):
        self.assertFalse(self._report([]).accepted)

    def test_a_report_whose_criteria_all_pass_is_accepted(self):
        """The complement, so "never accept" would not pass as readily."""
        self.assertTrue(self._report([self._criterion(True),
                                      self._criterion(True)]).accepted)

    def test_one_failing_criterion_is_enough_to_refuse(self):
        """`all`, not `any`: the gate asks whether every check passed."""
        self.assertFalse(self._report([self._criterion(True),
                                       self._criterion(False)]).accepted)

    def test_the_summary_agrees_with_the_verdict(self):
        """The word a reader sees and the boolean a caller switches on are one
        decision, and this file has already found two places where two
        descriptions of one thing drifted apart."""
        self.assertIn("REJECTED", self._report([]).summary())
        self.assertIn("ACCEPTED",
                      self._report([self._criterion(True)]).summary())


class TestTheLedgerDigestIsAContentIdentity(unittest.TestCase):
    """The sixteenth hole, in ``services/provenance.py``.

    The digest exists so two runs can be compared. ``json.dumps(...,
    sort_keys=True)`` is what makes it a function of the *content*: a record's
    ``detail`` is built from keyword arguments, so its key order follows
    whatever order the caller happened to write them in. Without the sort, two
    runs recording the same facts in a different argument order get different
    digests.

    An identity that changes for no visible reason is worse than none -- the
    same lesson ``Transcriptome.digest`` carries about load order, which *is*
    pinned. This one was not.
    """

    def _ledger(self, **detail):
        from cas13guide.services.provenance import ProvenanceLedger

        ledger = ProvenanceLedger("spec-fingerprint")
        ledger.record("model", "activity", "act/v1", **detail)
        return ledger

    def test_argument_order_does_not_change_the_digest(self):
        first = self._ledger(alpha=1, beta=2, gamma=3)
        second = self._ledger(gamma=3, alpha=1, beta=2)
        self.assertEqual(first.digest(), second.digest())

    def test_but_the_content_does(self):
        """Or a digest that ignored everything would pass the first test."""
        self.assertNotEqual(self._ledger(alpha=1).digest(),
                            self._ledger(alpha=2).digest())

    def test_it_is_stable_across_repeated_calls(self):
        ledger = self._ledger(alpha=1, beta=2)
        self.assertEqual(ledger.digest(), ledger.digest())


class TestCoverageIsNotCertifiedUnmeasured(unittest.TestCase):
    """The seventeenth hole, and the first found by reasoning rather than
    mutating.

    After the acceptance gate's ``bool(self.criteria) and all(...)`` turned out
    to be unpinned, the obvious next question was where else in the package an
    ``all()`` runs over a collection that can be empty. Seven places; four
    already guarded, one correct to answer True on nothing (a panel with no
    required targets is trivially feasible), one whose criteria are built two
    lines above it -- and this.

    ``BenchmarkReport.coverage_ok`` ran ``all(c.ok for c in self.coverage)``
    with no guard. The coverage tuple is built per stratum from the benchmark
    cases, so a run that evaluated no cases builds none, and the property
    answered "yes, the bounds mean what they claim" having measured nothing.

    Two gates read it: ``cas13guide check`` exits 0 on it, and it is the
    "conformal coverage" criterion in the acceptance report. So an empty
    benchmark certified a model on the axis that says its intervals are
    trustworthy.

    Empty is not ok, on purpose. Coverage is the claim that a calibrated
    interval contains the truth about as often as it says; with nothing
    measured there is no evidence for it, and absence of evidence has been the
    wrong answer everywhere else in this package too.
    """

    def _report(self, coverage):
        from cas13guide.services.benchmark import BenchmarkReport

        return BenchmarkReport(n=len(coverage), spearman=0.9, mae=0.1,
                               coverage=tuple(coverage))

    def _stratum(self, ok: bool):
        from cas13guide.services.calibration import CoverageReport

        # empirical well above or well below nominal, so `ok` is unambiguous
        return CoverageReport(stratum="s", n=100, nominal=0.90,
                              empirical=0.95 if ok else 0.10)

    def test_a_benchmark_with_no_strata_is_not_covered(self):
        self.assertFalse(self._report([]).coverage_ok)

    def test_a_benchmark_whose_strata_all_cover_is(self):
        """The complement, so "never ok" would not pass as readily."""
        self.assertTrue(self._report([self._stratum(True),
                                      self._stratum(True)]).coverage_ok)

    def test_one_uncovered_stratum_is_enough_to_refuse(self):
        self.assertFalse(self._report([self._stratum(True),
                                       self._stratum(False)]).coverage_ok)


class TestTheSecondBoundaryDefenceHolds(unittest.TestCase):
    """Not a hole -- a redundant guard, which is a different thing.

    ``fmindex.py`` documents two independent defences against a match spanning
    two loci: the separator is excluded from the walk alphabet, and ``_locate``
    rejects any offset whose window would run off the end of its locus. The
    comment says the redundancy is intentional, "a cross-locus hit is an
    invented off-target, and it would be reported as a real one".

    So the sweep reports the second defence as unpinned, and it is right that
    nothing catches a change to it -- while the first defence stands, the walk
    never produces an offset for it to reject. Neither guard can be observed
    failing through the search, which is what defence in depth means.

    That makes it worth testing *directly* rather than worth removing. These
    call ``_locate`` with the offsets the walk is not supposed to be able to
    produce, so the redundancy is verified rather than assumed -- and if the
    separator defence is ever weakened, this one is known to work.
    """

    def _index(self):
        from cas13guide.contract.enums import Universe
        from cas13guide.services.fmindex import FMIndexSearchIndex
        from cas13guide.services.transcriptome import Locus, Transcriptome

        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("L_A", "ACGU" * 10, Universe.CODING, expression=1.0))
        world.add(Locus("L_B", "GGCC" * 10, Universe.CODING, expression=1.0))
        return FMIndexSearchIndex(world, frozenset({Universe.CODING}))

    def _spans(self, index):
        return getattr(index, "_spans", None) or index._index._spans

    def test_a_window_inside_a_locus_is_located(self):
        index = self._index()
        spans = self._spans(index)
        first = spans[0]
        self.assertEqual(index._locate(first.start, 10), (first.locus_id, 0))

    def test_a_window_running_past_the_end_of_its_locus_is_refused(self):
        """The offset the separator is supposed to make unreachable. If the
        walk ever produces one, this is what stops it becoming an off-target
        row naming a site that exists in the concatenated text and in no
        molecule."""
        index = self._index()
        spans = self._spans(index)
        first = spans[0]
        # Exactly one base past the end, which is the case that separates a
        # correct guard from one loosened by an off-by-one: a window starting
        # further back is refused either way and proves nothing about where
        # the line sits.
        overshoot_by_one = first.start + first.length - 10 + 1
        self.assertIsNone(index._locate(overshoot_by_one, 10))

    def test_the_last_window_that_fits_is_still_accepted(self):
        """The boundary itself, so the guard cannot be tightened into
        discarding real hits either."""
        index = self._index()
        spans = self._spans(index)
        first = spans[0]
        located = index._locate(first.start + first.length - 10, 10)
        self.assertEqual(located, (first.locus_id, first.length - 10))


class TestTheOffTargetTableShowsTheWorstHits(unittest.TestCase):
    """The eighteenth and nineteenth holes, both in the table a validation
    plan points at.

    ``g_output._off_target_rows`` keeps one row per (guide, locus) carrying the
    *worst* context, then sorts by hazard descending and truncates to a limit.
    Two rules, neither pinned:

    * sorting ascending puts the mildest hits at the top, and ``rows[:limit]``
      then keeps the safest -- discarding exactly the rows the table exists to
      show;
    * keeping the mildest hit per locus reports a real off-target at a hazard
      it does not have, in the column a reader ranks by.

    The comment on that column says it runs the same code path as
    ``worst_off_target`` "so the column a reviewer eyeballs cannot drift from
    the metric it reproduces". That was true of the number and not of the
    ordering, and the README tells a reader to "sequence the top off-target
    loci from the explorer table" -- so an inverted sort sends them to the
    wrong bench work while every individual hazard on the page stays correct.
    """

    def _assessment(self, guide_id, hits):
        """``hits``: (locus_id, activity_upper) per off-target."""
        from dataclasses import replace

        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.models.evidence import Candidate, GateResult, GuideAssessment

        evidence = []
        for locus_id, upper in hits:
            # target_class None is the unannotated case, and `is_intended` is
            # false for it -- which is what makes these off-targets.
            one = make_evidence(  # noqa: F405
                0.0, upper, guide_id=guide_id, locus_id=locus_id,
                target_id=None, target_class=None,
            )
            evidence.append(replace(one, hit=replace(one.hit, target_id=None)))
        return GuideAssessment(
            guide_id=guide_id,
            candidate=Candidate(guide_id=guide_id, spacer="GCAUUGCUGUGGAGCUGAAGUC",
                                source_target_id="t1", start=0),
            qc=clean_qc(guide_id),  # noqa: F405
            metrics=MetricVector(0.5, 0.0, 0.0, 0.0, 0.5, 1.0),
            gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
            evidence=tuple(evidence),
            context_evidence=tuple(evidence),
        )

    def _rows(self, assessments, limit=50):
        from cas13guide.pipeline.g_output import _off_target_rows

        return _off_target_rows(assessments, limit=limit)

    def test_rows_are_ordered_worst_hazard_first(self):
        rows = self._rows([self._assessment(
            "g1", [("mild", 0.10), ("severe", 0.90), ("middling", 0.50)])])
        hazards = [r["hazard"] for r in rows]
        self.assertEqual(hazards, sorted(hazards, reverse=True), rows)

    def test_the_limit_keeps_the_worst_not_the_safest(self):
        """The two rules compound: an ascending sort plus a truncation drops
        precisely the rows a reader is told to act on."""
        rows = self._rows([self._assessment(
            "g1", [("mild", 0.10), ("severe", 0.90), ("middling", 0.50)])], limit=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["locus"], "severe")

    def test_one_row_per_locus_carries_its_worst_hit(self):
        """Two hits on one locus collapse to one row, and it must be the worse
        of them -- the row stands in for the locus."""
        rows = self._rows([self._assessment(
            "g1", [("shared", 0.20), ("shared", 0.80)])])
        self.assertEqual(len(rows), 1, rows)
        self.assertAlmostEqual(rows[0]["hazard"],
                               max(r["hazard"] for r in rows), places=9)
        self.assertGreater(rows[0]["hazard"], 0.2 * 0.5)

    def test_the_hazard_column_matches_the_metric_it_reproduces(self):
        """The claim the comment beside that column makes."""
        from cas13guide.metrics.aggregate import pair_hazard

        assessment = self._assessment("g1", [("only", 0.60)])
        rows = self._rows([assessment])
        expected = round(pair_hazard(assessment.off_targets()[0]), 4)
        self.assertAlmostEqual(rows[0]["hazard"], expected, places=9)


class TestAWhitespacePasteIsNoSequence(unittest.TestCase):
    """The twentieth hole, at the door a browser user comes in through.

    ``_read_records`` tries the input doors in order, and each text door is
    guarded with ``.strip()`` so that a box containing only spaces or newlines
    counts as empty and falls through to::

        No sequence supplied. Paste a sequence, upload a FASTA file, or pick a
        built-in target.

    Without the guard the whitespace is treated as supplied input, and the
    reader gets a parse failure about a sequence they did not knowingly give.
    That is the difference between an error naming what to do next and one
    naming a symptom -- the distinction this package makes everywhere else, in
    the one place a user is most likely to arrive by accident.

    Both doors are checked, because they are guarded separately and a fix to
    one has already been forgotten in the other elsewhere in this codebase.
    """

    WHITESPACE = ("   ", chr(10) + chr(9) + " ", "", chr(10) * 3)

    def _error(self, **fields):
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            run_request,
        )

        try:
            run_request(DesignRequest(task="knockdown",
                                      assay="cell_culture_knockdown",
                                      use_signal=False, **fields))
        except DesignRequestError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - the point is which type
            return f"{type(exc).__name__}: {exc}"
        return ""

    def test_a_whitespace_paste_says_no_sequence_was_supplied(self):
        for text in self.WHITESPACE:
            with self.subTest(text=repr(text)):
                self.assertIn("No sequence supplied", self._error(sequence_text=text))

    def test_the_same_for_a_whitespace_fasta_box(self):
        """The upload-as-text door, guarded separately."""
        for text in self.WHITESPACE:
            with self.subTest(text=repr(text)):
                self.assertIn("No sequence supplied", self._error(fasta_text=text))

    def test_the_message_says_what_to_do_next(self):
        """A refusal a user cannot act on is worse than the wrong answer --
        this package's own words, in `services/ingest.py`."""
        message = self._error(sequence_text="   ")
        self.assertIn("Paste a sequence", message)
        self.assertIn("upload a FASTA file", message)
        self.assertIn("built-in target", message)

    def test_a_real_sequence_still_gets_through(self):
        """The guard must not swallow input that only looks empty."""
        target = (
            "GGACAUCGAAUCGCGGAACUUGGCAUCAAGCUAGCUAGGCAUCGAUCGAUCAGCUAGCAUCGAUCGGAUCC"
            "AUGGAUUCCAGGCACUUCUCGAUCAUAGGACUCGGUAAUGCCAUGAGUCGAUGGACUUCAGUGCCAUCGAU"
        )
        self.assertEqual(self._error(sequence_text=f"  {target}  "), "")


class TestTheSeedTaperKeepsItsShape(unittest.TestCase):
    """TERMINAL_WEIGHT is a calibration, and calibrations move quietly.

    Raising it from 0.25 to 0.30 changes every position-weighted penalty --
    a 28-mer with three mismatches goes from 2.0663 to 2.1286 -- which feeds
    the discrimination margin and the activity model's penalty term, and so
    the ranking. The whole suite stayed green.

    Pinning the endpoints alone would not catch it: the taper is *defined* as
    reaching TERMINAL_WEIGHT at each terminus, so an assertion written in
    terms of the constant holds whatever the constant becomes. That is the
    same identity trap as CROSSOVER_BASES. These use literals.
    """

    def test_a_mismatch_inside_the_seed_carries_full_weight(self):
        for length in (22, 28, 30):
            with self.subTest(length=length):
                self.assertEqual(seed_weight(length // 2, length), 1.0)

    def test_the_flanks_taper_to_the_documented_floor(self):
        """Both flanks bottom out at 0.25, which is what own-width buys."""
        lo, hi = SEED_SPAN
        for frac, flank in ((0.0, "5'"), (1.0, "3'")):
            with self.subTest(flank=flank):
                # Position that lands exactly on the terminus in frac space.
                length = 1000
                pos = frac * length + 0.5
                self.assertAlmostEqual(seed_weight(pos, length), 0.25, places=6)

    def test_the_first_and_last_positions_are_not_equally_weighted(self):
        """The taper is symmetric in its endpoints, not in its positions.

        SEED_SPAN is not centred -- the flanks are 0.35 and 0.25 wide -- so
        the first and last positions sit the same distance in from each end
        and are divided by different widths. The comment in seed_weight
        claimed they were penalized exactly alike, which they are not.
        """
        first, last = seed_weight(1, 28), seed_weight(28, 28)
        self.assertNotEqual(first, last)
        self.assertAlmostEqual(first, 0.288265, places=6)
        self.assertAlmostEqual(last, 0.303571, places=6)


class TestAbundanceHalfSaturatesWhereItSaysItDoes(unittest.TestCase):
    """SATURATION_K is the half-saturation point, in TPM-like units.

    Shifting it 50 -> 60 moves every abundance factor -- 0.500 to 0.455 at
    50 TPM -- which reweights every expression-based hazard, and nothing
    failed. The literal is the point: `abundance_factor(SATURATION_K)` is
    0.5 whatever SATURATION_K is.
    """

    def test_fifty_tpm_is_the_half_way_point(self):
        self.assertAlmostEqual(abundance_factor(50.0), 0.5, places=12)

    def test_the_shape_is_saturating_either_side_of_it(self):
        self.assertAlmostEqual(abundance_factor(0.0), 0.0, places=12)
        self.assertLess(abundance_factor(5.0), 0.1)
        self.assertGreater(abundance_factor(500.0), 0.9)


class TestTheHomopolymerLimitFiresAtItsStatedLength(unittest.TestCase):
    """MAX_HOMOPOLYMER is 4, and the existing test used a run of six.

    Six is flagged whether the limit is 4 or 5, so relaxing the limit by one
    -- which reads like an ordinary loosening -- passed the whole suite while
    letting a five-base run through to an ordered oligo. The boundary tested
    away from the boundary, again.
    """

    def test_a_run_of_four_is_allowed(self):
        cand = Candidate("g", "GCAUUGCUGGGGCUGAAGUCAU", "t1", 0)
        self.assertNotIn(QCFlag.HOMOPOLYMER_RUN,
                         qc_candidate(cand, make_spec()).flags)

    def test_a_run_of_five_is_flagged(self):
        cand = Candidate("g", "GCAUUGCUGGGGGCUGAAGUCA", "t1", 0)
        self.assertIn(QCFlag.HOMOPOLYMER_RUN,
                      qc_candidate(cand, make_spec()).flags)


class TestADisfavouredPfsCostsWhatItSaysItCosts(unittest.TestCase):
    """DISFAVOURED_PFS_FACTOR is a bias, and its size is the claim.

    Softening it 0.55 -> 0.60 moves the pfs_preference metric -- one of the
    six named metrics, and an input to the ranking -- for every candidate
    with a disfavoured flanking base, with the suite green. Only LwaCas13a
    publishes a preferred set (3' A/U/C), so exactly one flank is reachable.
    """

    def _policy(self):
        return EFFECTORS[Effector.LWA_CAS13A].pfs

    def test_a_preferred_flank_costs_nothing(self):
        self.assertEqual(self._policy().preference(None, "A"), 1.0)

    def test_a_disfavoured_flank_costs_the_documented_factor(self):
        self.assertAlmostEqual(self._policy().preference(None, "G"), 0.55,
                               places=12)

    def test_an_unread_flank_is_not_a_disfavoured_one(self):
        """Absence of evidence is not a disfavoured base -- the docstring."""
        for flank in (None, "N"):
            with self.subTest(flank=flank):
                self.assertEqual(self._policy().preference(None, flank), 1.0)


class TestEveryMutationStillHasSomethingToMutate(unittest.TestCase):
    """An anchor that no longer matches is a mutation that proves nothing.

    The sweep applies each mutation by exact string replacement. When the code
    it names is edited, the anchor stops matching and that mutation silently
    stops being evidence -- it is not caught and it is not a hole, it simply
    does not run. The instrument reports this at the end ("anchor no longer
    present -- has the code moved?") and is careful to say "Nothing above is
    evidence about what they guard", but only after a full sweep, and the
    documents in the meantime say "N mutations, N caught".

    That happened: rewriting the universe gate and the background loop left
    two anchors stale, so 104 were listed and 102 applied while README and the
    landing page both claimed 104 caught.

    It is the same shape as the defect this whole module exists for -- the
    sweep once could not report a hole at all, so "0 survived" was true by
    construction rather than by measurement. An unmatched anchor is that in
    miniature, one mutation at a time, and it is checkable in milliseconds
    instead of at the end of a long run.
    """

    @staticmethod
    def _mutations():
        import sys

        root = pathlib.Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(root / "examples"))
        import verify_mutation_coverage

        return verify_mutation_coverage.MUTATIONS

    def test_every_anchor_appears_exactly_once_in_the_tree(self):
        root = pathlib.Path(__file__).resolve().parent.parent
        for name, relative, old, _new, _tests in self._mutations():
            with self.subTest(mutation=name):
                source = (root / relative).read_text(encoding="utf-8")
                self.assertEqual(
                    source.count(old), 1,
                    f"{name}: its anchor appears {source.count(old)} times in "
                    f"{relative}, so the sweep cannot apply it and it is "
                    f"evidence about nothing",
                )

    def test_every_replacement_actually_changes_the_source(self):
        """A no-op mutation is caught by nothing and is not a hole either."""
        for name, relative, old, new, _tests in self._mutations():
            with self.subTest(mutation=name):
                self.assertNotEqual(old, new, name)

    def test_every_screening_hint_names_a_test_file_that_exists(self):
        """The sweep's own note: a screening module that is not there is worse
        than a wrong one.

        `pytest tests/nope.py` exits 4, and the sweep reads any non-zero exit
        as "the suite went red", so a stale hint would report the mutation
        caught without it ever having been judged. The sweep guards this at
        run time; checking it here means a rename is caught by the rename's
        own commit.
        """
        root = pathlib.Path(__file__).resolve().parent
        for name, _relative, _old, _new, hints in self._mutations():
            for hint in hints:
                with self.subTest(mutation=name, hint=hint):
                    self.assertTrue(
                        (root / hint).exists(),
                        f"{name} screens against {hint}, which does not exist",
                    )

    def test_there_are_mutations_and_hints_to_check(self):
        """Or every assertion above is true of an empty list."""
        mutations = self._mutations()
        self.assertGreater(len(mutations), 0)
        self.assertGreater(sum(len(hints) for *_, hints in mutations), 0)


if __name__ == "__main__":
    unittest.main()
