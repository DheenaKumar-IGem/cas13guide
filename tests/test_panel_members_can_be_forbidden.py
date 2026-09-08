"""A multiplex panel was not expressible through any door.

Every user-facing entry point filed record 0 as REQUIRED and records 1..n as
DESIRED. DESIRED counts as ``is_intended``, and ``off_target_hazards`` filters
on ``not is_intended``, so a guide that cross-reacts with another panel member
was credited to ``desired_utility`` -- a maximised objective -- and excluded
from the hazard sum entirely. That is a diagnostic panel read as a co-targeting
problem, and it is the reading that rewards the failure it should catch.

The machinery for the other reading already existed and was unreachable:
FORBIDDEN carries a hazard weight of 5.0, the gate raises
CREDIBLE_FORBIDDEN_HIT above ``forbidden_activity_max``, and
``discrimination_margin`` is a ratio of a calibrated lower bound on the
intended target to an upper bound on the forbidden one. There was no CLI flag,
no web field, and no way to say it in a FASTA header.

``biotype=`` was already parsed out of a record's description so one file could
hold several universes. ``class=`` is the same mechanism.
"""

import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass

#: A biomarker, and one two bases from it at the 3' end. The pair a multiplex
#: panel has to be able to ask about.
TARGET = "AUCACAUUGCCAGGGAUUACCACGCAACCACGACC"
NEAR = "AUCACAUUGCCAGGGAUUACCACGCAACCACGAUU"
UNRELATED = "GGUUCAAGCGAUUCUCCUGCCUCAGCCUCCCGAGU"

EOL = chr(10)


def _write(directory: Path, second_header: str, second: str) -> Path:
    path = directory / "panel.fa"
    path.write_text(
        f">bmk_A{EOL}{TARGET}{EOL}>bmk_B{second_header}{EOL}{second}{EOL}",
        encoding="utf-8")
    return path


def _design(path: Path):
    from cas13guide.services.workflow import DesignRequest, run_request

    return run_request(DesignRequest(
        fasta_path=str(path), effector="RfxCas13d",
        assay="in_vitro_detection", task="detection", spacer_length=30,
    )).report


class TestARecordCanDeclareItsClass(unittest.TestCase):
    """The parse, asked of the helper that does it."""

    def _record(self, description):
        from cas13guide.services.fasta import SequenceRecord

        return SequenceRecord(name="r", sequence=TARGET,
                              description=description)

    def _classify(self, description, default=TargetClass.DESIRED):
        from cas13guide.services.workflow import _record_target_class

        return _record_target_class(self._record(description), default, "--fasta")

    def test_a_header_without_it_keeps_the_positional_default(self):
        """A file with no `class=` anywhere must behave exactly as before."""
        self.assertIs(self._classify(""), TargetClass.DESIRED)
        self.assertIs(self._classify("biotype=ncrna"), TargetClass.DESIRED)
        self.assertIs(self._classify("", TargetClass.REQUIRED),
                      TargetClass.REQUIRED)

    def test_every_class_the_contract_knows_can_be_said(self):
        for target_class in TargetClass:
            with self.subTest(target_class=target_class.value):
                self.assertIs(self._classify(f"class={target_class.value}"),
                              target_class)

    def test_it_reads_alongside_biotype(self):
        """Both tokens are scanned from the same description."""
        self.assertIs(self._classify("biotype=ncrna class=forbidden"),
                      TargetClass.FORBIDDEN)
        self.assertIs(self._classify("class=forbidden biotype=ncrna"),
                      TargetClass.FORBIDDEN)

    def test_case_does_not_matter(self):
        self.assertIs(self._classify("CLASS=Forbidden"), TargetClass.FORBIDDEN)

    def test_an_unknown_class_names_the_ones_that_exist(self):
        from cas13guide.services.workflow import DesignRequestError

        with self.assertRaises(DesignRequestError) as caught:
            self._classify("class=hostile")
        message = str(caught.exception)
        self.assertIn("hostile", message)
        for target_class in TargetClass:
            with self.subTest(target_class=target_class.value):
                self.assertIn(target_class.value, message)


class TestTheClassReachesTheGate(unittest.TestCase):
    """The point of saying it: a forbidden member becomes a hazard.

    Both directions, because a rule that refuses everything is no more useful
    than one that refuses nothing.
    """

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        home = Path(cls._directory.name)
        cls.intended = _design(_write(home, "", NEAR))
        cls.forbidden_near = _design(_write(home, " class=forbidden", NEAR))
        cls.forbidden_far = _design(_write(home, " class=forbidden", UNRELATED))

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    @staticmethod
    def _reasons(report):
        found = set()
        for row in (report.get("rejected") or []):
            found.update(row.get("reasons") or [])
        return found

    def test_the_fixture_produces_candidates_at_all(self):
        """Or every assertion below is about an empty run."""
        self.assertTrue(sum(self.intended["verdict_counts"].values()))

    def test_left_intended_the_cross_reaction_is_credited_not_penalised(self):
        """The behaviour `class=` exists to make sayable otherwise.

        Both records intended: the guide binds the near-identical partner, and
        that hit lands in `desired_utility` -- a maximised objective -- while
        the off-target hazard stays at zero.
        """
        metrics = self.intended["ranked"][0]["metrics"]
        self.assertGreater(metrics["desired_utility"], 0.0)
        self.assertEqual(metrics["worst_off_target"], 0.0)
        self.assertEqual(metrics["cumulative_burden"], 0)

    def test_marked_forbidden_a_near_identical_member_fails_the_gate(self):
        self.assertIn("credible_forbidden_hit",
                      self._reasons(self.forbidden_near))
        self.assertEqual(self.forbidden_near["verdict_counts"].get("PASS", 0), 0)

    def test_an_unrelated_forbidden_member_does_not(self):
        """Otherwise `class=forbidden` is a switch that refuses everything."""
        self.assertNotIn("credible_forbidden_hit",
                         self._reasons(self.forbidden_far))
        counts = self.forbidden_far["verdict_counts"]
        self.assertGreater(counts.get("PASS", 0) + counts.get("REVIEW", 0), 0)

    def test_the_two_runs_differ_only_in_that_header(self):
        """Same sequences, same flags; the verdicts diverge because of one
        word in a FASTA description."""
        self.assertNotEqual(self.intended["verdict_counts"],
                            self.forbidden_near["verdict_counts"])


class TestARunStillNeedsSomethingToHit(unittest.TestCase):
    def test_classing_every_record_away_is_refused_in_words(self):
        from cas13guide.services.workflow import DesignRequestError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "none.fa"
            path.write_text(
                f">a class=forbidden{EOL}{TARGET}{EOL}"
                f">b class=forbidden{EOL}{UNRELATED}{EOL}", encoding="utf-8")
            with self.assertRaises(DesignRequestError) as caught:
                _design(path)
        message = str(caught.exception)
        self.assertIn("class=required", message)
        self.assertIn("class=forbidden", message)


class TestBothDoorsSayTheSyntaxExists(unittest.TestCase):
    """An affordance nobody can find is the defect this fixes, one level up."""

    def test_the_cli_help_names_it(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "cas13guide", "design", "--help"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr[-400:])
        self.assertIn("class=forbidden", result.stdout)

    def test_the_web_form_names_it(self):
        from cas13guide.services.webapp import _form_page

        self.assertIn("class=forbidden", _form_page())


class TestTheRatioIsAskedForAPanelToo(unittest.TestCase):
    """`discrimination_margin` is the better answer, and it was switched off.

    `requires_discrimination` read the task label alone, so an allele contract
    got the ratio and a multiplex detection panel did not -- however its
    records were classed. The two cases are the same shape: targets a base or
    two apart, where thresholding LCB(on) and UCB(off) independently fails
    every guide and only their ratio carries the question.

    Keyed on the contract's shape now, so a contract with no FORBIDDEN target
    is untouched and the metric stays None exactly as before.
    """

    @staticmethod
    def _spec(task, classes):
        from cas13guide.contract.enums import (
            AssayType, Effector, TargetClass, Task, Universe,
        )
        from cas13guide.contract.spec import (
            ContextScope, EffectorAssayProfile, ResolvedDesignSpec,
            TargetContract, TargetSpec,
        )

        targets = tuple(
            TargetSpec(target_id=f"t{index}", sequence="ACGU" * 8,
                       target_class=cls)
            for index, cls in enumerate(classes)
        )
        return ResolvedDesignSpec(
            contract=TargetContract(task=task, targets=targets),
            profile=EffectorAssayProfile(
                effector=Effector.RFX_CAS13D,
                assay=AssayType.IN_VITRO_DETECTION),
            context=ContextScope(organism="H", genome_build="G",
                                 annotation_version="v",
                                 declared_universes=frozenset({Universe.NCRNA})),
        )

    def test_a_detection_panel_with_a_forbidden_member_asks_for_it(self):
        from cas13guide.contract.enums import TargetClass, Task

        spec = self._spec(Task.DETECTION,
                          [TargetClass.REQUIRED, TargetClass.FORBIDDEN])
        self.assertTrue(spec.requires_discrimination)

    def test_a_detection_panel_without_one_does_not(self):
        """The guard that keeps this additive: no FORBIDDEN, no change."""
        from cas13guide.contract.enums import TargetClass, Task

        spec = self._spec(Task.DETECTION,
                          [TargetClass.REQUIRED, TargetClass.DESIRED])
        self.assertFalse(spec.requires_discrimination)

    def test_an_allele_task_still_asks_for_it_on_the_label_alone(self):
        """The original condition, kept: an allele contract qualifies whether
        or not this particular one happens to declare a FORBIDDEN target."""
        from cas13guide.contract.enums import TargetClass, Task

        spec = self._spec(Task.ALLELE, [TargetClass.REQUIRED])
        self.assertTrue(spec.requires_discrimination)

    def test_every_task_qualifies_once_a_record_is_forbidden(self):
        """It is a property of what was declared, not of what it was called."""
        from cas13guide.contract.enums import TargetClass, Task

        for task in Task:
            with self.subTest(task=task.value):
                spec = self._spec(
                    task, [TargetClass.REQUIRED, TargetClass.FORBIDDEN])
                self.assertTrue(spec.requires_discrimination)


class TestTheRatioReachesTheVerdict(unittest.TestCase):
    """Through a real design, because the property that matters is the verdict.

    The class above asserts the flag. This asserts that turning it on changes
    what the run says about a guide that cannot tell two biomarkers apart --
    which is the whole point of computing the ratio.
    """

    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path

        from cas13guide.services.workflow import DesignRequest, run_request

        # Two bases apart, both outside the seed. RfxCas13d is fit at 22 nt,
        # so the run is in-domain and the verdict is about discrimination
        # rather than about an unvalidated spacer length.
        required = "UAACACUGUCUGGUAAAGAUGG"
        near = "UAACACUGUCUGGUAACGAUGU"
        cls._directory = tempfile.TemporaryDirectory()
        home = Path(cls._directory.name)
        eol = chr(10)
        (home / "forbidden.fa").write_text(
            ">miR141 class=required" + eol + required + eol
            + ">miR200a class=forbidden" + eol + near + eol, encoding="utf-8")
        (home / "unclassed.fa").write_text(
            ">miR141" + eol + required + eol + ">miR200a" + eol + near + eol,
            encoding="utf-8")

        def design(name):
            return run_request(DesignRequest(
                fasta_path=str(home / name), effector="RfxCas13d",
                assay="in_vitro_detection", task="detection")).report

        cls.forbidden = design("forbidden.fa")
        cls.unclassed = design("unclassed.fa")

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    @staticmethod
    def _reasons(report):
        return {code for row in (report.get("rejected") or [])
                for code in (row.get("reasons") or [])}

    def test_unclassed_the_cross_reaction_is_still_credited(self):
        """The state this exists to change, kept on record.

        Left unclassed the second biomarker is DESIRED, DESIRED counts as
        `is_intended`, and the hit feeds `desired_utility` -- a maximised
        objective -- while contributing nothing to the hazard.
        """
        ranked = self.unclassed["ranked"]
        self.assertTrue(ranked, "nothing ranked, so this shows nothing")
        metrics = ranked[0]["metrics"]
        self.assertGreater(metrics["desired_utility"], 0.0)
        self.assertEqual(metrics["worst_off_target"], 0.0)
        self.assertIsNone(metrics["discrimination_margin"],
                          "no FORBIDDEN target, so the ratio must stay absent")

    def test_classed_forbidden_the_ratio_is_what_refuses_it(self):
        reasons = self._reasons(self.forbidden)
        self.assertIn("discrimination_margin_insufficient", reasons)

    def test_and_the_hazard_gate_fires_as_well(self):
        """Two independent readings of the same fact, both now reachable."""
        self.assertIn("credible_forbidden_hit", self._reasons(self.forbidden))

    def test_the_two_runs_differ_only_in_those_headers(self):
        """Or the comparison is about something else."""
        import re
        from pathlib import Path

        home = Path(self._directory.name)
        strip = lambda text: re.sub(r" class=\w+", "",  # noqa: E731
                                    text.replace(chr(13), ""))
        self.assertEqual(
            strip((home / "forbidden.fa").read_text(encoding="utf-8")),
            strip((home / "unclassed.fa").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
