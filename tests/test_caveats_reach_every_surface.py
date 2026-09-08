"""The caveats reached the reader and not the machine, and not the demo at all.

This package's argument is that it says what it could not establish. Four
CRITICAL limitations apply to an ordinary run: the calibration is synthetic so
every PASS is exercisable rather than earned, the activity model is a saturating
heuristic, collateral is not predicted, and off-targets were searched only in
what you loaded.

Two surfaces did not carry them.

``report.json`` -- the machine-readable artefact, the one a pipeline consumes --
had no limitations key at all. A caller reading it saw PASS verdicts and
calibrated-looking intervals with nothing to qualify either. That is the one
surface where a caveat cannot be inferred from tone or layout, and it was the
only surface without them.

``cas13guide demo`` printed none of them anywhere: not the panel's caveats, not
the model's. A grep of its whole output for "critical", "warning" or
"limitation" returned zero. It is the first command anybody runs and the one the
README quotes.

Found by sweeping every dataclass field for values that are constructed and read
by nothing, which is what turned up ``DesignOutcome.limitations``: populated
correctly, and consumed only by tests.
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


class TestTheDemoSaysWhatItDoesNotEstablish(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            cls.result = subprocess.run(
                [sys.executable, "-m", "cas13guide.cli", "demo"],
                cwd=directory, capture_output=True, text=True,
            )
            report = Path(directory) / "report.json"
            cls.report = (
                json.loads(report.read_text(encoding="utf-8"))
                if report.exists() else None
            )

    def test_the_demo_still_runs(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[-1500:])

    def test_it_prints_a_section_of_them(self):
        self.assertIn("what this run does not establish", self.result.stdout)

    def test_it_prints_the_critical_ones(self):
        """A demo that lists only the INFO-level caveats would satisfy a test
        for the section's existence and none of its purpose."""
        self.assertIn("CRITICAL:", self.result.stdout)
        self.assertGreaterEqual(self.result.stdout.count("CRITICAL:"), 2)

    def test_it_names_the_two_that_qualify_every_number_on_the_page(self):
        for phrase in ("heuristic", "Collateral"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.result.stdout)

    def test_the_json_it_writes_carries_them_too(self):
        self.assertIsNotNone(self.report, "the demo wrote no report.json")
        self.assertIn("limitations", self.report)
        self.assertTrue(self.report["limitations"])

    def test_the_json_carries_them_structured_not_rendered(self):
        """A caller switching on severity should not have to parse "CRITICAL: "
        off the front of a sentence."""
        for item in self.report["limitations"]:
            with self.subTest(limitation=item.get("id")):
                self.assertIn("id", item)
                self.assertIn("severity", item)
                self.assertIn("text", item)
                self.assertNotIn("CRITICAL:", item["text"])

    def test_every_severity_is_a_real_one(self):
        from cas13guide.services.limitations import Severity

        known = {level.value for level in Severity}
        severities = {item["severity"] for item in self.report["limitations"]}
        self.assertTrue(severities)
        self.assertLessEqual(severities, known)

    def test_at_least_one_is_critical(self):
        self.assertIn(
            "critical",
            {item["severity"] for item in self.report["limitations"]},
        )


class TestTheReportDictCarriesThem(unittest.TestCase):
    """The library and web paths, which go through ``run_request``."""

    @classmethod
    def setUpClass(cls):
        import random

        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(9)
        sequence = "".join(rng.choice("ACGU") for _ in range(300))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.fa"
            path.write_text(f">t1\n{sequence}\n", encoding="utf-8")
            cls.outcome = run_request(
                DesignRequest(fasta_path=str(path), task="knockdown",
                              assay="cell_culture_knockdown",
                              effector="RfxCas13d", use_structure=False,
                              use_signal=False)
            )

    def test_the_report_dict_has_them(self):
        self.assertIn("limitations", self.outcome.report)
        self.assertTrue(self.outcome.report["limitations"])

    def test_they_are_the_same_ones_the_outcome_carries(self):
        """``DesignOutcome.limitations`` was populated correctly all along and
        read only by tests. The two must not be able to disagree."""
        from_report = [item["id"] for item in self.outcome.report["limitations"]]
        from_outcome = [item.id for item in self.outcome.limitations]
        self.assertEqual(from_report, from_outcome)

    def test_the_report_survives_json_serialisation(self):
        """It is the machine-readable surface or it is nothing."""
        blob = json.dumps(self.outcome.report, default=str)
        self.assertIn("limitations", json.loads(blob))

    def test_a_critical_limitation_reaches_it(self):
        severities = {item["severity"] for item in self.outcome.report["limitations"]}
        self.assertIn("critical", severities)


class TestEveryShippedExampleSaysWhatItCannotEstablish(unittest.TestCase):
    """Running to completion is not the same as demonstrating something.

    `examples/` is documentation that executes, and two of the three scripts
    printed a ranking, a panel and predicted probabilities with no caveats
    anywhere in their output -- the finding at the top of this file, in the
    siblings it was never re-run against. `demo_diagnostic.py` also prints a
    predicted collateral signal per guide, and its stand-in calibration set
    neither `synthetic` nor `source`, so the CRITICAL saying every interval
    rests on generated data could not fire even had the block been printed.

    A test that only checks the exit code cannot see any of that, and there
    was one: `test_panel_site_consistency.TestExamplesStillRun` runs
    `demo_diagnostic.py` and asserts it returns 0.
    """

    #: Every script under `examples/` that runs a design and shows its result.
    #: The `verify_*` scripts are instruments rather than showcases; they
    #: report on the package instead of recommending guides from it.
    SHOWCASES = ("demo_knockdown.py", "demo_diagnostic.py", "demo_trained.py")

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent.parent / "examples"
        cls.output = {}
        for name in cls.SHOWCASES:
            script = root / name
            if not script.exists():
                continue
            with tempfile.TemporaryDirectory() as directory:
                # Scratch cwd: demo_knockdown writes report.json beside it.
                cls.output[name] = subprocess.run(
                    [sys.executable, str(script)], cwd=directory,
                    capture_output=True, text=True, timeout=600,
                )

    def test_they_are_all_here_to_be_checked(self):
        """Or a missing script would pass this class by being absent."""
        self.assertEqual(sorted(self.output), sorted(self.SHOWCASES))

    def test_each_one_runs(self):
        for name, result in self.output.items():
            with self.subTest(script=name):
                self.assertEqual(result.returncode, 0, result.stderr[-1500:])

    def test_each_one_prints_the_block(self):
        for name, result in self.output.items():
            with self.subTest(script=name):
                self.assertIn("what this run does not establish", result.stdout)

    def test_each_one_carries_a_critical(self):
        """The block existing is not the same as it saying anything.

        Every one of these runs the built-in heuristic or searches a corpus it
        was handed, so a CRITICAL applies to all three.
        """
        for name, result in self.output.items():
            with self.subTest(script=name):
                self.assertIn("CRITICAL:", result.stdout)

    def test_the_diagnostic_says_its_calibration_is_synthetic(self):
        """Its stand-in set neither `synthetic` nor `source`, so this could
        not fire whatever the run did. It shares the one builder now."""
        self.assertIn("calibrated on SYNTHETIC data",
                      self.output["demo_diagnostic.py"].stdout)

    def test_the_structure_remedy_reaches_a_real_run(self):
        """The helper is asserted next door; this is it arriving in output.

        `demo_knockdown.py` runs the demo, whose registry uses
        NullAccessibilityModel deliberately, so the no-accessibility WARNING
        is raised on every run of it and carries whichever remedy applies to
        the machine the suite is on.
        """
        from cas13guide.services.limitations import _structure_remedy

        self.assertIn(_structure_remedy(),
                      self.output["demo_knockdown.py"].stdout)

    def test_the_trained_one_does_not(self):
        """The other direction, which is what makes the flag mean something.

        `demo_trained.py` fits a model on a screen and calibrates on held-out
        rows of it, so the synthetic CRITICAL must be absent -- otherwise the
        assertion above is satisfied by a caveat that is simply always on.
        """
        stdout = self.output["demo_trained.py"].stdout
        self.assertIn("what this run does not establish", stdout)
        self.assertNotIn("calibrated on SYNTHETIC data", stdout)


class TestTheStructureRemedyMatchesTheSituation(unittest.TestCase):
    """A caveat that tells you to install what you already have.

    On a machine with ViennaRNA 2.7.2 installed, a run with no accessibility
    said `Install ViennaRNA (pip install "cas13guide[structure]")`. Following
    that changes nothing: the demo builds its registry with
    `NullAccessibilityModel` on purpose and `--no-structure` opts out the same
    way. The limitation keys on `accessibility/none`, which is what both a
    missing engine and a declined one look like from where it stands, and it
    named only the first cause.

    Its neighbour `accessibility-floor-inert` exists because a generic warning
    "never connected the two". Same fix, one layer up.
    """

    def _remedy(self, installed):
        from cas13guide.services import accessibility, limitations

        original = accessibility.folding_engine_available
        accessibility.folding_engine_available = lambda: installed
        try:
            return limitations._structure_remedy()
        finally:
            accessibility.folding_engine_available = original

    def test_without_the_engine_it_says_to_install_it(self):
        remedy = self._remedy(False)
        self.assertIn("Install ViennaRNA", remedy)
        self.assertIn("cas13guide[structure]", remedy)

    def test_with_the_engine_it_does_not(self):
        remedy = self._remedy(True)
        self.assertNotIn("Install ViennaRNA", remedy)
        self.assertIn("declined to fold", remedy)
        self.assertIn("--no-structure", remedy)

    def test_the_probe_answers_for_this_environment(self):
        """Whichever way, it agrees with whether the import works."""
        from cas13guide.services.accessibility import folding_engine_available

        try:
            import RNA  # noqa: F401

            present = True
        except ImportError:
            present = False
        self.assertEqual(folding_engine_available(), present)

if __name__ == "__main__":
    unittest.main()
