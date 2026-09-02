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


if __name__ == "__main__":
    unittest.main()
