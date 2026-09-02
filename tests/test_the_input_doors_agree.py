"""The three ways in must produce the same run.

``_design``'s docstring calls itself "a thin wrapper over
``services.workflow.run_request``: the CLI and the web server share that one
code path so a form POST and a command line cannot diverge". That is an
architectural promise, and the suite checked only half of it -- that each door
builds the right ``DesignRequest`` fields. Nothing checked that the three
requests then produce the same *report*, which is the part a user would notice.

There are three doors and they are deliberately distinct fields, because
folding two of them together is what once let the web form silently drop an
upload whenever the textarea was also filled:

* ``fasta_path``   -- ``--fasta``, a file on disk;
* ``sequence_text`` -- ``--sequence`` or the textarea, pasted text;
* ``fasta_text``   -- a file uploaded through the browser.

Measured: the reports are byte-identical, provenance and spec fingerprint
included. The fingerprint agreeing is the sharper half of that -- it identifies
the design rather than the route taken to state it, so two people who reach the
same design different ways can compare runs.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def _run(**request_fields) -> dict:
    from cas13guide.services.workflow import DesignRequest, run_request

    return run_request(
        DesignRequest(task="knockdown", assay="cell_culture_knockdown",
                      effector="RfxCas13d", use_structure=False,
                      use_signal=False, **request_fields)
    ).report


def _blob(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class TestTheThreeDoorsProduceTheSameRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The second record shares an 80 nt stretch with the first, so its
        # guides actually rank. An unrelated record contributes nothing to the
        # ranking, and the doors would then be compared on a single-record run.
        rng = random.Random(9)
        primary = "".join(rng.choice("ACGU") for _ in range(300))
        second = (
            "".join(rng.choice("ACGU") for _ in range(60))
            + primary[100:180]
            + "".join(rng.choice("ACGU") for _ in range(60))
        )
        cls.text = f">t1\n{primary}\n>t2\n{second}\n"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.fa"
            path.write_text(cls.text, encoding="utf-8")
            cls.from_file = _run(fasta_path=str(path))
        cls.from_paste = _run(sequence_text=cls.text)
        cls.from_upload = _run(fasta_text=cls.text)

    def test_the_fixture_is_a_real_design(self):
        """Two identical empty reports would satisfy every test below."""
        self.assertGreater(len(self.from_file["ranked"]), 20)

    def test_a_pasted_fasta_gives_the_same_report_as_the_file(self):
        self.assertEqual(_blob(self.from_file), _blob(self.from_paste))

    def test_an_uploaded_fasta_gives_the_same_report_as_the_file(self):
        """The door that once dropped its input when another was also filled."""
        self.assertEqual(_blob(self.from_file), _blob(self.from_upload))

    def test_the_fingerprint_identifies_the_design_not_the_route(self):
        """Sharper than report equality: two people who reach the same design
        by different doors can compare runs by fingerprint alone."""
        fingerprints = {
            report["spec_fingerprint"]
            for report in (self.from_file, self.from_paste, self.from_upload)
        }
        self.assertEqual(len(fingerprints), 1, fingerprints)

    def test_every_door_found_both_records(self):
        for label, report in (("file", self.from_file),
                              ("paste", self.from_paste),
                              ("upload", self.from_upload)):
            with self.subTest(door=label):
                self.assertEqual(
                    {row["source_target_id"] for row in report["ranked"]},
                    {"t1", "t2"},
                )


class TestARawPasteIsStillADesign(unittest.TestCase):
    """FR1.1: pasted text need not be FASTA at all."""

    def test_a_headerless_sequence_is_named_and_designed_against(self):
        rng = random.Random(3)
        report = _run(sequence_text="".join(rng.choice("ACGU") for _ in range(300)))

        self.assertTrue(report["ranked"])
        self.assertEqual(
            {row["source_target_id"] for row in report["ranked"]}, {"query"},
            "a headerless paste should become one named record, not zero",
        )


if __name__ == "__main__":
    unittest.main()
