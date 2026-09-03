"""A twelfth pass, over the code eleven rounds barely touched.

Round eleven's critic named what its dimensions never reached: the learning
half -- ingest, training, acceptance, benchmark, calibration, features, about
1900 lines behind the `train` and `check` subcommands -- and stages B and F.

Three confirmed of eleven verified. Two of the three are the class the critic
said this whole method is structurally unable to find: a defect where every
surface agrees and all of them are wrong. A duplicate header inverts every
training label and *no printed number changes*, because rank normalisation is
invariant under inversion. Both runs print "dataset ACCEPTED". Only the fitted
weights differ, and they differ in sign.
"""

import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

from helpers import *  # noqa: F401,F403

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _screen(path: pathlib.Path, header: str, rows: list[str]) -> pathlib.Path:
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


class TestADuplicateColumnIsRefusedRatherThanGuessed(unittest.TestCase):
    """`csv.DictReader` keeps the LAST column of a repeated name.

    The required-column check tested membership only, so a screen carrying two
    columns called `activity` -- an ordinary artefact of a join or a merged
    replicate export -- was read from whichever came second, with no warning.

    That is the failure this module's docstring calls fatal: "activity reported
    as log2 fold change, where more negative means better knockdown --
    silently training on that inverts the model". Measured on a 672-row screen
    whose stray second `activity` column held the negated value, every fitted
    weight flipped sign while all six acceptance numbers stayed identical, and
    both runs printed "dataset ACCEPTED".

    On the grouping column it is worse than an inversion: grouping by a
    duplicate that happens to hold a plate id puts the same transcript on both
    sides of the split, which is the leak that column exists to prevent.

    Refused rather than resolved, for the reason stated elsewhere in the same
    module about a different ambiguity: the wrong resolution inverts every
    label, so there is no safe default.
    """

    def _rows(self, n: int = 12) -> list[str]:
        spacer = "ACGU" * 6
        return [f"{spacer},{(i % 10) * 0.1:.2f},T{i % 4},{9 - (i % 10)}"
                for i in range(n)]

    def test_a_duplicate_activity_column_is_an_error(self):
        from cas13guide.services.ingest import load_screen

        with tempfile.TemporaryDirectory() as directory:
            path = _screen(pathlib.Path(directory) / "dup.csv",
                           "spacer,activity,transcript,activity", self._rows())
            examples, report = load_screen(path)

        self.assertEqual(examples, [])
        self.assertTrue(report.errors)
        self.assertIn("appears 2 times", report.errors[0])
        self.assertIn("activity", report.errors[0])

    def test_a_duplicate_grouping_column_is_an_error(self):
        """The split-leak half: grouping by a plate id instead of a transcript
        puts the same transcript on both sides."""
        from cas13guide.services.ingest import load_screen

        spacer = "ACGU" * 6
        rows = [f"{spacer},{(i % 10) * 0.1:.2f},T{i},PLATE{i % 3}" for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            path = _screen(pathlib.Path(directory) / "dupgrp.csv",
                           "spacer,activity,transcript,transcript", rows)
            examples, report = load_screen(path)

        self.assertEqual(examples, [])
        self.assertTrue(any("appears 2 times" in e for e in report.errors))

    def test_a_screen_without_duplicates_still_loads(self):
        """The refusal must not cost the ordinary case."""
        from cas13guide.services.ingest import load_screen

        spacer = "ACGU" * 6
        rows = [f"{spacer},{(i % 10) * 0.1:.2f},T{i % 4},GENE" for i in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            path = _screen(pathlib.Path(directory) / "ok.csv",
                           "spacer,activity,transcript,gene", rows)
            examples, report = load_screen(path)

        self.assertEqual(len(examples), 12, report.errors)
        self.assertFalse(report.errors)

    def test_the_message_says_which_column_and_why_it_refuses(self):
        """A refusal a user cannot act on is worse than the wrong answer."""
        from cas13guide.services.ingest import load_screen

        with tempfile.TemporaryDirectory() as directory:
            path = _screen(pathlib.Path(directory) / "dup.csv",
                           "spacer,activity,transcript,activity", self._rows())
            _examples, report = load_screen(path)

        message = report.errors[0]
        self.assertIn("keeps only the last", message)
        self.assertIn("Rename", message)


class TestAMalformedCsvIsALocatedRefusal(unittest.TestCase):
    """A stray double-quote escaped as a traceback and exit 1.

    `csv` consumes to EOF looking for the closing quote, so any file past the
    128 KB field limit raises `_csv.Error`. Only `UnicodeDecodeError` was
    converted to a `DatasetError`, and `cli._train` catches that alone -- so
    the process printed a traceback and exited 1, which is the code `train`
    uses for "loaded, and rejected by the acceptance gate". A malformed file
    and a rejected dataset were indistinguishable to anything reading status.
    """

    def _stray_quote_screen(self, directory: pathlib.Path) -> pathlib.Path:
        spacer = "ACGU" * 6
        rows = []
        for i in range(6000):
            gene = '"IFNB1' if i == 100 else "GENE"
            rows.append(f"{spacer},{(i % 10) * 0.1:.2f},T{i % 12},{gene}")
        return _screen(directory / "quote.csv",
                       "spacer,activity,transcript,gene", rows)

    def test_it_raises_the_error_the_caller_handles(self):
        from cas13guide.services.ingest import DatasetError, load_screen

        with tempfile.TemporaryDirectory() as directory:
            path = self._stray_quote_screen(pathlib.Path(directory))
            with self.assertRaises(DatasetError) as caught:
                load_screen(path)
        self.assertIn("not valid CSV", str(caught.exception))

    def test_the_message_warns_the_position_may_mislead(self):
        """The reported position is where the parser gave up, which can be far
        below the line that is actually wrong. Saying so saves a hunt."""
        from cas13guide.services.ingest import DatasetError, load_screen

        with tempfile.TemporaryDirectory() as directory:
            path = self._stray_quote_screen(pathlib.Path(directory))
            with self.assertRaises(DatasetError) as caught:
                load_screen(path)
        self.assertIn("double-quote", str(caught.exception))

    def test_the_train_subcommand_exits_two(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._stray_quote_screen(pathlib.Path(directory))
            result = subprocess.run(
                [sys.executable, "-m", "cas13guide.cli", "train", str(path)],
                cwd=ROOT, capture_output=True, text=True,
            )
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.returncode, 2, result.stderr[-600:])


class TestTheSelfFoldingWarningStatesWhatWasMeasured(unittest.TestCase):
    """The WARNING restated a measured fact as a different, false one.

    `b_qc`'s own docstring is right: "only 21% of its flags are sequences that
    really do fold -- four in five are false", where "fold" means reaching
    -8 kcal/mol. The user-facing limitation paraphrased that as "four in five
    of the guides it does reject fold no more tightly than average".

    Those are not the same claim and the second is false. Measured at the
    sample and seed b_qc itself cites -- 60,000 random 22-mers, seed 12345 --
    the population mean MFE is -1.861 kcal/mol while the flagged spacers
    average -5.564, and only 9.7% of flagged guides fold no more tightly than
    that mean. The proxy is over-strict, not arbitrary, and the warning now
    says which.

    This limitation reaches stdout and the HTML report on every run without
    ViennaRNA, which is the default install.
    """

    def _text(self) -> str:
        from cas13guide.services.limitations import limitations_for_run
        from cas13guide.services.registry import (
            HeuristicCas13Model, ModelRegistry, NullAccessibilityModel,
        )
        from cas13guide.services.provenance import ProvenanceLedger

        ledger = ProvenanceLedger("test")
        ledger.record("model", "qc_self_fold", "qc/stem-proxy/v1",
                      criterion="longest self-complement > 6 nt (~35% recall)")
        found = limitations_for_run(
            make_spec(),
            registry=ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel()),
            provenance=ledger.to_dict(),
        )
        matching = [item for item in found if item.id == "self-folding-is-a-proxy"]
        self.assertTrue(matching, "fixture must trigger the proxy warning")
        return matching[0].text

    def test_it_does_not_claim_the_flags_are_average(self):
        """The false paraphrase. Flagged spacers average -5.56 kcal/mol against
        a population -1.86: they are markedly more structured than average."""
        self.assertNotIn("fold no more tightly than average", self._text())

    def test_it_names_the_criterion_the_proxy_stands_in_for(self):
        """What is true: four in five flags do not reach -8 kcal/mol. Naming
        the threshold is what makes the claim checkable."""
        text = self._text()
        self.assertIn("four in five", text)
        self.assertIn("-8 kcal/mol", text)

    def test_it_still_says_the_proxy_recovers_about_a_third(self):
        """The other measured figure, unchanged and still correct."""
        self.assertIn("third of the guides a real fold rejects", self._text())

    def test_the_two_places_that_state_this_agree(self):
        """b_qc's docstring and the user-facing warning describe one
        measurement. Two accounts of one fact is how this drifted."""
        import inspect

        from cas13guide.pipeline import b_qc

        source = inspect.getsource(b_qc)
        self.assertIn("four in five are", source)
        self.assertIn("four in five", self._text())


if __name__ == "__main__":
    unittest.main()
