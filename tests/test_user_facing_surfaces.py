"""Three surfaces a user meets that nothing exercised.

Found by subprocess-traced branch coverage, which is the only way to see them:
a plain coverage run cannot follow the child processes these tests drive, so
`cli.py` reads 14 unfired guards untraced and 4 traced. The four are real.

Each of these is a case where the code is right and the evidence was missing —
a documented command, a security warning, and the headline output of a
feature — so what they pin is that the surface still exists, not that it works
differently than it did.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def run_module(*args, timeout=180):
    return subprocess.run(
        [sys.executable, "-m", "cas13guide", *args],
        capture_output=True, text=True, timeout=timeout,
    )


class TestTheListPresetsCommandRuns(unittest.TestCase):
    """`cas13guide design --list-presets` is in the README's quickstart.

    Its branch had never executed in any test. A documented command that
    nothing runs is one refactor away from being a documented command that
    does not work, and the failure would reach a user before it reached a
    suite.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = run_module("design", "--list-presets")

    def test_it_exits_cleanly(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[:400])

    def test_it_names_the_requirement_it_serves(self):
        self.assertIn("FR1.3", self.result.stdout)

    def test_it_lists_every_shipped_preset_with_its_accession_and_length(self):
        from cas13guide.contract.presets import get_preset, preset_ids

        ids = list(preset_ids())
        self.assertTrue(ids, "fixture sanity: no presets are shipped")
        for preset_id in ids:
            with self.subTest(preset=preset_id):
                self.assertIn(preset_id, self.result.stdout)
                preset = get_preset(preset_id)
                self.assertIn(preset.accession, self.result.stdout)
                self.assertIn(f"{preset.length:3d} nt", self.result.stdout)

    def test_it_designs_nothing(self):
        """It is a catalogue, not a run: no ranking table, no verdicts."""
        self.assertNotIn("verdicts:", self.result.stdout)


class TestTheServerSaysWhenItIsBoundPastLocalhost(unittest.TestCase):
    """The only thing between `--host 0.0.0.0` and an open server.

    `serve` prints a warning that the development server has no
    authentication and no request limits. The branch that prints it had never
    run in the whole suite, and testing it in place would mean binding a
    socket to every interface on the machine — so the message is a function
    now and this checks the function.
    """

    def test_localhost_is_silent(self):
        from cas13guide.services.webapp import bind_warning

        for host in ("127.0.0.1", "localhost", "::1"):
            with self.subTest(host=host):
                self.assertIsNone(bind_warning(host))

    def test_anything_else_warns(self):
        from cas13guide.services.webapp import bind_warning

        for host in ("0.0.0.0", "10.0.0.5", "example.internal"):
            with self.subTest(host=host):
                warning = bind_warning(host)
                self.assertIsNotNone(warning, "bound past localhost, silently")
                self.assertIn("no authentication", warning)
                self.assertIn("exposes it", warning)

    def test_it_says_what_to_do_about_it(self):
        """A warning with no remedy is a warning a reader cannot act on."""
        from cas13guide.services.webapp import bind_warning

        self.assertIn("put a real server in front of it",
                      bind_warning("0.0.0.0"))


class TestThePanelIntervalReachesTheTerminal(unittest.TestCase):
    """FR6.3's headline number, as a user sees it.

    `test_panel_confidence.py` covers the model thoroughly. What nothing
    covered is the block that prints it: the branch guarding it had never
    executed, so the interval was computed, exported, and never once checked
    as text on the surface a person reads.
    """

    @classmethod
    def setUpClass(cls):
        import random

        rng = random.Random(23)

        def seq(n):
            return "".join(rng.choice("ACGU") for _ in range(n))

        cls._directory = tempfile.TemporaryDirectory()
        home = Path(cls._directory.name)
        eol = chr(10)
        (home / "t.fa").write_text(">TGT1" + eol + seq(300) + eol,
                                   encoding="utf-8")
        rows = []
        for name, biotype in (("RN18S1", "rrna"), ("MT-CO1", "mito"),
                              ("LINC01", "ncrna"), ("PREM1", "pre_mrna")):
            rows += [">" + name + " biotype=" + biotype, seq(300)]
        (home / "bg.fa").write_text(eol.join(rows) + eol, encoding="utf-8")
        cls.result = run_module(
            "design", "--fasta", str(home / "t.fa"),
            "--background", str(home / "bg.fa"),
            "--assay", "cell_culture_knockdown", "--effector", "RfxCas13d",
        )

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_the_fixture_really_recommends_a_panel(self):
        """Or there would be no interval to print and this proves nothing."""
        self.assertEqual(self.result.returncode, 0, self.result.stderr[:400])
        self.assertIn("'PASS':", self.result.stdout)
        self.assertNotIn("'PASS': 0", self.result.stdout)

    def test_the_interval_is_printed_and_named(self):
        self.assertIn("panel detection probability (FR6.3):",
                      self.result.stdout)
        self.assertIn("TGT1", self.result.stdout)

    def test_it_is_printed_as_an_interval_with_both_ends_explained(self):
        """The number is an interval because that is what it is: printing one
        end alone would be the assumption the model refuses to make."""
        self.assertIn("lower = members fail together", self.result.stdout)
        self.assertIn("upper = members fail independently", self.result.stdout)


class TestRobustnessIsNotClaimedWhereNothingWasVaried(unittest.TestCase):
    """A number about a question the run never asked.

    Robustness is the minimum required coverage across every evaluated
    context. Over one context that minimum *is* the coverage, so a run with no
    variants reported a robustness equal to its coverage -- and the HTML
    report's tooltip says "A guide that works on the reference and loses one
    variant shows it here", which nothing here could have shown.

    Found by running `cas13guide serve` and reading the page. The terminal has
    always dropped the column when `--variants` is absent; the HTML surface did
    not, and no caveat on any surface said the question had gone unasked, so a
    reader had nothing to correct the impression with.
    """

    @classmethod
    def setUpClass(cls):
        import random
        import tempfile
        from pathlib import Path

        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(19)

        def sequence(n):
            return "".join(rng.choice("ACGU") for _ in range(n))

        cls._directory = tempfile.TemporaryDirectory()
        home = Path(cls._directory.name)
        eol = chr(10)
        reference = sequence(200)
        (home / "t.fa").write_text(
            ">TGT1 biotype=coding" + eol + reference + eol, encoding="utf-8")
        # An isoform: the same transcript with a substitution every 37 nt, so
        # some windows are hit and others are not.
        variant = list(reference)
        for index in range(0, len(variant), 37):
            variant[index] = rng.choice(
                [b for b in "ACGU" if b != variant[index]])
        (home / "var.fa").write_text(
            ">TGT1_iso2" + eol + "".join(variant) + eol, encoding="utf-8")
        # Every universe a knockdown assay exposes the guide to, or the run is
        # REVIEW `universe_unrepresented` throughout and says so instead.
        rows = []
        for name, biotype in (("RN18S1", "rrna"), ("MT-CO1", "mito"),
                              ("LINC01", "ncrna"), ("PREM1", "pre_mrna"),
                              ("OTHER", "coding")):
            rows += [">" + name + " biotype=" + biotype, sequence(200)]
        (home / "bg.fa").write_text(eol.join(rows) + eol, encoding="utf-8")

        def design(variants):
            return run_request(DesignRequest(
                fasta_path=str(home / "t.fa"),
                background_path=str(home / "bg.fa"),
                assay="cell_culture_knockdown",
                variants_path=str(home / "var.fa") if variants else None,
            )).report

        cls.alone = design(False)
        cls.varied = design(True)

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    @staticmethod
    def _contexts(report):
        return [record["name"] for record in report["provenance"]["records"]
                if record["kind"] == "transcriptome"]

    def test_the_fixture_really_evaluates_one_context_and_two(self):
        """Or neither branch below is the branch it says it is."""
        self.assertEqual(self._contexts(self.alone), ["reference"])
        self.assertEqual(len(self._contexts(self.varied)), 2)

    def test_over_one_context_robustness_is_just_the_coverage(self):
        """The observation itself, so the reason for the rest is on record."""
        for row in self.alone["ranked"]:
            metrics = row["metrics"]
            with self.subTest(guide=row["guide_id"]):
                self.assertAlmostEqual(metrics["robustness"],
                                       metrics["required_coverage"], places=9)

    def test_the_run_says_robustness_went_unevaluated(self):
        found = [item for item in self.alone["limitations"]
                 if item["id"] == "robustness-unevaluated"]
        self.assertEqual(len(found), 1)
        self.assertIn("--variants", found[0]["text"])

    def test_it_sorts_with_the_other_warnings(self):
        """Severity orders the list, so a reader who stops early reads what
        matters. `limitations_for_run` sorts before returning, and appending
        after that sort put this one below the INFO notes in every real run --
        which the ordering test could not see, because its fixture supplies no
        provenance and so never raises this caveat.
        """
        rank = {"critical": 0, "warning": 1, "info": 2}
        found = self.alone["limitations"]
        self.assertIn("robustness-unevaluated",
                      [item["id"] for item in found])
        positions = [rank[item["severity"]] for item in found]
        self.assertEqual(positions, sorted(positions),
                         "limitations are not ordered by severity")

    def test_and_does_not_say_it_when_a_variant_was_supplied(self):
        """Or the caveat is simply always on and says nothing."""
        self.assertFalse([item for item in self.varied["limitations"]
                          if item["id"] == "robustness-unevaluated"])

    def test_the_html_report_drops_the_column_it_cannot_fill(self):
        from cas13guide.services.report_html import render_report

        self.assertNotIn(">Robustness<", render_report(self.alone))

    def test_and_keeps_it_when_there_is_something_to_report(self):
        from cas13guide.services.report_html import render_report

        self.assertIn(">Robustness<", render_report(self.varied))

    def test_the_other_columns_are_not_disturbed(self):
        """Dropping one must not take its neighbours with it."""
        from cas13guide.services.report_html import render_report

        page = render_report(self.alone)
        for heading in ("Coverage", "Worst off-target", "Score", "Verdict"):
            with self.subTest(column=heading):
                self.assertIn(">" + heading + "<", page)


if __name__ == "__main__":
    unittest.main()
