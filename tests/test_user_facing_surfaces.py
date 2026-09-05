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


if __name__ == "__main__":
    unittest.main()
