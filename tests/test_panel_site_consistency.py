"""One run must not give two answers about the same question.

`g_output` marks a ranked guide as a distinct site when it is at least one
spacer away from every better-ranked guide. `select_panel` enforced its own
spacing with a hardcoded default of 15 nt, shorter than every ortholog's spacer
(22, 28 and 30 nt). So a panel could contain two guides that the very same
report marked as one site -- and the panel is the half a user actually orders.

That is the same defect already fixed for the ranked list (see
test_decision_order.py and f_decide.distinct_site_ids); it simply survived in
the panel, where it mattered more. These tests tie the two together so the
constants cannot drift apart again.

`test_examples_still_run` is here for an unrelated reason with the same shape:
examples/ is not imported by any test, so removing a key from PRESET_SOURCE
broke demo_diagnostic.py and nothing noticed.
"""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.effectors import EFFECTORS
from cas13guide.pipeline.optimizer import select_panel


class TestPanelSpacingMatchesSiteMarking(unittest.TestCase):
    def test_the_default_is_not_a_bare_constant(self):
        """It must come from the spec, or it cannot track the ortholog."""
        import inspect

        default = inspect.signature(select_panel).parameters["min_spacing"].default
        self.assertIsNone(
            default,
            "a numeric default cannot track spacer_length across orthologs",
        )

    def test_every_ortholog_spacer_is_longer_than_the_old_default(self):
        """Why the old 15 was wrong for all three, not just an edge case.

        If some ortholog had a spacer shorter than 15 this would have been a
        tuning question. None does, so it was always a contradiction.
        """
        for effector, params in EFFECTORS.items():
            with self.subTest(effector=effector):
                self.assertGreater(params.preferred_spacer_length, 15)

    def test_two_guides_inside_one_spacer_cannot_both_be_chosen(self):
        """The actual defect, at the width where it used to slip through.

        20 nt apart is >= the old default of 15 and < a 22 nt RfxCas13d spacer,
        so the old code took both while g_output marked them as one site.
        """
        from test_optimizer import guide

        spec = make_spec()
        self.assertEqual(spec.spacer_length, 22)
        pool = [
            guide("near_a", {"t1": 0.90}, start=100),
            guide("near_b", {"t1": 0.88}, start=120),  # 20 nt away: overlapping
        ]
        panel = select_panel(pool, spec, size=2)
        self.assertEqual(panel.size, 1, "took two shifted copies of one site")

    def test_genuinely_separate_sites_are_still_both_chosen(self):
        """Both directions: the constraint must not simply block everything."""
        from test_optimizer import guide

        spec = make_spec()
        pool = [
            guide("far_a", {"t1": 0.90}, start=100),
            guide("far_b", {"t1": 0.88}, start=400),
        ]
        self.assertEqual(select_panel(pool, spec, size=2).size, 2)

    def test_an_explicit_value_still_wins(self):
        """The default is a floor, not a lock -- widening must stay possible."""
        from test_optimizer import guide

        spec = make_spec()
        pool = [
            guide("a", {"t1": 0.90}, start=100),
            guide("b", {"t1": 0.88}, start=400),
        ]
        self.assertEqual(select_panel(pool, spec, size=2, min_spacing=1000).size, 1)


class TestExamplesStillRun(unittest.TestCase):
    """examples/ is documentation that executes, so it can rot silently.

    demo_diagnostic.py read PRESET_SOURCE['release']. That key was removed when
    the invented provenance came out, and the demo has raised KeyError ever
    since -- through a full green suite, because nothing imports examples/.
    """

    def _run(self, name):
        import pathlib
        import subprocess
        import sys

        path = pathlib.Path(__file__).resolve().parent.parent / "examples" / name
        if not path.exists():
            self.skipTest(f"{name} not present")
        return subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, timeout=600
        )

    def test_demo_diagnostic_runs_to_completion(self):
        result = self._run("demo_diagnostic.py")
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_it_does_not_reference_removed_provenance_keys(self):
        """The specific failure: reading a key that no longer exists."""
        import pathlib

        from cas13guide.contract.presets import PRESET_SOURCE

        root = pathlib.Path(__file__).resolve().parent.parent / "examples"
        if not root.exists():
            self.skipTest("no examples directory")
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for key in ("release", "sha256", "retrieved"):
                with self.subTest(file=path.name, key=key):
                    self.assertNotIn(
                        f"PRESET_SOURCE['{key}']",
                        text,
                        f"{path.name} reads a key not in PRESET_SOURCE "
                        f"({sorted(PRESET_SOURCE)})",
                    )


class TestPanelCoverageStatesItsAssumption(unittest.TestCase):
    """The noisy-OR reads as a probability, and it is not one.

    Panel.coverage is 1 - prod(1 - lcb) over members, which assumes they fail
    independently. They share a transcript and a single activity model whose
    errors are correlated; the spacing constraint only stops them sharing a
    site. That was documented in a docstring, where the people making ordering
    decisions from the number never see it.
    """

    def _ids(self, panel_size):
        from cas13guide.services.limitations import limitations_for_run

        return {l.id for l in limitations_for_run(make_spec(), panel_size=panel_size)}

    WARNING = "panel-coverage-assumes-independence"

    def test_a_multi_guide_panel_states_the_assumption(self):
        self.assertIn(self.WARNING, self._ids(3))

    def test_a_single_guide_panel_does_not(self):
        """Nothing is being combined, so there is no independence claim."""
        self.assertNotIn(self.WARNING, self._ids(1))

    def test_an_empty_panel_does_not(self):
        self.assertNotIn(self.WARNING, self._ids(0))

    def test_it_names_the_number_of_members(self):
        """Generic warnings get skimmed; a specific one is about this run."""
        from cas13guide.services.limitations import limitations_for_run

        text = next(
            l.text for l in limitations_for_run(make_spec(), panel_size=4)
            if l.id == self.WARNING
        )
        self.assertIn("4", text)

    def test_it_does_not_claim_the_number_is_meaningless(self):
        """Ranking aid, not garbage -- the distinction the text has to carry."""
        from cas13guide.services.limitations import limitations_for_run

        text = next(
            l.text for l in limitations_for_run(make_spec(), panel_size=2)
            if l.id == self.WARNING
        ).lower()
        self.assertIn("ranking aid", text)
        self.assertIn("overstates", text)
