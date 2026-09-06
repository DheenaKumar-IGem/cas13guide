"""The demo has to reach the end of its own pipeline.

``cas13guide demo`` is the command ``--help`` sends a new reader to, and it ran
green for a long time while showing a tool that certified nothing: 0 PASS,
81 REVIEW, every one of them ``calibration_unsupported``, and therefore -- the
panel being PASS-only -- no panel, no controls, and a shortfall in place of a
recommendation. The README quoted that shortfall as the demo's transcript.

Nothing caught it, and the suite did run the demo. It asserted exit 0, that the
section headings appeared, and that the three verdict counts summed above zero.
All three hold for a demo that demonstrates nothing: ``--- panel ---`` prints
its heading whether or not a panel follows it, and 0 + 81 + 27 > 0.

The cause was in the demo's own fixture rather than the pipeline. Its stand-in
calibration was a copy of the design path's and had drifted from it, omitting
the ``group=`` label. With no group labels a stratum has zero independent
units, ``groups_ok`` is False, and ``calibration.py`` marks every interval
out-of-domain -- correctly. The guard fired on data that could not satisfy it.

That copy had drifted twice before, each time losing something the design path
had gained: the ``synthetic`` flag, so the demo dropped the CRITICAL that
qualifies every number it prints, and the provenance, which the ledger recorded
as "unstated". A calibration missing a field still answers, so nothing raised
on any of the three occasions. There is now one builder and two thin callers,
and the last class here is about keeping it that way.
"""

import json
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

README = Path(__file__).resolve().parent.parent / "README.md"

from cas13guide.contract.enums import (
    AssayType, Effector, TargetClass, Task, Universe,
)
from cas13guide.contract.spec import (
    ContextScope, EffectorAssayProfile, ResolvedDesignSpec, TargetContract,
    TargetSpec,
)


def _spec() -> ResolvedDesignSpec:
    return ResolvedDesignSpec(
        contract=TargetContract(
            task=Task.KNOCKDOWN,
            targets=(TargetSpec(target_id="t", sequence="ACGU" * 10,
                                target_class=TargetClass.REQUIRED),),
        ),
        profile=EffectorAssayProfile(effector=Effector.RFX_CAS13D,
                                     assay=AssayType.CELL_CULTURE_KNOCKDOWN),
        context=ContextScope(organism="H", genome_build="G",
                             annotation_version="v",
                             declared_universes=frozenset({Universe.CODING})),
    )


class TestTheDemoReachesAPanel(unittest.TestCase):
    """Run it and read the report it writes, rather than its headings."""

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        home = Path(cls._directory.name)
        cls.result = subprocess.run(
            [sys.executable, "-m", "cas13guide.cli", "demo"],
            cwd=home, capture_output=True, text=True,
        )
        cls.report = json.loads(
            (home / "report.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_it_exits_cleanly(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[-1500:])

    def test_something_passes_the_gate(self):
        """The assertion whose absence let this live for as long as it did."""
        counts = self.report["verdict_counts"]
        self.assertGreater(
            counts.get("PASS", 0), 0,
            f"the demo ranked guides and certified none of them: {counts}",
        )

    def test_no_guide_is_held_back_by_the_calibration_itself(self):
        """``calibration_unsupported`` on every survivor is the signature.

        It is a real reason and the demo must be able to raise it; what it must
        not do is raise it on every guide because its own fixture cannot
        satisfy the check.
        """
        ranked = self.report["ranked"]
        self.assertTrue(ranked, "nothing ranked, so this proves nothing")
        blocked = [row for row in ranked
                   if "calibration_unsupported" in json.dumps(row)]
        self.assertNotEqual(
            len(blocked), len(ranked),
            "every ranked guide cites calibration_unsupported, which is what "
            "a self-disabling calibration looks like from the outside",
        )

    def test_it_recommends_a_panel(self):
        self.assertTrue(self.report["panel"].get("guides"),
                        "the panel is empty, so the demo recommends nothing")

    def test_the_panel_gets_controls_and_a_plan(self):
        """The two things downstream of a panel; both vanish with it."""
        self.assertTrue(self.report["controls"])
        self.assertTrue(self.report["validation_plan"])

    def test_it_still_says_its_bounds_are_synthetic(self):
        """Reaching PASS must not quiet the caveat that qualifies it."""
        text = json.dumps(self.report["limitations"])
        self.assertIn("SYNTHETIC", text)
        self.assertIn("exercisable rather than earned", text)

    def test_the_readme_states_the_run_this_actually_produces(self):
        """The README describes this run in numbers. Compare them to it.

        A stated number with no procedure behind it is how the defect above
        lasted: the file described a run, the run changed, and nothing put the
        two side by side. Checked in one direction only -- the README against
        the behaviour, never the reverse.
        """
        stated = re.search(
            r"The documented command reaches `PASS (\d+) / REVIEW (\d+) / "
            r"FAIL (\d+)`, a (\d+)-guide panel at ([\d.]+) coverage",
            re.sub(r"\s+", " ", README.read_text(encoding="utf-8")))
        self.assertIsNotNone(stated, "the README no longer states them")
        counts = self.report["verdict_counts"]
        self.assertEqual(
            [int(group) for group in stated.groups()[:3]],
            [counts.get(verdict, 0) for verdict in ("PASS", "REVIEW", "FAIL")],
        )
        panel = self.report["panel"]
        self.assertEqual(int(stated.group(4)), len(panel["guides"]))
        self.assertAlmostEqual(
            float(stated.group(5)),
            round(max(panel["coverage"].values()), 3), places=3)

    def test_it_records_what_generated_that_calibration(self):
        records = self.report["provenance"]["records"]
        source = next(record["detail"]["source"] for record in records
                      if record["kind"] == "calibration")
        self.assertEqual(source["kind"], "synthetic")
        for field in ("seed", "n_per_stratum", "noise_sd"):
            with self.subTest(field=field):
                self.assertIn(field, source)


class TestTheDemoCalibrationIsNotSelfDisabling(unittest.TestCase):
    """Asked of the calibration directly, where the defect was."""

    def setUp(self):
        from cas13guide.demo import seed_calibration

        self.spec = _spec()
        self.calibration = seed_calibration(self.spec)

    def _key(self, regime):
        from cas13guide.services.calibration import CalibrationService

        return CalibrationService.stratum_key(
            self.spec.profile.effector.value,
            self.spec.profile.assay.value, regime)

    def test_the_regimes_the_pipeline_asks_for_are_the_ones_seeded(self):
        """Otherwise every lookup falls back and the rest proves nothing."""
        for regime in ("mm0", "mm1-2", "mm3+"):
            with self.subTest(regime=regime):
                self.assertIn(self._key(regime), self.calibration.describe())

    def test_its_intervals_are_in_domain(self):
        for regime in ("mm0", "mm1-2", "mm3+"):
            with self.subTest(regime=regime):
                interval = self.calibration.interval(0.8, self._key(regime), 0.1)
                self.assertFalse(
                    interval.ood,
                    f"out of domain as {interval.stratum!r}: the demo cannot "
                    "reach a PASS while this is true",
                )

    def test_the_strata_carry_group_labels(self):
        """Named directly, because this is the field that went missing.

        ``ood`` above is three conditions or-ed together, so it would still be
        False if group adequacy were dropped from it entirely.
        """
        for regime in ("mm0", "mm1-2", "mm3+"):
            with self.subTest(regime=regime):
                self.assertGreater(
                    self.calibration.describe_groups()[self._key(regime)], 1)


class TestOneBuilderServesBothPaths(unittest.TestCase):
    """The copy drifted three times; this is what notices a fourth."""

    def test_the_demo_and_the_design_path_agree_given_the_same_knobs(self):
        from cas13guide.demo import _SYNTHETIC, seed_calibration
        from cas13guide.services.workflow import build_synthetic_calibration

        spec = _spec()
        mine = seed_calibration(spec)
        theirs = build_synthetic_calibration(
            spec.profile.effector.value, spec.profile.assay.value,
            **_SYNTHETIC)
        self.assertEqual(mine.source, theirs.source)
        self.assertEqual(mine.describe(), theirs.describe())
        self.assertTrue(mine.describe(), "no strata, so this compares nothing")
        self.assertEqual(mine.describe_groups(), theirs.describe_groups())
        for key in mine.describe():
            with self.subTest(key=key):
                self.assertEqual(mine.resolve(key).residuals(),
                                 theirs.resolve(key).residuals())

    def test_the_recorded_seed_is_enough_to_rebuild_it(self):
        """The demo drew its calibration from a module-level RNG shared with
        its sequence generation, so the seed its ledger recorded reproduced
        that calibration only together with every draw that came before it."""
        from cas13guide.services.workflow import build_synthetic_calibration

        spec = _spec()
        knobs = dict(seed=11, n_per_stratum=40, noise_sd=0.13)
        first = build_synthetic_calibration(
            spec.profile.effector.value, spec.profile.assay.value, **knobs)
        random.random()  # ambient state moves; the calibration must not
        second = build_synthetic_calibration(
            spec.profile.effector.value, spec.profile.assay.value, **knobs)
        self.assertTrue(first.describe())
        for key in first.describe():
            with self.subTest(key=key):
                self.assertEqual(first.resolve(key).residuals(),
                                 second.resolve(key).residuals())

    def test_a_different_seed_gives_a_different_calibration(self):
        """Or the assertion above holds for a builder that ignores its seed."""
        from cas13guide.services.workflow import build_synthetic_calibration

        spec = _spec()
        residuals = []
        for seed in (11, 12):
            calibration = build_synthetic_calibration(
                spec.profile.effector.value, spec.profile.assay.value,
                seed=seed, n_per_stratum=40, noise_sd=0.13)
            key = sorted(calibration.describe())[0]
            residuals.append(calibration.resolve(key).residuals())
        self.assertNotEqual(residuals[0], residuals[1])


if __name__ == "__main__":
    unittest.main()
