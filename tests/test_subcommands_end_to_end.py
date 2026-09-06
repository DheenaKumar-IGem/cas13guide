"""``check`` and ``train``, driven as commands rather than as functions.

Coverage measurement put ``cli.py`` at 63% once subprocess tracing was turned on
(the 7% it reports without that is an artefact -- several tests drive the CLI
through ``subprocess``, and coverage does not follow a child process by
default). What is left uncovered is almost entirely these two subcommands:
``_check`` and ``_train`` are exercised by nothing in the suite.

That is the gap worth closing rather than the number. The sixth review pass
found three real defects in ``train`` -- a traceback on a directory, a traceback
on a UTF-16 screen, and a mistyped column blamed on the wrong column -- and
found them precisely because nobody had ever run it. Those are pinned at the
``load_screen`` level; this pins the commands, which is where a user meets them,
and where an exit code means something.

Driven out of process on purpose: an exit code and a stream are the contract
these commands have with a shell script, and calling ``main()`` directly tests
neither.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "cas13guide", *args],
        capture_output=True, text=True, timeout=900, cwd=str(cwd or REPO),
    )


def screen_csv(path: Path, rows: int = 800, groups: int = 40) -> Path:
    """A screen with enough transcripts and spread to be trainable.

    At least 300 rows, because the acceptance gate refuses a smaller screen
    before it reaches the criteria this exercises. Spacers vary so the model
    has something to learn, and activity tracks a weak function of GC content
    so the acceptance gate sees real signal rather than noise -- the point is
    to exercise the command, not to smuggle a model past its own gate.

    40 transcripts, not 12. Conformal calibration is measured on a held-out
    fifth of them and needs 8, so a screen needs 38 before it can reach the
    later criteria at all -- this helper claimed to be "enough to be trainable"
    at 12, where the run stops at the second criterion of six.
    """
    import random

    rng = random.Random(5)
    lines = ["spacer,activity,transcript"]
    for index in range(rows):
        spacer = "".join(rng.choice("ACGU") for _ in range(22))
        gc = (spacer.count("G") + spacer.count("C")) / len(spacer)
        activity = min(1.0, max(0.0, gc + rng.gauss(0, 0.08)))
        lines.append(f"{spacer},{activity:.4f},T{index % groups:02d}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestTheCheckCommand(unittest.TestCase):
    """The CI gate. Its exit code is the whole point of it."""

    def test_it_passes_and_says_what_it_gated(self):
        result = run("check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK:", result.stdout)
        # The narrowed claim from the eighth pass: it gates coverage, and
        # reports skill without gating it.
        self.assertIn("conformal coverage", result.stdout)
        self.assertIn("skill is reported, not gated", result.stdout)

    def test_it_prints_the_numbers_it_measured(self):
        result = run("check")
        for field in ("spearman", "mae", "coverage"):
            with self.subTest(field=field):
                self.assertIn(field, result.stdout)

    def test_the_seed_is_honoured(self):
        """Same seed, same numbers -- across processes, so hash randomisation
        is in play."""
        first, second = run("check", "--seed", "3"), run("check", "--seed", "3")
        self.assertEqual(first.stdout, second.stdout)

    def test_a_different_seed_moves_the_measurement(self):
        """Otherwise the flag is accepted and ignored, which is its own bug."""
        self.assertNotEqual(run("check", "--seed", "1").stdout,
                            run("check", "--seed", "2").stdout)

    def test_no_traceback_reaches_the_user(self):
        self.assertNotIn("Traceback", run("check").stderr)

    def test_it_covers_every_regime_the_design_path_stratifies_on(self):
        """One stratum was validated and it was misnamed.

        The generator draws 0 to 4 mismatches and every case was filed under a
        hardcoded `...|mm1-2` -- measured, mm0 73, mm1-2 153, mm3+ 74 -- so
        the printed `coverage[...|mm1-2:0.920]` read as a figure for that
        regime and was marginal coverage over a mixture of all three.

        The pooling was not hiding a failure. Split conformal is marginally
        valid whatever the pool, and pooled and per-regime coverage both hold
        under noise widening with mismatch depth and under a model simply
        wrong past two mismatches. What it did not do was exercise the strata
        the pipeline routes into, so a change to `mismatch_regime` was
        invisible to the gate that exists to catch calibration changes.
        """
        stdout = run("check").stdout
        for regime in ("mm0", "mm1-2", "mm3+"):
            with self.subTest(regime=regime):
                self.assertIn(
                    f"RfxCas13d|cell_culture_knockdown|{regime}:", stdout)

    def test_each_reported_stratum_holds_the_regime_it_names(self):
        """The label, against the data behind it.

        Asked of the generator rather than the printed line, because a name
        being present says nothing about what was filed under it -- which is
        exactly how the old one passed for as long as it did.
        """
        import random

        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.registry import HeuristicCas13Model
        from cas13guide.util.duplex import profile_duplex
        from cas13guide.util.seq import revcomp

        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        rng = random.Random(0)
        model = HeuristicCas13Model()
        seen = set()
        for _ in range(300):
            site = list(revcomp(spacer))
            for _ in range(rng.randint(0, 4)):
                index = rng.randrange(len(site))
                site[index] = rng.choice(
                    [b for b in "AUGC" if b != site[index]])
            site_text = "".join(site)
            depth = profile_duplex(spacer, site_text).n_mismatch
            regime = CalibrationService.mismatch_regime(depth)
            seen.add(regime)
            with self.subTest(depth=depth):
                if depth == 0:
                    self.assertEqual(regime, "mm0")
                elif depth <= 2:
                    self.assertEqual(regime, "mm1-2")
                else:
                    self.assertEqual(regime, "mm3+")
        self.assertEqual(seen, {"mm0", "mm1-2", "mm3+"},
                         "the generator no longer spans all three, so the "
                         "test above could pass on one regime")
        self.assertGreater(model.predict(spacer, profile_duplex(
            spacer, revcomp(spacer)), None), 0.0, "fixture sanity")


class TestTheTrainCommand(unittest.TestCase):
    """Ingest a screen, run the acceptance gate, report."""

    ROWS, GROUPS = 800, 40

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            path = screen_csv(Path(directory) / "screen.csv",
                              rows=cls.ROWS, groups=cls.GROUPS)
            cls.result = run("train", str(path))

    def test_a_good_screen_is_read_and_summarised(self):
        """Read off the fixture rather than transcribed from one run of it.

        These were the literals 360 and 12, which is the screen this helper
        used to build -- one the gate stops at its second criterion, because
        conformal calibration needs 8 of a held-out fifth of the transcripts.
        """
        self.assertIn(f"read {self.ROWS} rows", self.result.stdout)
        self.assertIn(f"{self.GROUPS} transcript group(s)", self.result.stdout)
        self.assertNotIn("Traceback", self.result.stderr)

    def test_it_reports_each_acceptance_criterion(self):
        for name in ("rows", "transcript groups", "grouped CV spearman",
                     "held-out spearman", "calibration transcripts",
                     "conformal coverage"):
            with self.subTest(criterion=name):
                self.assertIn(name, self.result.stdout)

    def test_a_good_screen_is_accepted(self):
        """The helper is named for producing one; nothing checked that it did.

        At 12 transcripts every run of it was REJECTED at the second of six
        criteria, so the four below it -- the skill checks, the calibration
        group count, the conformal coverage the gate exists for -- never ran.
        """
        self.assertIn("dataset ACCEPTED", self.result.stdout)
        self.assertNotIn("[FAIL]", self.result.stdout)

    def test_every_printed_criterion_agrees_with_its_own_numbers(self):
        """The eighth pass found a criterion printing PASS beside a value below
        the threshold on the same line. Checked here at the command, not at the
        function, because the line is what a user reads."""
        import re

        with tempfile.TemporaryDirectory() as directory:
            path = screen_csv(Path(directory) / "screen.csv")
            result = run("train", str(path))
        pattern = re.compile(r"\[(PASS|FAIL)\]\s+(.+?)\s+([-\d.]+) \(need ([-\d.]+)\)")
        rows = pattern.findall(result.stdout)
        self.assertTrue(rows, result.stdout)
        for mark, name, measured, threshold in rows:
            with self.subTest(criterion=name.strip()):
                self.assertEqual(mark == "PASS", float(measured) >= float(threshold))

    def test_a_missing_file_is_a_located_message(self):
        result = run("train", "no_such_screen.csv")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("cannot read dataset", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_directory_is_a_located_message(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run("train", directory)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("cannot read dataset", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_screen_with_no_grouping_column_is_refused_with_the_reason(self):
        """The one failure this module's docstring calls fatal rather than a
        warning."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nogroup.csv"
            path.write_text(
                "spacer,activity\nGCAUUGCUGUGGAGCUGAAGUC,0.5\n", encoding="utf-8"
            )
            result = run("train", str(path))
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("transcript", result.stdout.lower())
        self.assertNotIn("Traceback", result.stderr)

    def test_a_mistyped_column_names_the_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = screen_csv(Path(directory) / "screen.csv")
            result = run("train", str(path), "--spacer-column", "guide")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("guide", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_the_direction_flag_changes_the_outcome(self):
        """``--lower-is-better`` inverts the ranking; on a screen with real
        signal that cannot leave the report identical, or the flag is accepted
        and ignored."""
        with tempfile.TemporaryDirectory() as directory:
            path = screen_csv(Path(directory) / "screen.csv")
            normal = run("train", str(path))
            flipped = run("train", str(path), "--lower-is-better")
        self.assertNotEqual(normal.stdout, flipped.stdout)


class TestTheCommandSurfaceIsConsistent(unittest.TestCase):
    """Every subcommand answers ``--help`` and none of them traceback."""

    SUBCOMMANDS = ("demo", "check", "train", "design", "serve")

    def test_every_subcommand_has_help(self):
        for name in self.SUBCOMMANDS:
            with self.subTest(subcommand=name):
                result = run(name, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_every_subcommand_is_listed_in_the_top_level_help(self):
        listing = run("--help").stdout
        for name in self.SUBCOMMANDS:
            with self.subTest(subcommand=name):
                self.assertIn(name, listing)

    def test_an_unknown_subcommand_is_argparses_error_not_a_traceback(self):
        result = run("nonesuch")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)



class TestRuntimeIsBoundedByTheCandidateBudget(unittest.TestCase):
    """The README quotes runtime figures; this pins the *shape* behind them.

    The requirement-coverage table said "measured at 0.2-3.4 s" until a 6 kb
    transcript -- an ordinary mRNA -- was timed at 7.1 s. The figure was not
    wrong when it was taken; it was taken on targets short enough that folding
    was cheap, and nobody re-measured on a full-length one.

    Wall-clock is machine-dependent and a poor thing to assert, so this pins the
    property the figures rest on instead: runtime tracks the candidate count,
    the candidate count is capped by ``SearchBudget.max_candidates``, and past
    the cap a longer target costs little more. A change that made runtime scale
    with target length again would break this while any absolute threshold
    would either be flaky or unfalsifiable.
    """

    @staticmethod
    def _candidates(length: int) -> int:
        import random

        from cas13guide.contract.enums import TargetClass
        from cas13guide.contract.spec import TargetSpec
        from cas13guide.pipeline.a_candidates import design_candidates
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from helpers import make_spec

        rng = random.Random(length)
        sequence = "".join(rng.choice("ACGU") for _ in range(length))
        spec = make_spec(
            targets=(
                TargetSpec(target_id="t1", sequence=sequence,
                           target_class=TargetClass.REQUIRED),
            )
        )
        return len(design_candidates(spec)), spec.budget.max_candidates

    def test_the_candidate_count_is_capped(self):
        count, budget = self._candidates(40_000)
        self.assertLessEqual(count, budget)

    def test_a_much_longer_target_does_not_cost_proportionally_more(self):
        """The flattening the Runtime table shows, as a property."""
        smaller, _ = self._candidates(12_000)
        larger, _ = self._candidates(40_000)
        self.assertLess(
            larger, smaller * 2,
            "candidate count is scaling with target length again, so runtime will too",
        )

    def test_a_target_under_the_cap_is_tiled_densely(self):
        """The cap must not be paid by targets that do not need it."""
        length = 1_000
        count, _ = self._candidates(length)
        self.assertGreater(count, length * 0.5)

    def test_the_readme_quotes_a_range_that_covers_the_capped_case(self):
        """A figure measured only on short targets is how this went stale."""
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("bounded by the candidate budget", readme)
        self.assertNotIn("measured at 0.2–3.4 s", readme)


class TestOffTargetSitesAreNotFolded(unittest.TestCase):
    """Accessibility was computed for every hit, including off-target ones.

    Every reader of ``PairEvidence.accessibility`` is guarded by
    ``is_intended``: ``intended_accessibility`` and the metric vector both skip
    a non-intended pair, the off-target explorer has no accessibility column,
    and a folded profile is not independent evidence so it never reaches the
    activity model. Computing it for an off-target hit meant a full
    ``pfl_fold_up`` over that locus's whole sequence, thrown away.

    It is the dominant cost of any run with a real background corpus, because
    that is precisely where the off-target hits are: 7.85s to 0.90s of folding
    on a 1.5 kb target against 25 background loci, for a report that is
    byte-identical either way.
    """

    def _design(self, background: bool):
        import random
        import tempfile as tf

        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(3)
        target = "".join(rng.choice("ACGU") for _ in range(900))
        directory = Path(tf.mkdtemp())
        (directory / "t.fa").write_text(f">T1\n{target}\n", encoding="utf-8")
        extra = {}
        if background:
            loci = []
            for index in range(4):
                filler = "".join(rng.choice("ACGU") for _ in range(120))
                loci.append(f">BG{index}\n{filler}{target[100:300]}{filler}")
            (directory / "bg.fa").write_text("\n".join(loci) + "\n", encoding="utf-8")
            extra["background_path"] = str(directory / "bg.fa")
        return run_request(
            DesignRequest(fasta_path=str(directory / "t.fa"), task="knockdown",
                          assay="cell_culture_knockdown", effector="RfxCas13d", **extra)
        )

    def test_an_off_target_pair_carries_no_accessibility(self):
        """Asserted on the evidence, because the report has no column for it --
        which is exactly why computing it was waste."""
        import dataclasses

        from cas13guide.contract.enums import TargetClass
        from cas13guide.pipeline.d_pair_evidence import evidence_for_hit
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.context import StaticExpressionSource
        from cas13guide.services.registry import ModelRegistry
        from cas13guide.services.transcriptome import Locus, Transcriptome
        from cas13guide.contract.enums import Universe

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from helpers import SEQ_A, SEQ_B, make_hit, make_spec  # noqa: F401

        class _Folding:
            """Records every sequence it is asked to fold."""

            version = "accessibility/test/v1"
            is_independent_evidence = False

            def __init__(self):
                self.asked = []

            def accessibility(self, sequence, start, length):
                self.asked.append(sequence)
                return 0.5

        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("L_t1", SEQ_A, Universe.NCRNA, gene="t1", expression=10.0))
        world.add(Locus("L_off", SEQ_B, Universe.CODING, gene="off", expression=10.0))
        model = _Folding()
        from cas13guide.services.registry import HeuristicCas13Model

        registry = ModelRegistry(HeuristicCas13Model(), model)
        spec = make_spec()
        calibration = CalibrationService()
        expression = StaticExpressionSource({})

        intended = evidence_for_hit(
            make_hit("g1", "L_t1", "t1", TargetClass.REQUIRED), "A" * 22, spec,
            registry, calibration, world, expression,
        )
        off_target = evidence_for_hit(
            dataclasses.replace(make_hit("g1", "L_off"), target_class=None),
            "A" * 22, spec, registry, calibration, world, expression,
        )
        self.assertIsNotNone(intended.accessibility)
        self.assertIsNone(off_target.accessibility)
        self.assertEqual(model.asked, [SEQ_A], "an off-target locus was folded")

    def test_the_intended_site_still_has_one(self):
        """The saving must not have cost the value that is actually used."""
        outcome = self._design(background=True)
        values = [
            row["accessibility"] for row in outcome.report["ranked"][:10]
            if row["accessibility"] is not None
        ]
        self.assertTrue(values, "intended accessibility disappeared")
        for value in values:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_the_gate_still_sees_accessibility(self):
        """`min_accessibility` reads the intended-site value; it must survive."""
        from cas13guide.metrics.aggregate import intended_accessibility

        self.assertTrue(callable(intended_accessibility))
        outcome = self._design(background=False)
        self.assertTrue(
            any(row["accessibility"] is not None for row in outcome.report["ranked"])
        )

    def test_adding_background_does_not_change_the_intended_numbers(self):
        """The whole argument for the change: identical report, less work."""
        with_bg = self._design(background=True)
        without = self._design(background=False)
        pick = lambda outcome: {          # noqa: E731
            row["guide_id"]: row["accessibility"] for row in outcome.report["ranked"]
        }
        shared = set(pick(with_bg)) & set(pick(without))
        self.assertTrue(shared)
        for guide_id in shared:
            self.assertEqual(pick(with_bg)[guide_id], pick(without)[guide_id])


class TestTheBackgroundSaysWhatItIs(unittest.TestCase):
    """``--background-biotype`` has to reach the loci, not just the parser.

    Every ``--background`` record was ``Universe.CODING`` whatever it was, so
    the two largest entries in ``DEFAULT_HAZARD_BY_BIOTYPE`` -- rRNA at 2.0 and
    mito at 1.5 -- were unreachable from the CLI. A user who supplied rRNA as
    off-target evidence, which is the case those weights exist for, had it
    weighted like any other transcript.
    """

    @classmethod
    def setUpClass(cls):
        import random

        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(19)
        target = "".join(rng.choice("ACGU") for _ in range(300))
        # A near-copy, so there are off-target rows to carry a weight.
        swap = {"A": "G", "G": "A", "C": "U", "U": "C"}
        paralog = "".join(
            swap[base] if index % 11 == 5 else base
            for index, base in enumerate(target)
        )
        cls._directory = tempfile.TemporaryDirectory()
        home = Path(cls._directory.name)
        eol = chr(10)
        (home / "t.fa").write_text(">TGT1" + eol + target + eol, encoding="utf-8")
        (home / "bg.fa").write_text(">BG1" + eol + paralog + eol, encoding="utf-8")

        def run(biotype):
            return run_request(DesignRequest(
                fasta_path=str(home / "t.fa"),
                background_path=str(home / "bg.fa"),
                background_biotype=biotype,
                task="knockdown", assay="cell_culture_knockdown",
                effector="RfxCas13d", use_structure=False, use_signal=False,
            )).report

        cls.as_coding = run("coding")
        cls.as_rrna = run("rrna")

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def test_the_fixture_really_produces_off_target_rows(self):
        """Or every assertion below would hold over an empty table."""
        self.assertTrue(self.as_coding["off_targets"])
        self.assertTrue(self.as_rrna["off_targets"])

    def test_the_label_reaches_the_off_target_rows(self):
        self.assertEqual(
            {row["biotype"] for row in self.as_coding["off_targets"]}, {"coding"}
        )
        self.assertEqual(
            {row["biotype"] for row in self.as_rrna["off_targets"]}, {"rrna"}
        )

    def test_the_label_reaches_the_hazard_weight(self):
        """The consequence those weights exist for.

        The weight is ``unannotated * by_biotype[b]`` -- an off-target the
        contract does not name carries 0.6, times the biotype's prior. Pinning
        the product alone would not say which factor the flag moves, so this
        composes it from the shipped prior.
        """
        from cas13guide.contract.enums import Universe
        from cas13guide.contract.spec import DEFAULT_HAZARD_BY_BIOTYPE, HazardWeights

        unannotated = HazardWeights().unannotated
        self.assertEqual(
            {row["hazard_weight"] for row in self.as_coding["off_targets"]},
            {unannotated * DEFAULT_HAZARD_BY_BIOTYPE[Universe.CODING]},
        )
        self.assertEqual(
            {row["hazard_weight"] for row in self.as_rrna["off_targets"]},
            {unannotated * DEFAULT_HAZARD_BY_BIOTYPE[Universe.RRNA]},
        )
        # And the prior is not 1.0, or the assertion above would hold whatever
        # biotype the background was given.
        self.assertNotEqual(
            DEFAULT_HAZARD_BY_BIOTYPE[Universe.RRNA],
            DEFAULT_HAZARD_BY_BIOTYPE[Universe.CODING],
        )

    def test_an_rrna_background_is_the_more_hazardous_run(self):
        worst_coding = max(row["hazard"] for row in self.as_coding["off_targets"])
        worst_rrna = max(row["hazard"] for row in self.as_rrna["off_targets"])
        self.assertGreater(worst_rrna, worst_coding)


class TestOneBackgroundFileCanHoldSeveralUniverses(unittest.TestCase):
    """``biotype=`` in a FASTA header, so a corpus can cover what an assay needs.

    ``--background-biotype`` labels the whole file, which caps a corpus at two
    universes: the target's and the background's. ``cell_culture_knockdown`` is
    exposed to five, so through this door the adequacy gate was unsatisfiable
    however the user labelled the file -- and the limitation that fires told
    them to reach for the flag that cannot get there.
    """

    ASSAY = "cell_culture_knockdown"

    @classmethod
    def setUpClass(cls):
        import random

        rng = random.Random(23)
        cls._directory = tempfile.TemporaryDirectory()
        cls.home = Path(cls._directory.name)
        eol = chr(10)

        def seq(n):
            return "".join(rng.choice("ACGU") for _ in range(n))

        (cls.home / "t.fa").write_text(
            ">TGT1" + eol + seq(300) + eol, encoding="utf-8")
        labelled = []
        for name, biotype in (("RN18S1", "rrna"), ("MT-CO1", "mito"),
                              ("LINC01", "ncrna"), ("PREM1", "pre_mrna")):
            labelled += [">" + name + " biotype=" + biotype, seq(400)]
        (cls.home / "mixed.fa").write_text(
            eol.join(labelled) + eol, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def _run(self, background="mixed.fa", **extra):
        from cas13guide.services.workflow import DesignRequest, run_request

        return run_request(DesignRequest(
            fasta_path=str(self.home / "t.fa"),
            background_path=str(self.home / background),
            task="knockdown", assay=self.ASSAY, effector="RfxCas13d",
            use_structure=False, use_signal=False, **extra,
        )).report

    def test_each_record_lands_in_the_universe_its_header_names(self):
        report = self._run()
        by_locus = {row["locus"]: row["biotype"] for row in report["off_targets"]}
        # Only loci that produced a hit appear, so assert over what is there.
        for locus, biotype in by_locus.items():
            expected = {"RN18S1": "rrna", "MT-CO1": "mito",
                        "LINC01": "ncrna", "PREM1": "pre_mrna"}.get(locus)
            if expected:
                self.assertEqual(biotype, expected, locus)

    def test_the_corpus_now_covers_what_the_assay_is_exposed_to(self):
        """The whole point: this assay could not reach PASS before."""
        from cas13guide.contract.enums import ASSAY_REQUIRED_UNIVERSES, AssayType

        report = self._run()
        required = ASSAY_REQUIRED_UNIVERSES[AssayType(self.ASSAY)]
        searched = {
            item["text"] for item in report["limitations"]
            if item["id"] == "search-scope-is-what-you-loaded"
        }
        [sentence] = searched
        for universe in required:
            self.assertIn(universe.value, sentence)
        self.assertFalse(
            [item for item in report["limitations"] if item["id"] == "universe-gap"],
            "a corpus covering every required universe still reported a gap",
        )
        self.assertGreater(report["verdict_counts"]["PASS"], 0)

    def test_an_unlabelled_record_takes_the_files_default(self):
        eol = chr(10)
        (self.home / "plain.fa").write_text(
            ">BG1" + eol + ("AAGAAC" * 60) + eol, encoding="utf-8")
        report = self._run(background="plain.fa", background_biotype="rrna")
        for row in report["off_targets"]:
            if row["locus"] == "BG1":
                self.assertEqual(row["biotype"], "rrna")

    def test_the_files_default_is_not_declared_when_no_record_uses_it(self):
        """A label the corpus does not carry must not become a gap.

        `build_spec` used to add the file-level --background-biotype to
        `declared_universes` whenever a background was supplied, on the
        reasoning that an overclaim only widens the gap the gate reads and is
        therefore the safe direction. It is not safe: a widened gap is a
        REVIEW for a universe nobody needed and nobody was missing, and it
        lands on exactly the users these per-record labels exist for.

        Measured on an in_vitro_detection run against a background whose every
        record was labelled rrna or mito: declared {coding, synthetic},
        corpus {synthetic, rrna, mito}, and `coding` reported as a universe
        the search did not cover.
        """
        from cas13guide.contract.enums import Universe
        from cas13guide.services.workflow import (
            DesignRequest,
            _read_records,
            _target_biotype,
            build_spec,
        )

        # in_vitro_detection on purpose: its target biotype is `synthetic`,
        # so it differs from the `coding` file-level default. Under
        # cell_culture_knockdown the two are both `coding` and the set
        # collapses to one element either way -- a fixture that cannot tell
        # the cases apart, which is how the first version of this test passed
        # against the very code it was written to reject.
        request = DesignRequest(
            fasta_path=str(self.home / "t.fa"),
            background_path=str(self.home / "mixed.fa"),
            task="detection", assay="in_vitro_detection", effector="RfxCas13d",
        )
        spec = build_spec(request, _read_records(request), [])
        self.assertNotEqual(
            Universe(request.background_biotype),
            _target_biotype(request),
            "fixture does not discriminate: the default is the target's own "
            "biotype, so declaring it changes nothing",
        )
        self.assertNotIn(
            Universe(request.background_biotype),
            spec.context.declared_universes,
            "the file-level default was declared though no record uses it",
        )
        # What it may assert is the target's own biotype, and nothing else.
        self.assertEqual(
            spec.context.declared_universes, frozenset({_target_biotype(request)})
        )

    def test_no_gap_names_a_universe_no_record_was_labelled_with(self):
        report = self._run()
        gaps = [item for item in report["limitations"]
                if item["id"] == "universe-gap"]
        for item in gaps:
            self.assertNotIn("coding", item["text"])

    def test_an_explicit_target_biotype_beats_what_the_assay_implies(self):
        """`--target-biotype` was documented in the quickstart and tested by
        nothing.

        `_target_biotype` prefers an explicit value, then a preset's own
        biotype, then what the assay implies. Only the last two branches were
        ever reached: the guard honouring the explicit value never ran once in
        the whole suite, so nothing established that the flag does anything.
        """
        from cas13guide.contract.enums import Universe
        from cas13guide.services.workflow import DesignRequest, _target_biotype

        implied = _target_biotype(DesignRequest(
            sequence_text="ACGU", assay="in_vitro_detection"))
        self.assertEqual(implied, Universe.SYNTHETIC, "fixture sanity")

        explicit = _target_biotype(DesignRequest(
            sequence_text="ACGU", assay="in_vitro_detection",
            target_biotype="rrna"))
        self.assertEqual(explicit, Universe.RRNA)
        self.assertNotEqual(explicit, implied, "the flag changed nothing")

    def test_an_explicit_target_biotype_beats_a_presets_own(self):
        """The documented order is explicit, then preset, then assay."""
        from cas13guide.contract.enums import Universe
        from cas13guide.contract.presets import preset_ids
        from cas13guide.services.workflow import DesignRequest, _target_biotype

        preset = sorted(preset_ids())[0]
        from_preset = _target_biotype(DesignRequest(preset=preset))
        self.assertEqual(from_preset, Universe.NCRNA, "fixture sanity: miRNA")

        overridden = _target_biotype(DesignRequest(
            preset=preset, target_biotype="mito"))
        self.assertEqual(overridden, Universe.MITO)

    def test_an_unknown_target_biotype_names_the_flag_too(self):
        """Its sibling refusal names the flag and the valid values; this one
        let the bare enum error through, so the same typo read very
        differently depending on which input carried it."""
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            run_request,
        )

        with self.assertRaises(DesignRequestError) as caught:
            run_request(DesignRequest(
                fasta_path=str(self.home / "t.fa"),
                target_biotype="ribosomal",
                task="knockdown", assay=self.ASSAY, effector="RfxCas13d",
            ))
        message = str(caught.exception)
        self.assertIn("--target-biotype", message)
        self.assertIn("ribosomal", message)
        self.assertIn("rrna", message)

    def test_an_unknown_biotype_names_the_flag_and_the_valid_values(self):
        from cas13guide.services.workflow import DesignRequestError

        eol = chr(10)
        (self.home / "bad.fa").write_text(
            ">BG1 biotype=ribosomal" + eol + ("AAGAAC" * 60) + eol,
            encoding="utf-8")
        with self.assertRaises(DesignRequestError) as caught:
            self._run(background="bad.fa")
        message = str(caught.exception)
        self.assertIn("--background", message)
        self.assertIn("ribosomal", message)
        self.assertIn("rrna", message)


class TestAVariantDoesNotMoveTheTargetToAnotherUniverse(unittest.TestCase):
    """The target is one molecule and must be filed the same way in every world.

    `variant_contexts` builds a separate Transcriptome per haplotype, and the
    biotype it files the target under was passed in separately from the one
    the reference world uses. Once the reference stopped hardcoding NCRNA the
    two disagreed, and because the universe-adequacy gate reads what each
    index actually holds, the variant worlds were missing a universe the
    reference world had: adding --variants to an otherwise identical
    cell_culture run took 223 PASS to 223 REVIEW and reported `coding` as a
    universe the search did not cover.

    The general shape has bitten this module before -- `--variants` used to
    replace the whole context list, so adding one turned a report with
    off-target hits into a clean one. What is pinned here is that adding a
    variant changes only what a variant can change.
    """

    @classmethod
    def setUpClass(cls):
        import random

        from cas13guide.services.workflow import DesignRequest, run_request

        rng = random.Random(23)

        def seq(n):
            return "".join(rng.choice("ACGU") for _ in range(n))

        cls._directory = tempfile.TemporaryDirectory()
        home = Path(cls._directory.name)
        eol = chr(10)
        target = seq(300)
        (home / "t.fa").write_text(">TGT1" + eol + target + eol, encoding="utf-8")
        rows = []
        for name, biotype in (("RN18S1", "rrna"), ("MT-CO1", "mito"),
                              ("LINC01", "ncrna"), ("PREM1", "pre_mrna")):
            rows += [">" + name + " biotype=" + biotype, seq(300)]
        (home / "bg.fa").write_text(eol.join(rows) + eol, encoding="utf-8")
        variant = "".join(
            ("A" if base != "A" else "C") if index % 37 == 9 else base
            for index, base in enumerate(target)
        )
        (home / "var.fa").write_text(">hap2" + eol + variant + eol,
                                     encoding="utf-8")

        def run(**extra):
            return run_request(DesignRequest(
                fasta_path=str(home / "t.fa"),
                background_path=str(home / "bg.fa"),
                task="knockdown", assay="cell_culture_knockdown",
                effector="RfxCas13d", use_structure=False, use_signal=False,
                **extra,
            )).report

        cls.plain = run()
        cls.with_variant = run(variants_path=str(home / "var.fa"))

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    @staticmethod
    def _gap(report):
        return [item for item in report["limitations"]
                if item["id"] == "universe-gap"]

    def test_the_fixture_is_adequate_without_the_variant(self):
        """Or the assertion below would hold over a run that was never clean."""
        self.assertEqual(self._gap(self.plain), [])
        self.assertGreater(self.plain["verdict_counts"]["PASS"], 0)

    def test_adding_a_variant_introduces_no_universe_gap(self):
        self.assertEqual(
            self._gap(self.with_variant), [],
            "adding a variant reported a universe the reference run covered",
        )

    def test_the_searched_universes_are_the_same_either_way(self):
        def scope(report):
            [item] = [x for x in report["limitations"]
                      if x["id"] == "search-scope-is-what-you-loaded"]
            return item["text"]

        self.assertEqual(scope(self.plain), scope(self.with_variant))

    def test_a_variant_still_changes_what_a_variant_should(self):
        """Not a no-op: guides that lose the variant are downgraded."""
        self.assertLess(
            self.with_variant["verdict_counts"]["PASS"],
            self.plain["verdict_counts"]["PASS"],
        )
        reasons = {
            reason
            for row in self.with_variant["ranked"]
            for reason in row.get("reasons", ())
        }
        self.assertIn("variant_coverage_lost", reasons)


if __name__ == "__main__":
    unittest.main()
