"""A tenth pass, run as eight parallel hunts with two skeptics per finding.

Twenty-three candidate defects, fifteen refuted, eight confirmed. Three of the
eight were regressions introduced by the round immediately before it, which is
the argument for adversarial verification and also its limit: a pass that only
re-reads its own diff finds what it was already looking for.

The eight share a shape worth naming. Every one is a *disagreement between two
descriptions of the same run* -- a note that says a record was ignored beside a
metric computed from it, a table that omits the hazard the verdict was based
on, a glossary that denies a figure the CSV carries, a count that changes with
a display flag. None is a wrong formula. The arithmetic in this package has
been checked to death; what had not been checked was whether its own accounts
of itself agree.
"""

import json
import pathlib
import random
import subprocess
import sys
import tempfile
import unittest

from helpers import *  # noqa: F401,F403
from test_orchestrator import seeded_calibration

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _fasta(directory: pathlib.Path, name: str, records: dict[str, str]) -> pathlib.Path:
    path = directory / name
    path.write_text(
        "".join(f">{k}\n{v}\n" for k, v in records.items()), encoding="utf-8"
    )
    return path


class TestTheDemoDisclosesItsSyntheticCalibration(unittest.TestCase):
    """The demo printed three of the four CRITICALs that apply to it.

    `limitations_for_run` decides via `getattr(calibration, "synthetic", False)`.
    `services/workflow.seed_calibration` sets the flag; `demo.seed_calibration`
    built an equivalent stand-in from `RNG.random()` and never did, so the one
    caveat that qualifies every conformal interval on the page -- and every
    interval in the report.json the demo writes -- was absent from both.

    The commit that added the demo's limitations section named this exact
    caveat as the first thing that had been missing.
    """

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            cls.result = subprocess.run(
                [sys.executable, "-m", "cas13guide.cli", "demo"],
                cwd=directory, capture_output=True, text=True,
            )
            report = pathlib.Path(directory) / "report.json"
            cls.report = json.loads(report.read_text(encoding="utf-8"))

    def test_the_terminal_says_the_calibration_is_synthetic(self):
        self.assertEqual(self.result.returncode, 0, self.result.stderr[-1200:])
        self.assertIn("SYNTHETIC", self.result.stdout)

    def test_the_json_carries_the_limitation_too(self):
        ids = {item["id"] for item in self.report["limitations"]}
        self.assertIn("synthetic-calibration", ids)

    def test_all_four_criticals_are_present(self):
        """Named individually, because a bare count of "CRITICAL:" was what let
        three of four stay green."""
        critical = {
            item["id"] for item in self.report["limitations"]
            if item["severity"] == "critical"
        }
        self.assertEqual(
            critical,
            {"collateral-not-predicted", "heuristic-activity-model",
             "search-scope-is-what-you-loaded", "synthetic-calibration"},
        )

    def test_the_ledger_records_where_the_calibration_came_from(self):
        """The same omission, one field over: the provenance ledger recorded
        this calibration as "unstated" while the design path records the seed
        and noise that generated it."""
        sources = [
            record.get("detail", {}).get("source")
            for record in self.report["provenance"]["records"]
            if record.get("kind") == "calibration"
        ]
        self.assertTrue(sources, "no calibration record in the ledger")
        self.assertEqual(sources[0].get("kind"), "synthetic")


class TestTheRankedCountIsTheRankedCount(unittest.TestCase):
    """`--top 0` reported the distinct-site count, not the ranked count.

    The message says "N guide(s) ranked ... the exports carry them all" and
    pointed at a CSV holding a different number. Worse, N moved with
    `--all-sites` -- a pure display flag deciding a stated fact about the run,
    which is what the sibling fix to the exit status set out to stop.
    """

    @classmethod
    def setUpClass(cls):
        rng = random.Random(5)
        cls.directory = tempfile.mkdtemp()
        work = pathlib.Path(cls.directory)
        cls.fasta = _fasta(work, "two.fa", {
            f"r{i}": "".join(rng.choice("ACGU") for _ in range(300)) for i in range(2)
        })

    def _run(self, *extra):
        out = pathlib.Path(self.directory) / f"out{len(extra)}"
        result = subprocess.run(
            [sys.executable, "-m", "cas13guide.cli", "design",
             "--fasta", str(self.fasta), "--task", "knockdown",
             "--assay", "cell_culture_knockdown", "--effector", "RfxCas13d",
             "--no-structure", "--no-signal", "--top", "0",
             "--out", str(out), "--stem", "c", *extra],
            cwd=ROOT, capture_output=True, text=True,
        )
        line = next(
            (l for l in result.stdout.splitlines() if "guide(s) ranked" in l), ""
        )
        stated = int(line.split()[0]) if line else -1
        rows = [
            l for l in (out / "c.csv").read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")
        ]
        return stated, len(rows) - 1  # minus the header

    def test_the_count_matches_the_csv_it_points_at(self):
        stated, exported = self._run()
        self.assertGreater(exported, 10, "fixture must export a meaningful number")
        self.assertEqual(stated, exported)

    def test_a_display_flag_does_not_change_it(self):
        self.assertEqual(self._run()[0], self._run("--all-sites")[0])


class TestAnUnreadablePathIsLocatedNotATraceback(unittest.TestCase):
    """`read_fasta` caught only UnicodeDecodeError.

    Every other OSError -- a directory where a file was meant, an unreadable
    file, a broken symlink -- escaped as a traceback and exit 1, the status
    `design --help` documents as "the run completed and nothing survived the
    gate". A wrapper script read a typo'd path as a legitimate empty result.
    """

    def test_a_directory_given_to_each_path_flag_is_refused_with_a_message(self):
        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            (work / "adir").mkdir()
            good = _fasta(work, "t.fa", {"t1": "ACGU" * 60})
            for flag in ("--fasta", "--background", "--variants"):
                args = ([flag, str(work / "adir")] if flag == "--fasta"
                        else ["--fasta", str(good), flag, str(work / "adir")])
                with self.subTest(flag=flag):
                    result = subprocess.run(
                        [sys.executable, "-m", "cas13guide.cli", "design", *args,
                         "--task", "knockdown", "--assay", "cell_culture_knockdown",
                         "--effector", "RfxCas13d", "--no-structure", "--no-signal"],
                        cwd=ROOT, capture_output=True, text=True,
                    )
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual(result.returncode, 2, result.stderr[-600:])

    def test_read_fasta_raises_the_input_error_not_an_oserror(self):
        from cas13guide.services.fasta import SequenceInputError, read_fasta

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SequenceInputError) as caught:
                read_fasta(directory)
        self.assertIn("cannot read", str(caught.exception))


class TestABackgroundRecordSaidToBeIgnoredIsIgnored(unittest.TestCase):
    """The filtered list guarded the reference world; the unfiltered one went
    to the variant contexts.

    A background record sharing a target's name is dropped from the reference
    world with a note saying so -- and was then added to every *variant* world,
    where the name does not collide. Adding `--variants` to an otherwise
    identical request turned worst_off_target from 0.000 into 0.571 while the
    note kept saying the copy was ignored. A background corpus pulled from the
    same annotation as the target routinely contains the target's own id; the
    code's own comment says so.
    """

    @classmethod
    def setUpClass(cls):
        rng = random.Random(4)
        cls.target = "".join(rng.choice("ACGU") for _ in range(200))
        variant = list(cls.target)
        for i in (30, 90, 150):
            variant[i] = "A" if variant[i] != "A" else "G"
        cls.variant = "".join(variant)

    def _run(self, with_variants: bool):
        from cas13guide.services.workflow import DesignRequest, run_request

        with tempfile.TemporaryDirectory() as directory:
            work = pathlib.Path(directory)
            target = _fasta(work, "t.fa", {"TGT": self.target})
            background = _fasta(work, "bg.fa", {"TGT": self.target})
            extra = {}
            if with_variants:
                extra["variants_path"] = str(_fasta(work, "v.fa", {"VAR1": self.variant}))
            outcome = run_request(DesignRequest(
                fasta_path=str(target), background_path=str(background),
                task="knockdown", assay="cell_culture_knockdown",
                effector="RfxCas13d", use_structure=False, use_signal=False,
                max_mismatches=2, **extra))
        rows = outcome.report["ranked"]
        return {
            "worst": max((r["metrics"]["worst_off_target"] for r in rows), default=0.0),
            "burden": max((r["metrics"]["cumulative_burden"] for r in rows), default=0.0),
            "note": any("shares a name" in n for n in outcome.notes),
        }

    def test_the_note_is_true_without_variants(self):
        plain = self._run(with_variants=False)
        self.assertTrue(plain["note"], "fixture must trigger the collision note")
        self.assertEqual(plain["worst"], 0.0)

    def test_adding_variants_does_not_conjure_the_hazard_back(self):
        plain, with_variants = self._run(False), self._run(True)
        self.assertTrue(with_variants["note"])
        self.assertEqual(
            with_variants["worst"], plain["worst"],
            "the record the run says it ignored became an off-target in the "
            "variant contexts",
        )
        self.assertEqual(with_variants["burden"], plain["burden"])


class TestTheExplorerShowsTheHazardTheGateActedOn(unittest.TestCase):
    """`worst_off_target` is a max across contexts; the explorer read one.

    `GuideAssessment.evidence` is the reference context, and the off-target
    table was built from it, while the metric and the verdict come from every
    context. So a hazard living only in a variant world drove the verdict and
    appeared nowhere -- under a validation plan that instructs the reader to
    "sequence the top off-target loci from the explorer table".
    """

    @classmethod
    def setUpClass(cls):
        from cas13guide.contract.enums import Universe
        from cas13guide.pipeline.orchestrator import DesignContext, run_design
        from cas13guide.services.registry import (
            HeuristicCas13Model, ModelRegistry, NullAccessibilityModel,
        )
        from cas13guide.services.transcriptome import Locus, Transcriptome

        rng = random.Random(7)
        target = "".join(rng.choice("ACGU") for _ in range(180))

        def rna(n):
            return "".join(rng.choice("ACGU") for _ in range(n))

        def world(hazardous: bool) -> Transcriptome:
            t = Transcriptome("GRCh38", "v44")
            t.add(Locus("L_T1", target, Universe.CODING, gene="TGT", expression=100.0))
            # Same locus id in both worlds; innocent in one, a copy of a target
            # site at high abundance in the other.
            t.add(Locus(
                "OFF1",
                rna(60) + (target[40:62] if hazardous else rna(22)) + rna(60),
                Universe.CODING, gene="OFF1",
                expression=3000.0 if hazardous else 5.0,
            ))
            return t

        from cas13guide.contract.enums import TargetClass
        from cas13guide.contract.spec import TargetSpec

        spec = make_spec(targets=(TargetSpec(
            target_id="t1", sequence=target,
            target_class=TargetClass.REQUIRED, expression=100.0),))
        universes = frozenset(Universe)
        contexts = [
            DesignContext.auto("reference", world(False), universes, {"L_T1": "t1"}),
            DesignContext.auto("hap2", world(True), universes, {"L_T1": "t1"}),
        ]
        cls.data = run_design(
            spec, contexts,
            ModelRegistry(HeuristicCas13Model(), NullAccessibilityModel()),
            seeded_calibration(spec), panel_size=2,
        ).to_dict()

    def test_the_fixture_produces_a_variant_only_hazard(self):
        hazardous = [r for r in self.data["ranked"]
                     if r["metrics"]["worst_off_target"] > 0]
        self.assertTrue(hazardous, "fixture must produce a hazard to look for")

    def test_every_guide_judged_on_a_hazard_has_a_row(self):
        hazardous = [r["guide_id"] for r in self.data["ranked"]
                     if r["metrics"]["worst_off_target"] > 0]
        listed = {row["guide_id"] for row in self.data["off_targets"]}
        missing = [g for g in hazardous if g not in listed]
        self.assertFalse(
            missing,
            f"{len(missing)} guides were judged on a hazard the table omits",
        )

    def test_the_row_says_which_world_the_hit_was_seen_in(self):
        rows = self.data["off_targets"]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(locus=row["locus"]):
                self.assertIn("context", row)
                self.assertTrue(row["context"])
        self.assertIn("hap2", {row["context"] for row in rows})

    def test_one_row_per_guide_and_locus(self):
        """The table is capped at fifty. Spanning every context without
        deduplicating would fill it with repeats of one locus."""
        keys = [(row["guide_id"], row["locus"]) for row in self.data["off_targets"]]
        self.assertEqual(len(keys), len(set(keys)))


class TestTheReportDoesNotDenyAFigureItCarries(unittest.TestCase):
    """The Signal glossary was a module constant, so it described every run.

    It said "the kinetic constants shipped by default are placeholders, so no
    absolute figure is offered" -- including on a run that supplied measured
    constants, whose CSV carries `predicted_signal_nm` and whose own
    limitations correctly drop the placeholder warning. Three surfaces
    describing one run and one of them contradicting the other two, in the
    module whose docstring warns against a banner that "says the same thing
    whether or not it is true this time".
    """

    def _html(self, *, absolute: bool) -> str:
        from cas13guide.services.report_html import render_report

        row = {
            "rank": 1, "guide_id": "g1", "spacer": "ACGU", "verdict": "PASS",
            "score": 1.0, "predicted_signal": 0.0078,
            "predicted_signal_nm": 3.8765 if absolute else None,
        }
        return render_report({"verdict_counts": {}, "ranked": [row]})

    def test_a_placeholder_run_still_says_no_absolute_figure_is_offered(self):
        html = self._html(absolute=False)
        self.assertIn("shipped by default are placeholders", html)
        self.assertNotIn(">nM<", html)

    def test_a_measured_run_stops_denying_it(self):
        html = self._html(absolute=True)
        self.assertNotIn("shipped by default are placeholders", html)

    def test_a_measured_run_shows_the_column(self):
        html = self._html(absolute=True)
        self.assertIn(">nM<", html)
        self.assertIn("3.8765", html)

    def test_the_placeholder_wording_is_still_available_to_explain_the_absence(self):
        """The measured run should explain why the figure exists, not pretend
        the placeholder rule never applied."""
        self.assertIn("placeholder", self._html(absolute=True))


class TestATrainedModelIsNotAskedOutOfItsDomain(unittest.TestCase):
    """The domain check ran, then the model was called anyway.

    `evidence_for_hit` computed `in_domain`, called `predict()`
    unconditionally, and three lines later threw the result away when
    `in_domain` was False. Harmless for the built-in heuristic; fatal for a
    trained one, whose FeatureSpec is fixed at the screen's spacer length and
    raises for any other. Fitting on a 22 nt screen and designing at 23 nt --
    both canonical for RfxCas13d -- crashed the run on a prediction nothing
    was going to read.
    """

    def test_a_design_at_another_canonical_length_completes(self):
        from cas13guide.contract.enums import Effector, Universe
        from cas13guide.contract.spec import EffectorAssayProfile
        from cas13guide.pipeline.orchestrator import DesignContext, run_design
        from cas13guide.services.registry import ModelRegistry, NullAccessibilityModel
        from cas13guide.services.training import train_and_calibrate
        from cas13guide.services.transcriptome import Locus, Transcriptome
        from cas13guide.simulate import simulate_screen

        examples = simulate_screen(
            seed=21, step=4, signal_scale=10, transcript_effect_sd=0.8,
            noise_sd=0.2, n_transcripts=40, transcript_length=220)[0]
        trained = train_and_calibrate(examples, alpha=1.0, seed=5)

        rng = random.Random(4)
        target = "".join(rng.choice("ACGU") for _ in range(240))
        from cas13guide.contract.enums import TargetClass
        from cas13guide.contract.spec import TargetSpec

        spec = make_spec(targets=(TargetSpec(
            target_id="t1", sequence=target,
            target_class=TargetClass.REQUIRED, expression=100.0),))
        spec = spec.model_copy(update={"profile": EffectorAssayProfile(
            effector=Effector.RFX_CAS13D, assay=spec.profile.assay,
            spacer_length=23)})

        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("L1", target, Universe.CODING, gene="T1", expression=100.0))
        contexts = [DesignContext.auto(
            "reference", world, frozenset(Universe), {"L1": "t1"})]

        report = run_design(
            spec, contexts,
            ModelRegistry(trained.model, NullAccessibilityModel()),
            trained.calibration, panel_size=2)

        counts = report.verdict_counts
        self.assertGreater(sum(counts.values()), 0, "no guide was assessed at all")

    def test_an_out_of_domain_hit_never_reaches_the_model(self):
        """Directly: a model that raises on every call must not be called when
        the domain has already said it cannot answer."""
        from cas13guide.contract.enums import Universe
        from cas13guide.pipeline.d_pair_evidence import evidence_for_hit
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.registry import (
            ModelDomain, ModelRegistry, NullAccessibilityModel,
        )
        from cas13guide.services.transcriptome import Locus, Transcriptome

        spacer = "GCAUUGCUGUGGAGCUGAAGUC"

        class Exploding:
            """A model whose domain admits nothing at all."""

            version = "exploding/v1"
            # No effector is in the domain, so every hit is out of it.
            domain = ModelDomain(effectors=frozenset(),
                                 spacer_lengths=frozenset(), max_mismatches=0)

            def predict(self, *args, **kwargs):  # pragma: no cover - must not run
                raise AssertionError("predict() called on an out-of-domain hit")

        hit = make_hit("g1", "L1")
        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("L1", hit.target_site, Universe.CODING, expression=10.0))

        class _NoExpression:
            def expression_tpm(self, locus_id):
                return None

        evidence = evidence_for_hit(
            hit, spacer, make_spec(),
            ModelRegistry(Exploding(), NullAccessibilityModel()),
            CalibrationService(), world, _NoExpression(),
        )
        self.assertTrue(
            evidence.ood or evidence.activity.ood,
            "an out-of-domain hit must be marked, not silently narrowed",
        )


if __name__ == "__main__":
    unittest.main()
