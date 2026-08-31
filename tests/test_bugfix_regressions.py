"""Defects found by an adversarial review pass, each pinned so it stays fixed.

Every test here fails against the code as it stood. They are grouped by what
went wrong rather than by module, because the grouping is the finding: most of
these are a claim the code made about itself and did not keep, which is the
failure mode this package is arranged against and the one a green suite is
worst at catching.
"""

import dataclasses
import io
import json
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.contract.effectors import EFFECTORS
from cas13guide.contract.enums import Effector, TargetClass, Universe
from cas13guide.contract.spec import TargetSpec
from cas13guide.services.registry import HeuristicCas13Model, ModelDomain


class TestTheModelDomainPairsLengthWithOrtholog(unittest.TestCase):
    """A union across orthologs is not a domain.

    ``ModelDomain.supports`` tested effector and spacer length as independent
    axes while ``HeuristicCas13Model`` declared ``spacer_lengths`` as the union
    over both orthologs it covers. RfxCas13d publishes 22/23/30 and LwaCas13a
    publishes 28, so the union admitted all four for both -- exactly the
    geometries neither was validated at. It is the only check between an
    unvalidated spacer length and a certified narrow bound, and four separate
    docstrings promised it did the opposite.
    """

    def setUp(self):
        self.domain = HeuristicCas13Model().domain

    def test_each_ortholog_admits_only_its_own_published_lengths(self):
        for effector, params in EFFECTORS.items():
            if effector not in self.domain.effectors:
                continue
            for length in (22, 23, 28, 30):
                with self.subTest(effector=effector.value, length=length):
                    self.assertEqual(
                        self.domain.supports(effector, length, 0),
                        length in params.spacer_lengths,
                    )

    def test_the_specific_crossings_that_were_admitted(self):
        """The four that mattered: every miRNA preset resolves to 22 or 23."""
        self.assertFalse(self.domain.supports(Effector.LWA_CAS13A, 22, 0))
        self.assertFalse(self.domain.supports(Effector.LWA_CAS13A, 23, 0))
        self.assertFalse(self.domain.supports(Effector.LWA_CAS13A, 30, 0))
        self.assertFalse(self.domain.supports(Effector.RFX_CAS13D, 28, 0))

    def test_a_multi_effector_domain_cannot_be_built_from_a_flat_set(self):
        """The footgun is removed, not merely avoided at the one call site."""
        with self.assertRaises(ValueError) as caught:
            ModelDomain(
                effectors=frozenset({Effector.RFX_CAS13D, Effector.LWA_CAS13A}),
                spacer_lengths=frozenset({22, 28}),
                max_mismatches=4,
            )
        self.assertIn("per effector", str(caught.exception))

    def test_a_single_effector_domain_still_takes_a_flat_set(self):
        """There is nothing to conflate with one ortholog, and the trained and
        Arc models both build theirs that way."""
        domain = ModelDomain(
            effectors=frozenset({Effector.RFX_CAS13D}),
            spacer_lengths=frozenset({30}),
            max_mismatches=0,
        )
        self.assertTrue(domain.supports(Effector.RFX_CAS13D, 30, 0))
        self.assertFalse(domain.supports(Effector.LWA_CAS13A, 30, 0))

    def test_an_unvalidated_length_is_refused_rather_than_certified(self):
        """End to end, through the door a user actually comes in by."""
        from cas13guide.services.workflow import DesignRequest, run_request

        outcome = run_request(
            DesignRequest(
                preset="hsa-miR-216a-5p",       # 22 nt
                effector="LwaCas13a",           # publishes 28 only
                assay="in_vitro_detection",
                task="detection",
                expression=100.0,
                use_structure=False,
                use_signal=False,
            )
        )
        self.assertEqual(outcome.report["verdict_counts"]["PASS"], 0)


class TestTheFingerprintIsStableAcrossProcesses(unittest.TestCase):
    """``fingerprint()`` promises two identical designs share one.

    ``declared_universes`` is a frozenset of a str Enum, which pydantic
    serialises as a list in iteration order -- and Enum hashing is a
    per-process-randomised string hash. ``sort_keys=True`` orders dict keys,
    not list elements, so the digest moved between runs of the same design.
    """

    def test_the_serialised_universes_are_sorted(self):
        spec = make_spec()
        dumped = spec.model_dump(mode="json")
        universes = dumped["context"]["declared_universes"]
        self.assertEqual(universes, sorted(universes))

    def test_the_fingerprint_survives_a_fresh_interpreter(self):
        """Out of process, twice, with hash randomisation left on.

        In-process repetition cannot see this: one process has one seed.
        """
        import subprocess
        import sys

        code = (
            "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'tests');"
            "from helpers import make_spec; print(make_spec().fingerprint())"
        )
        seen = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, timeout=120,
                cwd=str(Path(__file__).resolve().parents[1]),
            ).stdout.strip()
            for _ in range(3)
        }
        self.assertEqual(len(seen), 1, seen)
        self.assertNotIn("", seen)


class TestEveryIntendedTargetIsTiled(unittest.TestCase):
    """The candidate budget was spent in contract order.

    ``design_candidates`` returned the moment the budget was exhausted, so a
    later REQUIRED target was never tiled -- and the run then reported "no
    feasible panel covers required target(s): t3", which reads as "t3 has no
    usable guides" rather than "nobody looked". Reachable at the shipped
    default: 5000 candidates against three ordinary 3 kb transcripts.
    """

    @staticmethod
    def _sequence(seed_value, length):
        """Non-repeating, so windows are distinct.

        Candidates deduplicate by spacer sequence, so a periodic fixture would
        measure the deduplicator rather than the budget.
        """
        import random

        rng = random.Random(seed_value)
        return "".join(rng.choice("ACGU") for _ in range(length))

    def _spec(self, n_targets, length=3000):
        targets = tuple(
            TargetSpec(
                target_id=f"t{i + 1}",
                sequence=self._sequence(i, length),
                target_class=TargetClass.REQUIRED,
            )
            for i in range(n_targets)
        )
        return make_spec(targets=targets)

    def test_no_required_target_gets_zero_candidates(self):
        from cas13guide.pipeline.a_candidates import design_candidates

        spec = self._spec(3)
        candidates = design_candidates(spec)
        covered = {c.source_target_id for c in candidates}
        self.assertEqual(covered, {"t1", "t2", "t3"})

    def test_the_budget_is_still_respected(self):
        from cas13guide.pipeline.a_candidates import design_candidates

        spec = self._spec(3)
        self.assertLessEqual(
            len(design_candidates(spec)), spec.budget.max_candidates
        )

    def test_a_starved_target_is_spanned_sparsely_not_truncated(self):
        """Half a transcript at full density is a wrong answer; the whole
        transcript at low density is a worse-but-honest one."""
        from cas13guide.pipeline.a_candidates import design_candidates

        spec = self._spec(3)
        candidates = design_candidates(spec)
        for target_id in ("t1", "t2", "t3"):
            starts = [c.start for c in candidates if c.source_target_id == target_id]
            with self.subTest(target=target_id):
                self.assertEqual(min(starts), 0)
                self.assertGreater(max(starts), 2000)

    def test_target_order_does_not_change_who_is_designable(self):
        from cas13guide.pipeline.a_candidates import design_candidates

        spec = self._spec(3)
        forward = {c.source_target_id for c in design_candidates(spec)}
        reversed_spec = spec.model_copy(
            update={
                "contract": spec.contract.model_copy(
                    update={"targets": tuple(reversed(spec.contract.targets))}
                )
            }
        )
        self.assertEqual(forward, {c.source_target_id for c in design_candidates(reversed_spec)})


class TestVariantsDoNotDiscardTheOffTargetCorpus(unittest.TestCase):
    """``--variants`` replaced the whole context list.

    Each variant world held that variant and nothing else, so ``--background``
    was parsed and dropped and ``--blast-db`` was validated and never queried.
    Adding a variant to an otherwise identical request turned a report with
    off-target hits into a clean one -- the worst direction for this number.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        target = (SEQ_A * 6)[:300]
        self.target = root / "t.fa"
        self.target.write_text(f">T1\n{target}\n", encoding="utf-8")
        self.background = root / "bg.fa"
        self.background.write_text(f">decoy\n{SEQ_B}{target[80:230]}\n", encoding="utf-8")
        variant = list(target)
        variant[10] = "A" if variant[10] != "A" else "C"
        self.variants = root / "v.fa"
        self.variants.write_text(">V1\n" + "".join(variant) + "\n", encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def _run(self, **kw):
        from cas13guide.services.workflow import DesignRequest, run_request

        return run_request(
            DesignRequest(
                fasta_path=str(self.target),
                background_path=str(self.background),
                task="knockdown",
                assay="cell_culture_knockdown",
                effector="RfxCas13d",
                use_structure=False,
                use_signal=False,
                **kw,
            )
        )

    def test_the_background_is_searched_with_and_without_variants(self):
        without = len(self._run().report["off_targets"])
        with_variants = len(self._run(variants_path=str(self.variants)).report["off_targets"])
        self.assertGreater(without, 0, "fixture must produce off-targets to be a test")
        self.assertGreater(
            with_variants, 0,
            "adding --variants silently emptied the off-target report",
        )

    def test_variant_contexts_carry_the_background_loci(self):
        from cas13guide.services.conservation import variant_contexts
        from cas13guide.services.fasta import SequenceRecord
        from cas13guide.services.transcriptome import Locus

        decoy = Locus("decoy", SEQ_B, Universe.CODING, gene="decoy", expression=None)
        contexts = variant_contexts(
            [SequenceRecord("T1", SEQ_A), SequenceRecord("V1", SEQ_A)],
            frozenset(Universe),
            target_id="T1",
            background=[decoy],
        )
        for context in contexts:
            with self.subTest(context=context.name):
                self.assertIn("decoy", set(context.transcriptome.loci))


class TestDuplicateVariantNamesAreRefused(unittest.TestCase):
    """Worlds key loci by name, so two variants under one name collapse.

    The survivor's result was then reported under the shared name, i.e.
    "reaches every variant" for a variant never searched. ``services/fasta.py``
    deliberately does not deduplicate names, so a concatenated FASTA reaches
    here unchanged.
    """

    def test_conservation_refuses_repeated_names(self):
        from cas13guide.services.conservation import summarize_conservation
        from cas13guide.services.fasta import SequenceRecord

        records = [
            SequenceRecord("ref", SEQ_A),
            SequenceRecord("var", SEQ_B),
            SequenceRecord("var", SEQ_A),
        ]
        with self.assertRaises(ValueError) as caught:
            summarize_conservation(records, 22)
        self.assertIn("var", str(caught.exception))

    def test_distinct_names_are_untouched(self):
        from cas13guide.services.conservation import summarize_conservation
        from cas13guide.services.fasta import SequenceRecord

        summary = summarize_conservation(
            [SequenceRecord("ref", SEQ_A), SequenceRecord("v1", SEQ_A)], 22
        )
        self.assertGreater(summary.windows_examined, 0)


class TestDiscriminationMarginScoresAMissingAlleleAsWorst(unittest.TestCase):
    """``inf`` needs on-target benefit to be infinitely better *than*.

    The no-off-evidence branch returned ``inf`` before consulting the ``on``
    term, re-creating in that branch the exact defect the function's docstring
    records having fixed in the min: a context that lost a required allele
    scored the best value the metric can take.
    """

    def _margin(self, evidence, spec):
        from cas13guide.metrics.aggregate import discrimination_margin

        return discrimination_margin(evidence, spec)

    def _spec(self):
        from cas13guide.contract.enums import Task

        return make_spec(
            task=Task.ALLELE,
            targets=(
                TargetSpec(target_id="mutA", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
                TargetSpec(target_id="wt", sequence=SEQ_B, target_class=TargetClass.FORBIDDEN),
            ),
        )

    def test_no_evidence_for_a_required_allele_scores_zero_not_infinity(self):
        spec = self._spec()
        self.assertEqual(self._margin((), spec), 0.0)

    def test_covered_on_target_with_no_off_allele_evidence_is_still_infinite(self):
        from cas13guide.metrics.bounds import Interval
        from cas13guide.models.evidence import PairEvidence

        spec = self._spec()
        evidence = (
            PairEvidence(
                guide_id="g1",
                locus_id="L_mutA",
                context="reference",
                hit=make_hit("g1", "L_mutA", "mutA", TargetClass.REQUIRED),
                activity=Interval(0.9, 0.9, 0.95),
                expression=Interval(0.5, 0.5, 0.5),
                hazard_weight=0.0,
            ),
        )
        self.assertEqual(self._margin(evidence, spec), float("inf"))


class TestTheKineticIntegratorIsStableOrRefuses(unittest.TestCase):
    """A fixed 400 steps is unstable for ordinary measured constants.

    The per-step clamps then turned the divergence into a plausible wrong
    number rather than a NaN -- in the one path that unlocks when a user
    supplies real constants.
    """

    def _constants(self, **kw):
        from cas13guide.services.kinetics import KineticConstants

        base = dict(
            k_on=1.0, k_off=1e-4, k_cat=4e-3, k_m=200.0, complex_nm=50.0,
            target_nm=1.0, reporter_nm=500.0, assay_seconds=1800.0,
            source="measured", provisional=False,
        )
        base.update(kw)
        return KineticConstants(**base)

    def test_a_fast_association_rate_converges(self):
        from cas13guide.services.kinetics import CollateralKineticsModel

        model = CollateralKineticsModel(self._constants())
        _activated, _rate, cleaved = model.integrate(1.0)
        # 5.1352 from a 180k-step reference integration; 6.1373 from the
        # diverging 400-step one, a silent +19.5%.
        self.assertAlmostEqual(cleaved, 5.1352, places=3)

    def test_mass_balance_holds_when_target_exceeds_the_effector(self):
        from cas13guide.services.kinetics import CollateralKineticsModel

        model = CollateralKineticsModel(self._constants(target_nm=100.0))
        activated, _rate, _cleaved = model.integrate(100.0)
        # [C*] cannot exceed the complex supplied, and [T] is derived from it
        # rather than clamped separately, so the pair cannot drift apart.
        self.assertLessEqual(activated, 1.0 + 1e-12)

    def test_constants_too_stiff_to_resolve_are_refused_not_approximated(self):
        from cas13guide.services.kinetics import (
            CollateralKineticsModel,
            StiffKineticsError,
        )

        model = CollateralKineticsModel(self._constants(k_on=50.0))
        with self.assertRaises(StiffKineticsError) as caught:
            model.integrate(1.0)
        self.assertIn("stiff solver", str(caught.exception))

    def test_the_shipped_placeholders_are_unaffected(self):
        """The adaptive step count must not move the default ranking."""
        from cas13guide.services.kinetics import CollateralKineticsModel

        prediction = CollateralKineticsModel().predict(1.0)
        self.assertGreater(prediction.relative_signal, 0.0)
        self.assertLess(prediction.relative_signal, 1.0)


class TestShortSequencesDoNotKillTheProcess(unittest.TestCase):
    """``RNA.pfl_fold_up`` aborts natively at 3 and 4 nt.

    Not an exception -- a process kill with no traceback, which under ``serve``
    would take every in-flight design down with it. The docstring said these
    lengths were handled.
    """

    def test_every_short_length_returns_a_profile(self):
        try:
            from cas13guide.services.accessibility import ViennaRNAAccessibilityModel

            model = ViennaRNAAccessibilityModel()
        except Exception:
            self.skipTest("ViennaRNA is not installed")
        for length in range(0, 8):
            with self.subTest(length=length):
                profile = model.unpaired_profile("GCAUGCAU"[:length])
                self.assertEqual(len(profile), length)

    def test_a_sequence_too_short_to_pair_is_fully_unpaired(self):
        try:
            from cas13guide.services.accessibility import ViennaRNAAccessibilityModel

            model = ViennaRNAAccessibilityModel()
        except Exception:
            self.skipTest("ViennaRNA is not installed")
        self.assertEqual(model.unpaired_profile("GCA"), (1.0, 1.0, 1.0))


class TestInputPathsFailWithTheDocumentedType(unittest.TestCase):
    """``run_request`` documents ``DesignRequestError`` on bad input.

    ``_read_records`` upheld it for ``--fasta``; ``--background``,
    ``--variants`` and ``--kinetics-constants`` did not, so the identical typo
    printed a clean message on one flag and a stack trace on the next.
    """

    def _raises(self, **kw):
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            run_request,
        )

        with self.assertRaises(DesignRequestError) as caught:
            run_request(DesignRequest(preset="hsa-miR-21-5p", **kw))
        return str(caught.exception)

    def test_a_missing_background_file(self):
        self.assertIn("--background", self._raises(background_path="nope_missing.fa"))

    def test_a_missing_variants_file(self):
        self.assertIn("--variants", self._raises(variants_path="nope_missing.fa"))

    def test_a_missing_kinetics_file(self):
        self.assertIn(
            "--kinetics-constants", self._raises(kinetics_json="nope_missing.json")
        )


class TestAByteOrderMarkIsEncodingNotContent(unittest.TestCase):
    """A BOM'd FASTA was rejected with a message naming the one thing that was
    right: "must start with a '>' header line", about a file that does."""

    def test_a_bom_file_reads(self):
        from cas13guide.services.fasta import read_fasta

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bom.fa"
            path.write_bytes(">mir21 test\nAUGCAUGC\n".encode("utf-8-sig"))
            self.assertEqual([r.name for r in read_fasta(path)], ["mir21"])

    def test_bom_pasted_text_reads_the_same_way(self):
        """The web form decodes its own bytes, so the file and paste paths have
        to agree or the same file behaves differently by which door it used."""
        from cas13guide.services.fasta import parse_sequence_input

        self.assertEqual(
            [r.name for r in parse_sequence_input("﻿>mir21 test\nAUGCAUGC\n")],
            ["mir21"],
        )
        self.assertEqual(
            parse_sequence_input("﻿AUGCAUGC")[0].sequence, "AUGCAUGC"
        )


class TestScreenFilesAreUtf8(unittest.TestCase):
    """``path.open()`` with no encoding uses the locale code page.

    On Windows that is cp1252, and published screens are UTF-8: an undecodable
    byte raised a bare ``UnicodeDecodeError`` -- not the ``DatasetError`` the
    module documents -- so ``cli._train`` crashed with a traceback.
    """

    def test_a_utf8_screen_with_non_ascii_labels_loads(self):
        from cas13guide.services.ingest import load_screen

        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        rows = ["spacer,activity,transcript"]
        for i in range(40):
            rows.append(f"{spacer},0.5,gène_ENST{i % 8:05d}Á")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.csv"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            examples, report = load_screen(path)
        self.assertEqual(len(examples), 40)
        self.assertTrue(any("è" in group for group in {e.group for e in examples}))

    def test_a_blank_transcript_is_not_reported_as_an_activity_problem(self):
        from cas13guide.services.ingest import load_screen

        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        rows = ["spacer,activity,transcript"]
        for i in range(30):
            rows.append(f"{spacer},0.5,ENST{i % 8:05d}")
        for _ in range(5):
            rows.append(f"{spacer},0.5,")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.csv"
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            _examples, report = load_screen(path)
        warnings = " ".join(report.warnings)
        self.assertIn("blank", warnings)
        self.assertNotIn("5 row(s) with missing or non-numeric activity", warnings)


class TestTheRunRecordsWhichSelfFoldSourceItUsed(unittest.TestCase):
    """Two comments promised "the run says which one it used"; nothing did.

    The choice decides SPACER_SELF_FOLDING, an unbuyable-back FAIL, and the two
    criteria disagree -- measured, 109 vs 102 FAILs on one 400 nt target -- yet
    the reports were identical in every field describing the method.
    """

    def test_the_ledger_names_the_fold_source(self):
        from cas13guide.services.workflow import DesignRequest, run_request

        outcome = run_request(DesignRequest(preset="hsa-miR-21-5p", use_structure=False))
        records = outcome.report["provenance"]["records"]
        entry = next(r for r in records if r.get("name") == "qc_self_fold")
        self.assertIn(entry["version"], ("qc/vienna-mfe/v1", "qc/stem-proxy/v1"))
        self.assertTrue(entry["detail"]["criterion"])

    def test_the_proxy_case_raises_a_limitation_and_the_real_one_does_not(self):
        from cas13guide.services.limitations import limitations_for_run

        proxy = {"records": [{"kind": "model", "name": "qc_self_fold",
                              "version": "qc/stem-proxy/v1"}]}
        real = {"records": [{"kind": "model", "name": "qc_self_fold",
                             "version": "qc/vienna-mfe/v1"}]}
        spec = make_spec()
        self.assertIn(
            "self-folding-is-a-proxy",
            {l.id for l in limitations_for_run(spec, provenance=proxy)},
        )
        self.assertNotIn(
            "self-folding-is-a-proxy",
            {l.id for l in limitations_for_run(spec, provenance=real)},
        )
        self.assertNotIn(
            "self-folding-is-a-proxy",
            {l.id for l in limitations_for_run(spec)},
        )


class TestAnyHitDoesWhatItSays(unittest.TestCase):
    """The FM index's ``any_hit`` delegated to a full ``search``.

    It enumerated every hit, profiled every duplex and sorted the result --
    the whole cost the seam exists to avoid, on the index chosen precisely
    because enumeration hurts, and paid up to ``attempts`` times per control.
    """

    def _index(self):
        from cas13guide.services.fmindex import FMIndexSearchIndex
        from cas13guide.services.transcriptome import Locus, Transcriptome

        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("L1", "AUGC" * 300, Universe.CODING, expression=1.0))
        return FMIndexSearchIndex(world, frozenset({Universe.CODING}))

    def test_it_answers_the_same_question_as_search(self):
        from cas13guide.util.seq import revcomp

        index = self._index()
        for spacer in (revcomp("AUGC" * 5 + "AU"), "A" * 22):
            with self.subTest(spacer=spacer):
                self.assertEqual(
                    index.any_hit(spacer, 4),
                    bool(index.search("g", spacer, 4).hits),
                )

    def test_it_builds_no_hits_on_the_way(self):
        import cas13guide.services.fmindex as fmindex
        from cas13guide.util.seq import revcomp

        index = self._index()
        calls = []
        original = fmindex.profile_duplex
        fmindex.profile_duplex = lambda *a, **k: calls.append(1) or original(*a, **k)
        try:
            self.assertTrue(index.any_hit(revcomp("AUGC" * 5 + "AU"), 4))
        finally:
            fmindex.profile_duplex = original
        self.assertEqual(calls, [], "any_hit profiled duplexes it never needed")


class TestThePanelGuaranteeTracksTheConstraints(unittest.TestCase):
    """``constrained`` was decided by ``len(assessments)``.

    The pool size has nothing to do with whether the diversity constraint can
    bind, so a large ``max_shared_off`` declared the run unconstrained and
    printed the ``1 - 1/e`` bound while ``_compatible`` was still rejecting
    pairs -- the claim fcb1bd8 removed from the other branch.
    """

    def _guide(self, guide_id, coverage, start, off_loci):
        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.metrics.bounds import Interval
        from cas13guide.models.evidence import (
            Candidate,
            GateResult,
            GuideAssessment,
            MetricVector,
            PairEvidence,
        )

        spacer = "GCAUUGCUGUGGAGCUGAAGUC"
        evidence = [
            PairEvidence(
                guide_id=guide_id, locus_id="L_t1", context="reference",
                hit=dataclasses.replace(
                    make_hit(guide_id, "L_t1", "t1", TargetClass.REQUIRED), start=start
                ),
                activity=Interval(coverage, coverage, min(1.0, coverage + 0.05)),
                expression=Interval(0.5, 0.5, 0.5), hazard_weight=0.0,
            )
        ]
        for locus in off_loci:
            evidence.append(
                PairEvidence(
                    guide_id=guide_id, locus_id=locus, context="reference",
                    hit=make_hit(guide_id, locus),
                    activity=Interval(0.05, 0.0, 0.1),
                    expression=Interval(0.2, 0.1, 0.3), hazard_weight=0.5,
                )
            )
        return GuideAssessment(
            guide_id=guide_id,
            candidate=Candidate(guide_id=guide_id, spacer=spacer,
                                source_target_id="t1", start=start),
            qc=clean_qc(guide_id),
            metrics=MetricVector(coverage, 0.0, 0.05, 0.05, coverage, 1.0),
            gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
            evidence=tuple(evidence),
        )

    def test_a_diversity_constraint_that_binds_forfeits_the_bound(self):
        from cas13guide.pipeline.optimizer import select_panel

        shared = [f"off{i}" for i in range(6)]
        pool = [
            self._guide("gPan", 0.40, 0, shared),
            self._guide("gVar1", 0.99, 100, shared[:6]),
            self._guide("gVar2", 0.99, 200, shared[:6]),
        ]
        panel = select_panel(pool, make_spec(), size=3, min_spacing=0, max_shared_off=5)
        self.assertNotIn("1 - 1/e", panel.guarantee)

    def test_a_genuinely_unconstrained_run_still_reports_the_bound(self):
        from cas13guide.pipeline.optimizer import select_panel

        pool = [self._guide("gA", 0.9, 0, ()), self._guide("gB", 0.8, 100, ())]
        panel = select_panel(pool, make_spec(), size=2, min_spacing=0, max_shared_off=0)
        self.assertIn("1 - 1/e", panel.guarantee)


class TestReportedStatisticsAreWhatTheyAreCalled(unittest.TestCase):
    """``coverage_median`` was the upper-middle order statistic.

    For an even count it is biased upward by half the central gap, and at n = 2
    it equals ``coverage_max`` -- in the one field a reader consults to judge
    whether the activity model discriminated at all.
    """

    def _summary(self, coverages):
        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.models.evidence import (
            Candidate,
            GateResult,
            GuideAssessment,
            MetricVector,
        )
        from cas13guide.pipeline.g_output import summarize_metrics

        assessments = []
        for i, value in enumerate(coverages):
            assessments.append(
                GuideAssessment(
                    guide_id=f"g{i}",
                    candidate=Candidate(guide_id=f"g{i}", spacer="A" * 22,
                                        source_target_id="t1", start=i),
                    qc=clean_qc(f"g{i}"),
                    metrics=MetricVector(value, 0.0, 0.0, 0.0, value, 1.0),
                    gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
                )
            )
        return summarize_metrics(assessments)

    def test_an_even_count_reports_the_median(self):
        self.assertAlmostEqual(self._summary([0.10, 0.20, 0.80, 0.90]).coverage_median, 0.50)

    def test_two_guides_do_not_report_the_maximum_as_the_median(self):
        summary = self._summary([0.0, 1.0])
        self.assertAlmostEqual(summary.coverage_median, 0.5)
        self.assertNotAlmostEqual(summary.coverage_median, summary.coverage_max)

    def test_an_odd_count_is_unchanged(self):
        self.assertAlmostEqual(self._summary([0.1, 0.4, 0.9]).coverage_median, 0.4)


class TestTheReportFilterCanFindWhatItAdvertises(unittest.TestCase):
    """The filter promised "sequence, id, reason" and matched textContent.

    Neither guide_id nor reasons is rendered as a cell, so typing a reason code
    from the CSV emptied the table -- which reads as "no guide has that
    reason".
    """

    def _render(self):
        from cas13guide.services.report_html import render_report

        return render_report(
            {
                "verdict_counts": {"REVIEW": 1},
                "ranked": [
                    {
                        "rank": 1, "guide_id": "g_ab12cd", "spacer": "AUGC" * 5,
                        "score": 7.0, "verdict": "REVIEW", "distinct_site": True,
                        "reasons": ["expression_unknown"],
                        "qc_warnings": ["pfs_disfavoured"],
                        "metrics": {},
                    }
                ],
            }
        )

    def test_the_guide_id_is_searchable(self):
        self.assertIn("g_ab12cd", self._render())

    def test_the_reason_code_is_searchable(self):
        html = self._render()
        self.assertIn("expression_unknown", html)
        self.assertIn("data-search=", html)

    def test_the_filter_reads_that_attribute(self):
        self.assertIn("data-search", self._render().split("<script>")[-1])


class TestPanelConfidenceNotesMatchWhatWasReported(unittest.TestCase):
    """With a measured beta the report drew a point marker and the text under
    it said "reported as an interval, not a number"."""

    def _confidence(self, dependence):
        from cas13guide.pipeline.optimizer import Panel
        from cas13guide.services.calibration import CalibrationService
        from cas13guide.services.panel_confidence import BayesianPanelConfidence
        from test_panel_confidence import calibration_for, guide

        spec = make_spec(
            targets=(
                TargetSpec(target_id="t1", sequence=SEQ_A, target_class=TargetClass.REQUIRED),
            )
        )
        panel = Panel(
            guides=(guide("gA", {"t1": 0.9}), guide("gB", {"t1": 0.9}, start=100)),
            coverage={}, objective=0.0, guarantee="test",
        )
        return BayesianPanelConfidence(
            calibration_for(spec), spec, dependence=dependence
        ).for_panel(panel)

    def test_without_a_measured_beta_the_notes_say_interval(self):
        text = " ".join(self._confidence(None).notes).lower()
        self.assertIn("interval, not a number", text)

    def test_with_a_measured_beta_they_do_not(self):
        from cas13guide.services.panel_confidence import PanelDependence

        result = self._confidence(
            PanelDependence(0.2, source="replicate panels", provisional=False)
        )
        text = " ".join(result.notes).lower()
        self.assertTrue(result.has_point_estimates)
        self.assertNotIn("interval, not a number", text)
        self.assertNotIn("nothing here can tell you where between them", text)
        self.assertIn("point estimate", text)

    def test_a_provisional_beta_keeps_the_interval_wording(self):
        from cas13guide.services.panel_confidence import PanelDependence

        text = " ".join(self._confidence(PanelDependence(0.2)).notes).lower()
        self.assertIn("interval, not a number", text)


if __name__ == "__main__":
    unittest.main()
