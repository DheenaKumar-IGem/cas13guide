"""Fourth review pass, and the one that stopped the count converging.

Passes one to three read the code: module by module, then the diff, then the
newest diff. This pass ran the *product* instead -- combinations of CLI flags,
the multi-record path end to end, and the README audited claim by claim against
what the code does. It found six, up from two, which says something worth
recording: reading a module tells you whether it is self-consistent, and only
running the thing tells you whether two modules agree.

Three of the six are the same shape as defects earlier passes fixed, surviving
in a branch the fix did not reach.
"""

import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass
from cas13guide.contract.spec import TargetSpec


def _sequence(seed_value: int, length: int) -> str:
    """Non-repeating, so tiled windows are distinct rather than deduplicated."""
    import random

    rng = random.Random(seed_value)
    return "".join(rng.choice("ACGU") for _ in range(length))


class TestBadInputAlwaysArrivesAsDesignRequestError(unittest.TestCase):
    """``run_request`` documents one exception type and raised three.

    ``_read_fasta_or_fail`` was added in pass one to close exactly this hole,
    and closed it for ``--background`` and ``--variants`` while leaving the
    preset branch four lines above it open. ``build_spec`` had the same split
    inside a single function: the ``ResolvedDesignSpec`` construction was
    guarded and the ``TargetSpec`` comprehension above it was not.
    """

    def _message(self, **kw):
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            run_request,
        )

        with self.assertRaises(DesignRequestError) as caught:
            run_request(DesignRequest(**kw))
        return str(caught.exception)

    def test_a_mistyped_preset_names_the_available_ones(self):
        message = self._message(preset="hsa-miR-21")
        self.assertIn("unknown preset", message)
        self.assertIn("hsa-miR-21-5p", message)
        self.assertNotIn('"', message[:20], "KeyError repr quoting leaked through")

    def test_a_negative_expression_is_a_located_message(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.fa"
            path.write_text(f">t1\n{_sequence(1, 200)}\n", encoding="utf-8")
            message = self._message(fasta_path=str(path), expression=-5.0)
        self.assertIn("cannot build the design spec", message)
        self.assertIn("greater than or equal to 0", message)

    def test_the_neighbouring_rejection_still_works(self):
        """The guard must not have swallowed the one that already worked."""
        message = self._message(preset="hsa-miR-21-5p", spacer_length=30)
        self.assertIn("spacer length", message.lower())


class TestBackgroundCannotRedefineATarget(unittest.TestCase):
    """``Transcriptome.add`` overwrites by ``locus_id``.

    Background loci are added after the target records, so a background record
    sharing a target's name replaced that target's sequence, biotype and
    declared expression -- while ``locus_to_target`` still pointed at it, so the
    run searched the target against something that was not it. Measured:
    PASS 97 to 0, best required_coverage 0.7647 to 0.0, with nothing said.

    The route is ordinary: a background corpus pulled from the same annotation
    contains the target's own transcript id. ``variant_contexts`` already
    guarded this, so the two paths disagreed -- and pass three's whole point was
    that adding ``--variants`` must not move the other numbers.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        self.primary = _sequence(3, 300)
        self.target = root / "t.fa"
        self.target.write_text(f">t1\n{self.primary}\n", encoding="utf-8")
        self.colliding = root / "bg_same_name.fa"
        self.colliding.write_text(f">t1\n{_sequence(4, 300)}\n", encoding="utf-8")
        self.clean = root / "bg.fa"
        self.clean.write_text(f">decoy1\n{_sequence(4, 300)}\n", encoding="utf-8")

    def tearDown(self):
        self.dir.cleanup()

    def _run(self, background):
        from cas13guide.services.workflow import DesignRequest, run_request

        return run_request(
            DesignRequest(fasta_path=str(self.target), background_path=str(background),
                          task="knockdown", assay="cell_culture_knockdown",
                          effector="RfxCas13d", use_structure=False, use_signal=False)
        )

    @staticmethod
    def _best_coverage(outcome):
        return max(
            (row["metrics"]["required_coverage"] for row in outcome.report["ranked"]),
            default=0.0,
        )

    def test_the_target_keeps_its_own_sequence(self):
        colliding = self._run(self.colliding)
        clean = self._run(self.clean)
        self.assertGreater(self._best_coverage(clean), 0.0, "fixture sanity")
        self.assertAlmostEqual(
            self._best_coverage(colliding), self._best_coverage(clean), places=6,
            msg="a background record displaced the target locus",
        )

    def test_the_skip_is_stated_rather_than_silent(self):
        outcome = self._run(self.colliding)
        self.assertTrue(
            [note for note in outcome.notes if "shares a name" in note],
            "the collision was resolved silently",
        )

    def test_a_background_record_with_its_own_name_is_still_loaded(self):
        """The guard must skip collisions, not the whole background corpus."""
        outcome = self._run(self.clean)
        loci = [
            record["detail"].get("loci")
            for record in outcome.report["provenance"]["records"]
            if record["kind"] == "transcriptome"
        ]
        self.assertEqual(loci, [2])


class TestDistinctSitesAreComparedWhereGuidesBind(unittest.TestCase):
    """``distinct_site_ids`` keyed on the record a guide was tiled from.

    That is not the same question as where it binds. With more than one target
    record, a guide tiled from the second carries that record's coordinate and
    was never compared against anything from the first -- so two guides whose
    sites on the REQUIRED target sit one nucleotide apart were both marked
    distinct, which is the exact failure the function exists to prevent.
    """

    def _ranked(self, sites_per_guide):
        import dataclasses

        from cas13guide.contract.enums import ReasonCode, Verdict
        from cas13guide.metrics.bounds import Interval
        from cas13guide.models.evidence import (
            Candidate,
            GateResult,
            GuideAssessment,
            MetricVector,
            PairEvidence,
        )
        from cas13guide.pipeline.f_decide import RankedGuide

        ranked = []
        for rank, (guide_id, source, tiled_start, sites) in enumerate(sites_per_guide, 1):
            evidence = tuple(
                PairEvidence(
                    guide_id=guide_id,
                    locus_id=f"L_{target_id}",
                    context="reference",
                    hit=dataclasses.replace(
                        make_hit(guide_id, f"L_{target_id}", target_id,
                                 TargetClass.REQUIRED),
                        start=start,
                    ),
                    activity=Interval(0.9, 0.9, 0.95),
                    expression=Interval(0.5, 0.5, 0.5),
                    hazard_weight=0.0,
                )
                for target_id, start in sites.items()
            )
            assessment = GuideAssessment(
                guide_id=guide_id,
                candidate=Candidate(guide_id=guide_id, spacer="A" * 22,
                                    source_target_id=source, start=tiled_start),
                qc=clean_qc(guide_id),
                metrics=MetricVector(0.9, 0.0, 0.0, 0.0, 0.9, 1.0),
                gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
                evidence=evidence,
            )
            ranked.append(RankedGuide(rank=rank, assessment=assessment, score=1.0,
                                      rationale=(), sensitivity={}))
        return ranked

    def test_two_records_overlapping_on_the_primary_are_one_site(self):
        from cas13guide.pipeline.f_decide import distinct_site_ids

        ranked = self._ranked([
            ("gA", "recA", 1046, {"recA": 1046}),
            # Tiled from recB at 107, but binding recA one nucleotide along.
            ("gB", "recB", 107, {"recA": 1047, "recB": 107}),
        ])
        self.assertEqual(distinct_site_ids(ranked, 22), frozenset({"gA"}))

    def test_genuinely_separate_sites_stay_distinct(self):
        from cas13guide.pipeline.f_decide import distinct_site_ids

        ranked = self._ranked([
            ("gA", "recA", 100, {"recA": 100}),
            ("gB", "recB", 107, {"recA": 900, "recB": 107}),
        ])
        self.assertEqual(distinct_site_ids(ranked, 22), frozenset({"gA", "gB"}))

    def test_a_guide_with_no_intended_evidence_falls_back_to_its_tiled_site(self):
        """The only information there is about it, and better than dropping it."""
        from cas13guide.pipeline.f_decide import distinct_site_ids

        ranked = self._ranked([("gA", "recA", 10, {})])
        self.assertEqual(distinct_site_ids(ranked, 22), frozenset({"gA"}))


class TestTheExportsCarryTheDistinctSiteFlag(unittest.TestCase):
    """The README says they do: "nothing is dropped from the data; the exports
    carry the flag". Neither file had it, so an exported table of sixty guides
    that is really two sites and fifty-eight shifts of them said nothing."""

    REPORT = {
        "ranked": [
            {"rank": 1, "guide_id": "g1", "spacer": "AUGCAUGCAUGCAUGCAUGCAU",
             "score": 7.0, "verdict": "PASS", "distinct_site": True, "metrics": {}},
            {"rank": 2, "guide_id": "g2", "spacer": "UGCAUGCAUGCAUGCAUGCAUG",
             "score": 6.9, "verdict": "PASS", "distinct_site": False, "metrics": {}},
        ]
    }

    def test_the_csv_has_the_column(self):
        from cas13guide.services.fasta import RANKED_COLUMNS, ranked_to_csv

        self.assertIn("distinct_site", RANKED_COLUMNS)
        text = ranked_to_csv(self.REPORT)
        rows = [line for line in text.splitlines() if not line.startswith("#")]
        header = rows[0].split(",")
        self.assertIn("distinct_site", header)
        column = header.index("distinct_site")
        self.assertEqual(
            [row.split(",")[column] for row in rows[1:]], ["True", "False"]
        )

    def test_the_column_is_appended_so_positional_parsers_do_not_shift(self):
        from cas13guide.services.fasta import RANKED_COLUMNS

        self.assertEqual(RANKED_COLUMNS[-1], "distinct_site")

    def test_the_fasta_defline_carries_it(self):
        from cas13guide.services.fasta import parse_fasta, ranked_to_fasta

        records = parse_fasta(ranked_to_fasta(self.REPORT))
        self.assertIn("distinct_site=True", records[0].description)
        self.assertIn("distinct_site=False", records[1].description)

    def test_a_report_without_the_flag_still_exports(self):
        from cas13guide.services.fasta import ranked_to_fasta

        stripped = {"ranked": [dict(self.REPORT["ranked"][0])]}
        stripped["ranked"][0].pop("distinct_site")
        self.assertNotIn("distinct_site", ranked_to_fasta(stripped))


class TestTheCandidateBudgetIsSpentRatherThanRoundedAway(unittest.TestCase):
    """Widening the step to an integer overshoots badly just above 1.

    ``ceil(2973 / 2500) = 2`` keeps 1487 windows against an allowance of 2500 --
    40% of the share unspent and the target tiled at half the density the budget
    allowed, for nothing. The docstring promised the share was "spent".
    """

    def _spec(self, n_targets, length=3000):
        targets = tuple(
            TargetSpec(target_id=f"t{i + 1}", sequence=_sequence(i, length),
                       target_class=TargetClass.REQUIRED)
            for i in range(n_targets)
        )
        return make_spec(targets=targets)

    def test_two_targets_use_essentially_the_whole_budget(self):
        from cas13guide.pipeline.a_candidates import design_candidates

        spec = self._spec(2)
        candidates = design_candidates(spec)
        budget = spec.budget.max_candidates
        self.assertLessEqual(len(candidates), budget)
        self.assertGreaterEqual(
            len(candidates), int(budget * 0.98),
            "the split rounded most of the budget away",
        )

    def test_each_target_is_still_spanned_end_to_end(self):
        from cas13guide.pipeline.a_candidates import design_candidates

        spec = self._spec(3)
        candidates = design_candidates(spec)
        for target in spec.contract.targets:
            starts = [c.start for c in candidates if c.source_target_id == target.target_id]
            with self.subTest(target=target.target_id):
                self.assertEqual(min(starts), 0)
                self.assertEqual(
                    max(starts), len(target.sequence) - spec.spacer_length,
                    "the last window of the transcript was dropped",
                )

    def test_a_single_target_is_tiled_at_full_density(self):
        """One target that fits inside the budget must not be thinned at all."""
        from cas13guide.pipeline.a_candidates import design_candidates

        spec = self._spec(1)
        expected = len(spec.contract.targets[0].sequence) - spec.spacer_length + 1
        self.assertEqual(len(design_candidates(spec)), expected)

    def test_the_split_is_deterministic(self):
        from cas13guide.pipeline.a_candidates import design_candidates

        spec = self._spec(3)
        first = [c.guide_id for c in design_candidates(spec)]
        self.assertEqual(first, [c.guide_id for c in design_candidates(spec)])


if __name__ == "__main__":
    unittest.main()
