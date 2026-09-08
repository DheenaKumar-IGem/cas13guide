"""Second review pass: regressions in the first pass, and angles it did not use.

Round one read module by module. This round re-read the fix itself -- the
least-reviewed code in the repository the moment it landed -- and swept three
angles round one never took: concurrency through the threaded dev server,
numeric edges, and cross-module disagreement. Four defects survived two
skeptics, and one of them was a regression the first pass introduced.
"""

import json
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import Universe


class TestBlastAddsACorpusRatherThanReplacingTheSearch(unittest.TestCase):
    """``--blast-db`` swapped the whole index, including the on-target search.

    One index answers both of the pipeline's questions -- where does this guide
    bind off-target, and does it bind its intended target at all -- and a BLAST
    database can only answer the first: it never sees the run's own
    ``Transcriptome``, and its subject ids are accessions that ``annotate_hit``
    cannot match to the run's target ids. Substituting it left every guide with
    no on-target evidence, so the flag FAILed everything: measured 39/39 FAIL
    against 28 REVIEW for the identical design without it. A flag that fails
    every guide is indistinguishable from a target that has none.

    Pre-existing, and the first pass widened it: handing that same index to
    every variant context meant no variant sequence was searched either, so
    ``robustness`` -- the whole point of ``--variants`` -- read 0.000 for every
    guide.
    """

    def _world(self):
        from cas13guide.services.transcriptome import Locus, Transcriptome

        world = Transcriptome("GRCh38", "v44")
        world.add(Locus("T1", SEQ_A, Universe.NCRNA, gene="T1", expression=10.0))
        return world

    class _Empty:
        """An index that finds nothing -- a database without this target in it."""

        version = "index/stub-empty/v1"

        def search(self, guide_id, spacer, max_mismatches):
            from cas13guide.services.indexes import ExhaustivenessBound, SearchResult

            return SearchResult(
                hits=(),
                bound=ExhaustivenessBound(
                    max_mismatches=max_mismatches,
                    wobble_aware=False,
                    universes=frozenset({Universe.CODING}),
                    exhaustive=False,
                    method=self.version,
                ),
            )

    def _spacer(self):
        from cas13guide.util.seq import revcomp

        return revcomp(SEQ_A[10:32])

    def test_the_union_still_finds_the_on_target_site(self):
        from cas13guide.services.indexes import ExhaustiveScanIndex, UnionSearchIndex

        local = ExhaustiveScanIndex(self._world(), frozenset(Universe))
        union = UnionSearchIndex([local, self._Empty()])
        expected = [(h.locus_id, h.start) for h in local.search("g", self._spacer(), 2).hits]
        self.assertTrue(expected, "fixture must produce a hit to be a test")
        self.assertEqual(
            [(h.locus_id, h.start) for h in union.search("g", self._spacer(), 2).hits],
            expected,
        )

    def test_the_union_reports_the_weakest_bound(self):
        """A heuristic member makes the whole search heuristic."""
        from cas13guide.services.indexes import ExhaustiveScanIndex, UnionSearchIndex

        local = ExhaustiveScanIndex(self._world(), frozenset(Universe))
        self.assertTrue(local.search("g", self._spacer(), 2).bound.exhaustive)
        union = UnionSearchIndex([local, self._Empty()])
        self.assertFalse(union.search("g", self._spacer(), 2).bound.exhaustive)

    def test_the_union_takes_the_union_of_universes(self):
        from cas13guide.services.indexes import ExhaustiveScanIndex, UnionSearchIndex

        local = ExhaustiveScanIndex(self._world(), frozenset({Universe.NCRNA}))
        union = UnionSearchIndex([local, self._Empty()])
        bound = union.search("g", self._spacer(), 2).bound
        self.assertIn(Universe.NCRNA, bound.universes)
        self.assertIn(Universe.CODING, bound.universes)

    def test_a_shared_locus_is_not_counted_twice(self):
        """``cumulative_burden`` is a sum, so a repeated locus inflates hazard."""
        from cas13guide.services.indexes import ExhaustiveScanIndex, UnionSearchIndex

        local = ExhaustiveScanIndex(self._world(), frozenset(Universe))
        once = local.search("g", self._spacer(), 2).hits
        twice = UnionSearchIndex([local, local]).search("g", self._spacer(), 2).hits
        self.assertEqual(len(once), len(twice))

    def test_an_empty_union_is_refused(self):
        from cas13guide.services.indexes import UnionSearchIndex

        with self.assertRaises(ValueError):
            UnionSearchIndex([])

    def test_any_hit_short_circuits_across_members(self):
        from cas13guide.services.indexes import ExhaustiveScanIndex, UnionSearchIndex

        union = UnionSearchIndex(
            [self._Empty(), ExhaustiveScanIndex(self._world(), frozenset(Universe))]
        )
        self.assertTrue(union.any_hit(self._spacer(), 2))
        self.assertFalse(union.any_hit("A" * 22, 0))

    def test_the_workflow_unions_rather_than_substitutes(self):
        import cas13guide.services.blast as blast
        from cas13guide.services.indexes import UnionSearchIndex
        from cas13guide.services.workflow import DesignRequest, _reference_index

        class _Available(blast.BlastOffTargetIndex):
            def ensure_available(self):
                return None

        original = blast.BlastOffTargetIndex
        blast.BlastOffTargetIndex = _Available
        try:
            index = _reference_index(
                DesignRequest(blast_db="/db", blast_universes="coding"), self._world()
            )
        finally:
            blast.BlastOffTargetIndex = original
        self.assertIsInstance(index, UnionSearchIndex)
        self.assertEqual(len(index.indexes), 2)

    def test_without_a_database_nothing_is_wrapped(self):
        """The default path must stay exactly what it was."""
        from cas13guide.services.indexes import ExhaustiveScanIndex
        from cas13guide.services.workflow import DesignRequest, _reference_index

        index = _reference_index(DesignRequest(), self._world())
        self.assertIsInstance(index, ExhaustiveScanIndex)


class TestTheDegeneracyProbesAreNotRepaidPerGuide(unittest.TestCase):
    """``adds_information`` is a property of the constants, not of a candidate.

    ``build_metric_vector`` calls it once per candidate, so eight extra ODE
    integrations were paid per guide -- nine in total where the comment beside
    the step cap said one. Invisible at a fixed 400 steps, and not once the
    step count follows the constants.
    """

    def test_the_answer_is_memoised(self):
        from cas13guide.services.kinetics import CollateralKineticsModel

        model = CollateralKineticsModel()
        calls = []
        # Counted on the class: the model uses __slots__, so the instance
        # cannot carry a patched bound method.
        original = CollateralKineticsModel.integrate

        def counting(self, *args, **kwargs):
            calls.append(1)
            return original(self, *args, **kwargs)

        CollateralKineticsModel.integrate = counting
        try:
            first = model.adds_information()
            after_first = len(calls)
            second = model.adds_information()
        finally:
            CollateralKineticsModel.integrate = original
        self.assertEqual(first, second)
        self.assertGreater(after_first, 0, "the probes must run at least once")
        self.assertEqual(len(calls), after_first, "the second call re-integrated")

    def test_different_constants_get_their_own_answer(self):
        """Memoising on the wrong key would freeze the first model's verdict."""
        import dataclasses

        from cas13guide.services.kinetics import (
            PROVISIONAL_CONSTANTS,
            CollateralKineticsModel,
        )

        first = CollateralKineticsModel()
        second = CollateralKineticsModel(
            dataclasses.replace(PROVISIONAL_CONSTANTS, target_nm=500.0)
        )
        first.adds_information()
        self.assertIsInstance(second.adds_information(), bool)
        self.assertEqual(
            first._degeneracy_cache.keys() & second._degeneracy_cache.keys(), set()
        )

    def test_the_verdict_is_unchanged_by_memoising_it(self):
        from cas13guide.services.kinetics import CollateralKineticsModel

        self.assertFalse(CollateralKineticsModel().adds_information())


class TestPanelDependenceFailsWithTheDocumentedType(unittest.TestCase):
    """A quoted number in the JSON raised TypeError, not DesignRequestError.

    ``PanelDependence.__post_init__`` compares the value against 0.0 and 1.0,
    which raises TypeError for a string, and ``_panel_dependence`` caught only
    OSError and ValueError -- so the traceback reached the user through the
    flag added two commits earlier, while its sibling loader for
    ``--kinetics-constants`` already caught all three.
    """

    def test_a_non_numeric_beta_is_a_located_message(self):
        from cas13guide.services.workflow import (
            DesignRequest,
            DesignRequestError,
            run_request,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beta.json"
            path.write_text(
                json.dumps({"common_cause_beta": "0.18", "provisional": False}),
                encoding="utf-8",
            )
            with self.assertRaises(DesignRequestError) as caught:
                run_request(
                    DesignRequest(preset="hsa-miR-21-5p", panel_dependence_json=str(path))
                )
        self.assertIn("--panel-dependence", str(caught.exception))


class TestTheSpecSaysWhyItIsUnhashable(unittest.TestCase):
    """The replacement docstring gave two reasons and neither was the reason.

    Pinned from both sides, because the promise is one tidy-up away from being
    false: making the hazard maps hashable would silently make the whole spec
    hashable, and nothing would notice.
    """

    def test_hashing_the_spec_raises(self):
        with self.assertRaises(TypeError):
            hash(make_spec())

    def test_the_hazard_maps_are_the_only_obstacle(self):
        spec = make_spec()
        for name in ("contract", "profile", "context", "thresholds", "budget"):
            with self.subTest(field=name):
                self.assertIsInstance(hash(getattr(spec, name)), int)
        with self.assertRaises(TypeError):
            hash(spec.hazard)

    def test_the_docstring_names_that_obstacle(self):
        from cas13guide.contract.spec import ResolvedDesignSpec

        self.assertIn("HazardWeights", ResolvedDesignSpec.__doc__)
        self.assertIn("fingerprint", ResolvedDesignSpec.__doc__)


if __name__ == "__main__":
    unittest.main()
