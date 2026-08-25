"""LinearFold-CONTRAfold scores, held to the reference and to LinearFold's defaults.

Two independent things have to be right and only one of them the Arc fixture can
check.

The fixture checks the model. 133 guides x 3 folds = 399 known answers, and this
reproduces every one exactly, then drives the real CNN to within 5.96e-08 of the
reference scores.

The fixture cannot check the sharp-turn convention. LinearFold defaults to
``sharpturn=false`` -- no hairpin closed on fewer than three unpaired bases --
and CONTRAfold allows them. Measured, **zero** of the fixture's 266 sequences
contain one, so a parity suite built only from the fixture passes green with the
convention set either way. It is not rare in general: 4.0% of random 66-mers,
9.7% of GC-rich ones, and 2.0% of Arc-shaped guide constructs differ, by up to
3.22. So the vectors below are the other half of the gate, and they exist
precisely because the fixture could not supply them.
"""

import json
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.services.arc_model import STANDARD_PREFIX, TARGET_LENGTH

try:
    import gpu_contrafold  # noqa: F401

    HAVE_CONTRAFOLD = True
except ImportError:  # pragma: no cover
    HAVE_CONTRAFOLD = False

needs_contrafold = unittest.skipUnless(
    HAVE_CONTRAFOLD, 'gpu-contrafold not installed (pip install "cas13guide[contrafold]")'
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "arc_reference_pipeline.json"

#: (sequence, no-sharp-turn score, sharp-turn-allowed score). Every one of these
#: differs between the two conventions -- that is why they are here. The fixture
#: contains no such sequence.
SHARP_TURN_VECTORS = (
    ("AGCGGCGUGCCCGGCCACGCACUCCUCGGGGGGUGGCAGCCGGCACAACGUAAUUUGCUACCCCGC", 10.12, 10.41),
    ("GGAUGUUUCGAGGGGAGUUCGUCAUCUGGGCAAGCCGAGGACCCCUGCAGAGGCUUCGGGCUUCGC", 2.71, 3.06),
    ("CUUUGCCCCGACAUUGUCGCCGGCAGGUGACGACAGUCAGUCACCUCGUGCAGUACUGCCCCGGGC", 5.98, 7.59),
    ("AAUGGUGGCGCUGCUUCUAGCAGCCCCUAUUUUCAGUUGCUAGCGGUACGAGUGACGGGCCCGGUC", 4.33, 5.05),
)


def reference_guides():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["guides"]


def constraint_for(guide):
    """The mask arc_model builds: the 30 nt target forced unpaired in its flanks."""
    nearby = guide["nearby"]
    offset = nearby.find(guide["target"])
    tail = len(nearby) - offset - TARGET_LENGTH
    return "?" * offset + "." * TARGET_LENGTH + "?" * tail


@needs_contrafold
class TestReferenceParity(unittest.TestCase):
    """399 known answers. This is the gate on the model."""

    @classmethod
    def setUpClass(cls):
        from cas13guide.services.contrafold import ContrafoldMFESource

        cls.source = ContrafoldMFESource()
        cls.guides = reference_guides()

    def test_every_guide_fold_is_exact(self):
        for guide in self.guides:
            with self.subTest(spacer=guide["spacer"]):
                self.assertEqual(
                    self.source.mfe(STANDARD_PREFIX + guide["spacer"]), guide["guideMfe"]
                )

    def test_every_native_target_fold_is_exact(self):
        for guide in self.guides:
            with self.subTest(spacer=guide["spacer"]):
                self.assertEqual(self.source.mfe(guide["nearby"]), guide["nativeMfe"])

    def test_every_constrained_target_fold_is_exact(self):
        for guide in self.guides:
            with self.subTest(spacer=guide["spacer"]):
                self.assertEqual(
                    self.source.mfe_constrained(guide["nearby"], constraint_for(guide)),
                    guide["constrainedMfe"],
                )


@needs_contrafold
class TestEndToEndThroughTheRealModel(unittest.TestCase):
    """Folding is only correct if the thing downstream of it comes out right.

    The layer that existed before this drove the CNN from the fixture's
    *precomputed* features, so it could not have caught a wrong folding source
    at all. This one folds live and compares the CNN's output.
    """

    @classmethod
    def setUpClass(cls):
        import numpy as np

        from cas13guide.services.arc_model import ArcEnsemble, features_for_target
        from cas13guide.services.contrafold import ContrafoldMFESource

        try:
            cls.ensemble = ArcEnsemble()
        except Exception as exc:  # onnxruntime or weights missing
            raise unittest.SkipTest(f"Arc model unavailable: {exc}")

        source = ContrafoldMFESource()
        cls.guides = reference_guides()
        cls.np = np
        cls.spacers = [g["spacer"] for g in cls.guides]
        cls.features = [
            features_for_target(
                g["spacer"], g["nearby"], g["nearby"].find(g["target"]),
                source, g["isoformPercent"],
            )
            for g in cls.guides
        ]

    def test_derived_features_match_the_reference(self):
        for guide, feature in zip(self.guides, self.features):
            with self.subTest(spacer=guide["spacer"]):
                self.assertAlmostEqual(
                    feature.linearfold_val, guide["linearfoldVal"], places=12
                )
                self.assertAlmostEqual(
                    feature.target_unfold_energy, guide["targetUnfoldEnergy"], places=12
                )

    def test_logits_match_the_reference(self):
        got = self.ensemble.logits(self.spacers, self.features)
        want = self.np.array([g["logit"] for g in self.guides])
        self.assertLess(float(self.np.abs(got - want).max()), 1e-6)

    def test_scores_match_the_reference(self):
        got = self.ensemble.scores(self.spacers, self.features)
        want = self.np.array([g["score"] for g in self.guides])
        self.assertLess(float(self.np.abs(got - want).max()), 1e-6)


@needs_contrafold
class TestSharpTurnConvention(unittest.TestCase):
    """The half of the gate the fixture cannot provide."""

    @classmethod
    def setUpClass(cls):
        from cas13guide.services.contrafold import ContrafoldMFESource

        cls.linearfold = ContrafoldMFESource()               # sharpturn=False
        cls.contrafold = ContrafoldMFESource(sharpturn=True)  # CONTRAfold default

    def test_the_default_is_linearfolds_default(self):
        from cas13guide.services.contrafold import ContrafoldMFESource

        self.assertFalse(ContrafoldMFESource().sharpturn)

    def test_known_answers_under_no_sharp_turn(self):
        for sequence, no_sharp, _allowed in SHARP_TURN_VECTORS:
            with self.subTest(sequence=sequence[:20]):
                self.assertEqual(self.linearfold.mfe(sequence), no_sharp)

    def test_known_answers_with_sharp_turns_allowed(self):
        for sequence, _no_sharp, allowed in SHARP_TURN_VECTORS:
            with self.subTest(sequence=sequence[:20]):
                self.assertEqual(self.contrafold.mfe(sequence), allowed)

    def test_every_vector_actually_distinguishes_the_two(self):
        """Otherwise these are decoration, not a test."""
        for sequence, no_sharp, allowed in SHARP_TURN_VECTORS:
            with self.subTest(sequence=sequence[:20]):
                self.assertNotEqual(no_sharp, allowed)

    def test_the_fixture_alone_would_not_catch_the_difference(self):
        """Pinned so nobody deletes the vectors above as redundant.

        If this ever fails, the fixture has gained a sharp-turn sequence and the
        parity suite got stronger -- but the vectors are still what makes the
        convention explicit.
        """
        differing = 0
        for guide in reference_guides():
            for sequence in (STANDARD_PREFIX + guide["spacer"], guide["nearby"]):
                rna = sequence.replace("T", "U")
                if self.linearfold.mfe(rna) != self.contrafold.mfe(rna):
                    differing += 1
        self.assertEqual(differing, 0)

    def test_the_versions_differ_so_provenance_records_which(self):
        self.assertNotEqual(self.linearfold.version, self.contrafold.version)
        self.assertIn("no-sharp-turn", self.linearfold.version)


@needs_contrafold
class TestTheContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cas13guide.services.contrafold import ContrafoldMFESource

        cls.source = ContrafoldMFESource()

    def test_it_satisfies_the_mfe_source_protocol(self):
        from cas13guide.services.arc_model import MFESource

        self.assertIsInstance(self.source, MFESource)

    def test_scores_are_rounded_to_two_decimals(self):
        """LinearFold printed %.2f and the Arc pipeline parsed that text.

        Not cosmetic: leaving the extra precision in shifts Arc logits by about
        2.9e-03, which fails the 1e-6 parity gate.
        """
        for guide in reference_guides()[:20]:
            value = self.source.mfe(STANDARD_PREFIX + guide["spacer"])
            with self.subTest(spacer=guide["spacer"]):
                self.assertEqual(value, round(value, 2))

    def test_dna_is_folded_as_rna(self):
        self.assertEqual(self.source.mfe("GGGGAAAACCCC"), self.source.mfe("GGGGAAAACCCC".replace("T", "U")))
        self.assertEqual(self.source.mfe("GGGGTTTTCCCC"), self.source.mfe("GGGGUUUUCCCC"))

    def test_a_mismatched_constraint_length_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.source.mfe_constrained("GGGGAAAACCCC", "....")
        self.assertIn("must match", str(caught.exception))

    def test_constraining_bases_unpaired_lowers_the_score(self):
        """Higher = more stable, so removing structure must reduce it."""
        sequence = "GGGGAAAACCCC"
        free = self.source.mfe(sequence)
        forced = self.source.mfe_constrained(sequence, "." * len(sequence))
        self.assertLess(forced, free)

    def test_an_unconstrained_mask_reproduces_the_free_fold(self):
        sequence = "GGGGAAAACCCC"
        self.assertEqual(
            self.source.mfe_constrained(sequence, "?" * len(sequence)),
            self.source.mfe(sequence),
        )


class TestItSaysWhatIsMissing(unittest.TestCase):
    """The refusal has to name the package and rule out the wrong substitute.

    Driven by actually breaking the import, not by constructing the exception
    the test then asserts about -- that would only prove the test can build a
    string.
    """

    def _import_fails(self):
        """Make `from gpu_contrafold import cpu` raise, the way absence does."""
        import sys
        from contextlib import contextmanager

        @contextmanager
        def blocked():
            saved = {k: v for k, v in sys.modules.items() if k.startswith("gpu_contrafold")}
            for name in saved:
                sys.modules[name] = None
            try:
                yield
            finally:
                sys.modules.update(saved)
                for name in [k for k in sys.modules if k.startswith("gpu_contrafold")]:
                    if sys.modules[name] is None:
                        del sys.modules[name]

        return blocked()

    def test_a_missing_package_raises_the_named_error(self):
        from cas13guide.services.contrafold import ContrafoldUnavailable, _cpu

        with self._import_fails():
            with self.assertRaises(ContrafoldUnavailable) as caught:
                _cpu()
        self.assertIsInstance(caught.exception, RuntimeError)

    def test_the_message_names_the_install_extra(self):
        from cas13guide.services.contrafold import ContrafoldUnavailable, _cpu

        with self._import_fails():
            with self.assertRaises(ContrafoldUnavailable) as caught:
                _cpu()
        message = str(caught.exception)
        self.assertIn("gpu-contrafold", message)
        self.assertIn("cas13guide[contrafold]", message)

    def test_the_message_rules_out_viennarna_by_name(self):
        """The substitution that would be confidently wrong rather than absent."""
        from cas13guide.services.contrafold import ContrafoldUnavailable, _cpu

        with self._import_fails():
            with self.assertRaises(ContrafoldUnavailable) as caught:
                _cpu()
        message = str(caught.exception)
        self.assertIn("ViennaRNA", message)
        self.assertIn("NOT a substitute", message)
        self.assertIn("Turner", message)

    def test_the_module_documents_the_sign_convention(self):
        """A caller who assumes kcal/mol supplies a sign-inverted number."""
        import cas13guide.services.contrafold as module

        doc = " ".join(module.__doc__.lower().split())
        self.assertIn("higher** means more stable", doc.replace("**higher", "higher**"))
        self.assertIn("not kcal/mol", doc)
