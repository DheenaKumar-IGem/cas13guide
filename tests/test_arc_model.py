"""The Arc RNAtargeting CNN, held to the reference implementation.

End-to-end parity is the gate. Four things in this adapter keep every tensor
shape valid and every score in range while being wrong -- a flipped one-hot
channel order, an ensemble averaged in the wrong space, a normalizing constant
applied to the wrong feature, and the three scalars in the wrong order -- and
none of them is visible to a per-component test. The 133-guide fixture is the
only check that sees all four.
"""

import json
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import Effector
from cas13guide.services.arc_model import (
    BASE_POSITIONS,
    GUIDE_LENGTH,
    STANDARD_PREFIX,
    ArcActivityModel,
    ArcEnsemble,
    ArcFeatures,
    ArcModelUnavailable,
    MFESourceRequired,
    features_for_target,
    guide_linearfold_val,
    one_hot,
    target_unfold_energy,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "arc_reference_pipeline.json"

try:
    import onnxruntime  # noqa: F401

    HAVE_ORT = True
except ImportError:  # pragma: no cover
    HAVE_ORT = False

needs_ort = unittest.skipUnless(HAVE_ORT, "onnxruntime not installed")


def reference_guides():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["guides"]


class TestOneHotEncoding(unittest.TestCase):
    """Channel order is A, T, C, G -- the easiest silent break in the adapter."""

    def test_the_channel_order_is_not_acgt(self):
        self.assertEqual(BASE_POSITIONS, {"A": 0, "T": 1, "C": 2, "G": 3})
        self.assertNotEqual(list(BASE_POSITIONS), ["A", "C", "G", "T"])

    def test_each_base_lands_in_its_own_channel(self):
        encoded = one_hot("ATCG", length=4)
        self.assertEqual(encoded.tolist(), [
            [1, 0, 0, 0],   # A
            [0, 1, 0, 0],   # T
            [0, 0, 1, 0],   # C
            [0, 0, 0, 1],   # G
        ])

    def test_uracil_is_read_as_thymine(self):
        self.assertEqual(one_hot("U", length=1).tolist(), one_hot("T", length=1).tolist())

    def test_the_shape_is_fixed_at_the_models_input_width(self):
        self.assertEqual(one_hot("ACGT").shape, (GUIDE_LENGTH, 4))
        # a short guide is zero-padded rather than reshaping the tensor
        self.assertEqual(one_hot("AC").sum(), 2.0)

    def test_an_unknown_base_is_an_all_zero_row(self):
        encoded = one_hot("ANG", length=3)
        self.assertEqual(encoded[1].tolist(), [0, 0, 0, 0])

    def test_every_reference_spacer_encodes_to_exactly_thirty_bases(self):
        for guide in reference_guides():
            self.assertEqual(one_hot(guide["spacer"]).sum(), float(GUIDE_LENGTH))


class TestFeatureFormulas(unittest.TestCase):
    """The normalizers are training-time values, not tunables."""

    def test_linearfold_val_matches_the_reference_for_every_guide(self):
        for guide in reference_guides():
            self.assertAlmostEqual(
                guide_linearfold_val(guide["guideMfe"]), guide["linearfoldVal"], places=12
            )

    def test_target_unfold_energy_matches_the_reference_for_every_guide(self):
        for guide in reference_guides():
            self.assertAlmostEqual(
                target_unfold_energy(guide["nativeMfe"], guide["constrainedMfe"]),
                guide["targetUnfoldEnergy"],
                places=12,
            )

    def test_the_scalar_order_is_part_of_the_contract(self):
        features = ArcFeatures(
            linearfold_val=0.1, isoform_percent=0.2, target_unfold_energy=0.3
        )
        # float32, so compare with tolerance rather than by equality
        for got, want in zip(features.as_array().tolist(), [0.1, 0.2, 0.3]):
            self.assertAlmostEqual(got, want, places=6)

    def test_the_arc_prefix_is_this_packages_direct_repeat(self):
        """Two independent sources for the same sequence, cross-checked.

        The Arc prefix arrived from a published RfxCas13d tool; effectors.py was
        written by hand. Comparing them is what exposed the corrupted 45 nt DR
        this package used to carry -- see test_effector_constants.py. Pinned in
        both directions so that neither can now drift from the other unnoticed:
        editing one alone would silently change either every Arc feature or
        every DR-interaction verdict, depending on which way it went.
        """
        from cas13guide.contract.effectors import EFFECTORS

        recorded = EFFECTORS[Effector.RFX_CAS13D].direct_repeat
        self.assertEqual(STANDARD_PREFIX.replace("T", "U"), recorded)
        # its last 30 nt are the published CasRx mature repeat
        self.assertEqual(recorded[-30:], "AACCCCUACCAACUGGUCGGGGUUUGAAAC")


class TestItRefusesViennaEnergies(unittest.TestCase):
    """No fallback, on purpose.

    ViennaRNA uses the Turner model; this network was trained on LinearFold's
    CONTRAfold energies. Substituting one for the other keeps every score in
    range and makes all of them wrong, which is the failure this package exists
    to prevent -- so the refusal is the feature.
    """

    def test_no_mfe_source_raises_and_says_why(self):
        with self.assertRaises(MFESourceRequired) as caught:
            features_for_target("A" * 30, "A" * 60, 15, None)
        message = str(caught.exception)
        self.assertIn("LinearFold", message)
        self.assertIn("ViennaRNA", message)
        self.assertIn("Turner", message)

    def test_a_supplied_source_is_used_for_all_three_energies(self):
        calls = []

        class Recording:
            version = "test"

            def mfe(self, sequence):
                calls.append(("mfe", sequence))
                return 7.17 if sequence.startswith(STANDARD_PREFIX) else 3.54

            def mfe_constrained(self, sequence, constraint):
                calls.append(("constrained", constraint))
                return -0.29

        guide = reference_guides()[0]
        features = features_for_target(
            guide["spacer"], guide["nearby"], 0, Recording()
        )
        self.assertAlmostEqual(features.linearfold_val, guide["linearfoldVal"], places=12)
        self.assertAlmostEqual(
            features.target_unfold_energy, guide["targetUnfoldEnergy"], places=12
        )
        self.assertEqual(len(calls), 3)
        # the constraint masks exactly the 30 nt target inside its flanks
        constraint = calls[-1][1]
        self.assertEqual(len(constraint), len(guide["nearby"]))
        self.assertEqual(constraint[:30], "." * 30)
        self.assertTrue(set(constraint[30:]) <= {"?"})


@needs_ort
class TestEndToEndParity(unittest.TestCase):
    """THE gate. 133 guides, reference features in, reference scores out."""

    @classmethod
    def setUpClass(cls):
        cls.guides = reference_guides()
        cls.ensemble = ArcEnsemble()
        cls.spacers = [g["spacer"] for g in cls.guides]
        cls.features = [
            ArcFeatures(g["linearfoldVal"], g["isoformPercent"], g["targetUnfoldEnergy"])
            for g in cls.guides
        ]

    def test_scores_match_the_reference_exactly(self):
        import numpy as np

        got = self.ensemble.scores(self.spacers, self.features)
        want = np.array([g["score"] for g in self.guides])
        self.assertEqual(len(got), 133)
        self.assertLess(float(np.abs(got - want).max()), 1e-6)

    def test_logits_match_the_reference_exactly(self):
        import numpy as np

        got = self.ensemble.logits(self.spacers, self.features)
        want = np.array([g["logit"] for g in self.guides])
        self.assertLess(float(np.abs(got - want).max()), 1e-6)

    def test_averaging_sigmoids_instead_of_logits_would_be_visibly_wrong(self):
        """Pins the ensemble space: the wrong one is not a rounding difference."""
        import numpy as np

        from cas13guide.services.arc_model import _sigmoid

        guides = np.stack([one_hot(s) for s in self.spacers])
        scalars = np.stack([f.as_array() for f in self.features])
        per_fold = []
        for session in self.ensemble._sessions:
            first, second = (i.name for i in session.get_inputs())
            per_fold.append(session.run(None, {first: guides, second: scalars})[0].reshape(-1))
        per_fold = np.array(per_fold)

        correct = _sigmoid(per_fold.mean(axis=0))
        wrong = _sigmoid(per_fold).mean(axis=0)
        want = np.array([g["score"] for g in self.guides])
        self.assertLess(float(np.abs(correct - want).max()), 1e-6)
        # Measured: max 0.037, mean 0.007. Small enough to survive eyeballing,
        # large enough to reorder guides.
        self.assertGreater(float(np.abs(wrong - want).max()), 0.02)

    def test_an_empty_batch_is_handled(self):
        self.assertEqual(len(self.ensemble.scores([], [])), 0)

    def test_mismatched_batch_lengths_are_refused(self):
        with self.assertRaises(ValueError):
            self.ensemble.logits(["A" * 30], [])


@needs_ort
class TestActivityModelSeam(unittest.TestCase):
    """It has to satisfy the protocol the rest of the pipeline talks to."""

    @classmethod
    def setUpClass(cls):
        guides = reference_guides()
        cls.guides = guides
        cls.model = ArcActivityModel(
            ensemble=ArcEnsemble(),
            features_by_spacer={
                g["spacer"]: ArcFeatures(
                    g["linearfoldVal"], g["isoformPercent"], g["targetUnfoldEnergy"]
                )
                for g in guides
            },
        )

    def test_it_satisfies_the_activity_model_protocol(self):
        from cas13guide.services.registry import ActivityModel

        self.assertIsInstance(self.model, ActivityModel)
        self.assertTrue(self.model.version)

    def test_predict_reproduces_the_reference_score(self):
        from cas13guide.util.duplex import profile_duplex
        from cas13guide.util.seq import normalize

        guide = self.guides[0]
        spacer = normalize(guide["spacer"])
        value = self.model.predict(guide["spacer"], profile_duplex(spacer, spacer[::-1]), None)
        self.assertAlmostEqual(value, guide["score"], places=6)

    def test_the_domain_admits_only_perfect_duplexes(self):
        """The network never sees a mismatched site, so it cannot speak to one."""
        domain = self.model.domain
        self.assertTrue(domain.supports(Effector.RFX_CAS13D, 30, 0))
        self.assertFalse(domain.supports(Effector.RFX_CAS13D, 30, 1))
        self.assertFalse(domain.supports(Effector.LWA_CAS13A, 30, 0))
        self.assertFalse(domain.supports(Effector.RFX_CAS13D, 22, 0))

    def test_an_unprepared_spacer_returns_the_neutral_point(self):
        from cas13guide.util.duplex import profile_duplex

        spacer = "A" * 30
        self.assertEqual(
            self.model.predict(spacer, profile_duplex(spacer, "U" * 30), None), 0.5
        )

    def test_repeated_predictions_are_cached_and_stable(self):
        from cas13guide.util.duplex import profile_duplex

        guide = self.guides[3]["spacer"]
        duplex = profile_duplex("A" * 30, "U" * 30)
        first = self.model.predict(guide, duplex, None)
        self.assertEqual(self.model.predict(guide, duplex, None), first)

    def test_it_discriminates_far_more_finely_than_the_heuristic(self):
        """The reason for replacing the heuristic at all."""
        from cas13guide.services.registry import HeuristicCas13Model
        from cas13guide.util.duplex import profile_duplex
        from cas13guide.util.seq import normalize

        heuristic = HeuristicCas13Model()
        arc_values, heuristic_values = [], []
        for guide in self.guides[:60]:
            spacer = normalize(guide["spacer"])
            perfect = profile_duplex(spacer, spacer[::-1])
            arc_values.append(self.model.predict(guide["spacer"], perfect, None))
            heuristic_values.append(heuristic.predict(spacer, perfect, None))
        self.assertGreater(
            max(arc_values) - min(arc_values),
            max(heuristic_values) - min(heuristic_values),
        )


class TestMissingWeightsAreReportedClearly(unittest.TestCase):
    def test_a_bad_model_directory_names_the_missing_files(self):
        import tempfile

        if not HAVE_ORT:
            self.skipTest("onnxruntime not installed")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ArcModelUnavailable) as caught:
                ArcEnsemble(model_dir=Path(directory))
            self.assertIn("fold_0.onnx", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
