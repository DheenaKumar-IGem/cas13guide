"""Conserved regions across variants (FR7.1) and losing one (FR7.2)."""

import random
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import Universe
from cas13guide.services.conservation import (
    best_conserved_windows,
    find_conserved_windows,
    summarize_conservation,
    variant_contexts,
)
from cas13guide.services.fasta import SequenceRecord


def mutate(sequence: str, n: int, seed: int) -> str:
    rng = random.Random(seed)
    out = list(sequence)
    for _ in range(n):
        i = rng.randrange(len(out))
        out[i] = rng.choice([b for b in "AUGC" if b != out[i]])
    return "".join(out)


def random_rna(length: int, seed: int) -> str:
    """Seed the generator ONCE.

    ``"".join(random.Random(seed).choice("AUGC") for _ in range(n))`` builds a
    fresh generator per character and so returns a homopolymer -- which silently
    makes every conservation fixture trivially 100% conserved.
    """
    rng = random.Random(seed)
    return "".join(rng.choice("AUGC") for _ in range(length))


BASE = random_rna(200, 3)


def isoforms():
    return [
        SequenceRecord("iso1", BASE),
        SequenceRecord("iso2", mutate(BASE, 6, 11)),
        SequenceRecord("iso3", mutate(BASE, 6, 12)),
    ]


class TestConservedWindows(unittest.TestCase):
    """FR7.1 -- where can one guide serve every variant?"""

    def test_identical_sequences_are_conserved_everywhere(self):
        records = [SequenceRecord("a", BASE), SequenceRecord("b", BASE)]
        summary = summarize_conservation(records, 28, max_mismatches=0)
        self.assertEqual(summary.windows_covering_all, summary.windows_examined)
        self.assertEqual(summary.fully_conserved_windows, summary.windows_examined)
        self.assertTrue(summary.designable)

    def test_divergent_isoforms_leave_only_some_windows(self):
        summary = summarize_conservation(isoforms(), 28, max_mismatches=0)
        self.assertGreater(summary.windows_covering_all, 0)
        self.assertLess(summary.windows_covering_all, summary.windows_examined)

    def test_a_wider_envelope_admits_more_windows(self):
        strict = summarize_conservation(isoforms(), 28, max_mismatches=0)
        loose = summarize_conservation(isoforms(), 28, max_mismatches=2)
        self.assertGreater(loose.windows_covering_all, strict.windows_covering_all)
        # but "perfectly conserved" is a property of the sequences, not the budget
        self.assertEqual(loose.fully_conserved_windows, strict.fully_conserved_windows)

    def test_an_unrelated_sequence_is_named_as_unreachable(self):
        records = [
            SequenceRecord("ref", BASE),
            SequenceRecord("other", random_rna(200, 99)),
        ]
        summary = summarize_conservation(records, 28, max_mismatches=0)
        self.assertFalse(summary.designable)
        self.assertIn("other", summary.unreachable_variants)
        self.assertIn("no shared guide is possible", summary.summary())
        self.assertIn("other", summary.summary())

    def test_best_windows_are_sorted_by_worst_fit(self):
        best = best_conserved_windows(isoforms(), 28, max_mismatches=2)
        worst = [w.worst_mismatches for w in best]
        self.assertEqual(worst, sorted(worst))
        for window in best:
            self.assertTrue(window.covers_all(2))
            self.assertEqual(window.missing_variants, ())

    def test_a_window_that_loses_a_variant_is_reported_not_dropped(self):
        found = find_conserved_windows(isoforms(), 28, max_mismatches=0)
        lost = [w for w in found if w.missing_variants]
        self.assertTrue(lost, "fixture should contain windows that miss a variant")
        for window in lost:
            # None, not a large number: "no site at all" is worse than any
            # finite mismatch count and must not be flattened into one.
            self.assertIsNone(window.worst_mismatches)
            self.assertFalse(window.covers_all(99))

    def test_the_spacer_is_the_reverse_complement_of_the_reference_site(self):
        from cas13guide.util.seq import revcomp

        [window] = find_conserved_windows(
            [SequenceRecord("a", BASE[:28]), SequenceRecord("b", BASE[:28])], 28
        )
        self.assertEqual(window.spacer, revcomp(window.target_site))
        self.assertEqual(window.target_site, BASE[:28])


class TestConservationIsMeasuredInDuplexSpace(unittest.TestCase):
    """A G-U wobble is a real base pair, so it is not a loss of conservation.

    Scoring plain sequence identity would discard guides that work perfectly
    well against every variant, and would do it silently. This is the same
    asymmetry ``util/duplex.py`` exists to get right, applied to variants.
    """

    def _pair_with_substitution(self, frm: str, to: str):
        sequence = random_rna(60, 21)
        index = sequence.index(frm, 10)
        variant = sequence[:index] + to + sequence[index + 1 :]
        return [SequenceRecord("ref", sequence), SequenceRecord("var", variant)], index

    def test_an_a_to_g_change_still_pairs_and_stays_conserved(self):
        records, _index = self._pair_with_substitution("A", "G")
        summary = summarize_conservation(records, 28, max_mismatches=0)
        self.assertEqual(summary.windows_covering_all, summary.windows_examined)

    def test_a_c_to_u_change_still_pairs_and_stays_conserved(self):
        records, _index = self._pair_with_substitution("C", "U")
        summary = summarize_conservation(records, 28, max_mismatches=0)
        self.assertEqual(summary.windows_covering_all, summary.windows_examined)

    def test_but_a_true_mismatch_does_break_conservation(self):
        """The reverse substitution is not a wobble, so it must cost."""
        records, _index = self._pair_with_substitution("G", "A")
        summary = summarize_conservation(records, 28, max_mismatches=0)
        self.assertLess(summary.windows_covering_all, summary.windows_examined)


class TestBadInput(unittest.TestCase):
    def test_a_single_sequence_is_refused(self):
        with self.assertRaises(ValueError):
            summarize_conservation([SequenceRecord("a", BASE)], 28)

    def test_an_unknown_reference_names_the_available_ones(self):
        with self.assertRaises(KeyError) as caught:
            find_conserved_windows(isoforms(), 28, reference="nope")
        self.assertIn("iso1", str(caught.exception))

    def test_the_reference_can_be_chosen_explicitly(self):
        from_iso1 = summarize_conservation(isoforms(), 28, reference="iso1")
        from_iso2 = summarize_conservation(isoforms(), 28, reference="iso2")
        self.assertEqual(from_iso1.reference, "iso1")
        self.assertEqual(from_iso2.reference, "iso2")
        self.assertNotIn("iso2", from_iso2.variants)


class TestVariantContextsDriveRobustness(unittest.TestCase):
    """FR7.2 -- losing a variant shows up as a lower robustness score.

    The metric already exists; variant_contexts is what feeds it, by putting
    each variant in its own context and mapping them all to the same target id.
    """

    UNIVERSES = frozenset({Universe.NCRNA, Universe.SYNTHETIC})

    def test_one_context_per_variant_all_naming_the_same_target(self):
        contexts = variant_contexts(isoforms(), self.UNIVERSES, target_id="t1")
        self.assertEqual(len(contexts), 3)
        self.assertEqual(contexts[0].name, "reference")
        self.assertEqual({c.name for c in contexts}, {"reference", "iso2", "iso3"})
        for context in contexts:
            self.assertEqual(set(context.locus_to_target.values()), {"t1"})

    def test_a_guide_conserved_everywhere_keeps_its_coverage(self):
        metrics = self._robustness(conserved=True)
        self.assertGreater(metrics, 0.0)

    def test_a_guide_that_loses_a_variant_scores_lower(self):
        self.assertLess(self._robustness(conserved=False), self._robustness(conserved=True))

    def _robustness(self, conserved: bool) -> float:
        """Worst required coverage across contexts, for one hand-built guide."""
        from cas13guide.metrics.aggregate import build_metric_vector
        from cas13guide.contract.enums import TargetClass

        present = make_evidence(0.9, 0.95, target_id="t1", target_class=TargetClass.REQUIRED)
        contexts = {"reference": (present,)}
        # A variant the guide reaches, versus one where it has no site at all.
        contexts["variant"] = (present,) if conserved else ()
        spec = make_spec()
        return build_metric_vector(contexts, clean_qc(), spec).robustness


if __name__ == "__main__":
    unittest.main()
