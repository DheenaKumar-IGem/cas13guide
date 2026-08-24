"""Preset sequences are the other data nothing exercises.

The direct-repeat bug (test_effector_constants.py) was one wrong constant in a
file of constants nobody tested. These are the same shape of risk and worse
consequence: a wrong DR flags the wrong guides, a wrong *target* means every
guide on the page is against something the user is not measuring, and no score
on that page can show it.

They cannot be settled from inside this package the way the DR could -- a mature
miRNA is 21-24 nt, so length disqualifies nothing, and there is no structural
invariant to appeal to. So this file does two things instead: it pins the
invariants that *are* checkable without a reference, and it tests the machinery
that discharges the rest against a real miRBase file.
"""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.presets import (
    MIRNA_PRESETS,
    PRESET_SOURCE,
    PRESETS_VERIFIED,
    verify_against_mirbase,
)

REQUIRED = (
    "hsa-miR-21-5p",
    "hsa-miR-23a-3p",
    "hsa-miR-216a-5p",
    "hsa-miR-23b-3p",
    "hsa-miR-29c-3p",
)


class TestThePresetsAreTheOnesAskedFor(unittest.TestCase):
    def test_all_five_named_targets_are_present(self):
        self.assertEqual(set(MIRNA_PRESETS), set(REQUIRED))

    def test_every_sequence_is_rna(self):
        for preset in MIRNA_PRESETS.values():
            with self.subTest(preset=preset.preset_id):
                self.assertTrue(set(preset.sequence) <= set("AUGC"))

    def test_every_accession_is_a_well_formed_mimat(self):
        for preset in MIRNA_PRESETS.values():
            with self.subTest(preset=preset.preset_id):
                self.assertRegex(preset.accession, r"^MIMAT\d{7}$")

    def test_accessions_are_distinct(self):
        accessions = [p.accession for p in MIRNA_PRESETS.values()]
        self.assertEqual(len(set(accessions)), len(accessions))

    def test_every_length_is_in_the_mature_mirna_range(self):
        """Loose on purpose -- it excludes a pasted hairpin, and not much else."""
        for preset in MIRNA_PRESETS.values():
            with self.subTest(preset=preset.preset_id):
                self.assertGreaterEqual(preset.length, 20)
                self.assertLessEqual(preset.length, 25)


class TestTheCrossReactivityNoteIsTrue(unittest.TestCase):
    """miR-23a-3p and miR-23b-3p carry a claim in prose, and prose cannot fail.

    The note says they differ only near the 3-prime end, and that the pair is a
    discrimination task rather than two independent targets. That is a testable
    statement about the two sequences, and it is the reason a user should not
    read two PASS rows as two independent successes.
    """

    def setUp(self):
        self.a = MIRNA_PRESETS["hsa-miR-23a-3p"].sequence
        self.b = MIRNA_PRESETS["hsa-miR-23b-3p"].sequence

    def test_they_differ_only_in_the_last_quarter(self):
        differing = [i for i, (x, y) in enumerate(zip(self.a, self.b)) if x != y]
        self.assertTrue(differing, "the two presets are identical")
        self.assertGreater(min(differing), 0.7 * len(self.a))

    def test_a_guide_against_one_is_within_mismatch_range_of_the_other(self):
        """The operative consequence: an off-target search at mm<=2 finds it."""
        differing = sum(1 for x, y in zip(self.a, self.b) if x != y)
        self.assertLessEqual(differing, 2)

    def test_the_note_actually_says_so(self):
        self.assertIn("cross-react", MIRNA_PRESETS["hsa-miR-23b-3p"].note)


class TestProvenanceIsNotOverclaimed(unittest.TestCase):
    """The failure this replaced: an invented sha256.

    A checksum is read as proof. One that nobody computed is worse than no
    provenance at all, because it stops the reader looking. These exist so that
    re-adding one without doing the work fails.
    """

    def test_no_checksum_is_claimed_while_unverified(self):
        if not PRESETS_VERIFIED:
            self.assertNotIn("sha256", PRESET_SOURCE)

    def test_no_release_or_retrieval_date_is_claimed_while_unverified(self):
        if not PRESETS_VERIFIED:
            self.assertNotIn("release", PRESET_SOURCE)
            self.assertNotIn("retrieved", PRESET_SOURCE)

    def test_the_source_says_it_is_unverified(self):
        if not PRESETS_VERIFIED:
            self.assertIn("verified", PRESET_SOURCE)
            self.assertIn("no", PRESET_SOURCE["verified"].lower())

    def test_a_run_using_a_preset_says_so_in_its_limitations(self):
        from cas13guide.services.limitations import limitations_for_run

        spec = make_spec(targets=(MIRNA_PRESETS["hsa-miR-21-5p"].to_target_spec(),))
        ids = {limitation.id for limitation in limitations_for_run(spec)}
        self.assertIn("preset-sequence-unverified", ids)

    def test_a_run_on_a_pasted_sequence_does_not(self):
        """A warning that fires unconditionally is indistinguishable from a bug."""
        from cas13guide.services.limitations import limitations_for_run

        ids = {limitation.id for limitation in limitations_for_run(make_spec())}
        self.assertNotIn("preset-sequence-unverified", ids)


class TestVerificationAgainstMirbase(unittest.TestCase):
    """The escape hatch has to work, or the limitation is a dead end."""

    HEADER = ">hsa-miR-21-5p MIMAT0000076 Homo sapiens miR-21-5p"
    CANONICAL = "UAGCUUAUCAGACUGAUGUUGA"

    def _write(self, text):
        import pathlib
        import tempfile

        path = pathlib.Path(tempfile.mkdtemp()) / "mature.fa"
        path.write_text(text, encoding="utf-8")
        return path

    def test_it_returns_the_files_real_checksum(self):
        import hashlib

        path = self._write(self.HEADER + "\n" + self.CANONICAL + "\n")
        digest, _ = verify_against_mirbase(path)
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_a_matching_sequence_reports_ok(self):
        recorded = MIRNA_PRESETS["hsa-miR-21-5p"].sequence
        path = self._write(self.HEADER + "\n" + recorded + "\n")
        _, checks = verify_against_mirbase(path)
        self.assertTrue(next(c for c in checks if c.preset_id == "hsa-miR-21-5p").ok)

    def test_a_mismatch_is_reported_with_both_sequences(self):
        path = self._write(self.HEADER + "\n" + self.CANONICAL + "\n")
        _, checks = verify_against_mirbase(path)
        check = next(c for c in checks if c.preset_id == "hsa-miR-21-5p")
        self.assertTrue(check.found)
        self.assertFalse(check.matches)
        self.assertIn(check.expected, check.line())
        self.assertIn(check.actual, check.line())

    def test_an_absent_accession_is_not_silently_a_pass(self):
        path = self._write(self.HEADER + "\n" + self.CANONICAL + "\n")
        _, checks = verify_against_mirbase(path)
        missing = [c for c in checks if not c.found]
        self.assertEqual(len(missing), 4)
        for check in missing:
            self.assertFalse(check.ok)
            self.assertIn("NOT FOUND", check.line())

    def test_it_keys_on_accession_not_name(self):
        """Names are reused across releases; MIMAT accessions are stable."""
        recorded = MIRNA_PRESETS["hsa-miR-21-5p"].sequence
        path = self._write(">something-else MIMAT0000076 whatever\n" + recorded + "\n")
        _, checks = verify_against_mirbase(path)
        self.assertTrue(next(c for c in checks if c.preset_id == "hsa-miR-21-5p").ok)

    def test_a_wrapped_sequence_is_joined(self):
        recorded = MIRNA_PRESETS["hsa-miR-21-5p"].sequence
        path = self._write(self.HEADER + "\n" + recorded[:10] + "\n" + recorded[10:] + "\n")
        _, checks = verify_against_mirbase(path)
        self.assertTrue(next(c for c in checks if c.preset_id == "hsa-miR-21-5p").ok)

    def test_dna_in_the_file_is_normalized_before_comparing(self):
        recorded = MIRNA_PRESETS["hsa-miR-21-5p"].sequence
        path = self._write(self.HEADER + "\n" + recorded.replace("U", "T") + "\n")
        _, checks = verify_against_mirbase(path)
        self.assertTrue(next(c for c in checks if c.preset_id == "hsa-miR-21-5p").ok)

    def test_it_reports_and_does_not_mutate(self):
        """A verifier that edits the data it checks cannot be run twice."""
        recorded = MIRNA_PRESETS["hsa-miR-21-5p"].sequence
        verify_against_mirbase(self._write(self.HEADER + "\n" + self.CANONICAL + "\n"))
        self.assertEqual(MIRNA_PRESETS["hsa-miR-21-5p"].sequence, recorded)
        self.assertFalse(PRESETS_VERIFIED)
