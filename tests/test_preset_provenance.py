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

    def test_a_verified_preset_no_longer_claims_to_be_unverified(self):
        """The warning has to stop once the work behind it is done."""
        from cas13guide.services.limitations import limitations_for_run

        spec = make_spec(targets=(MIRNA_PRESETS["hsa-miR-21-5p"].to_target_spec(),))
        ids = {limitation.id for limitation in limitations_for_run(spec)}
        if PRESETS_VERIFIED:
            self.assertNotIn("preset-sequence-unverified", ids)
        else:
            self.assertIn("preset-sequence-unverified", ids)

    def test_a_preset_miRBase_moved_in_release_23_says_so(self):
        """Verified is not the same as unchanging.

        miRBase release 23 revised the mature sequence of MIMAT0000076 and
        MIMAT0000078. Our copy matches the new one, so anyone comparing against
        a paper from before that release sees a different 3' end and needs to
        know it is a release difference and not an error.
        """
        from cas13guide.contract.presets import CHANGED_IN_RELEASE_23
        from cas13guide.services.limitations import limitations_for_run

        self.assertIn(MIRNA_PRESETS["hsa-miR-21-5p"].accession, CHANGED_IN_RELEASE_23)
        spec = make_spec(targets=(MIRNA_PRESETS["hsa-miR-21-5p"].to_target_spec(),))
        ids = {limitation.id for limitation in limitations_for_run(spec)}
        self.assertIn("preset-sequence-changed-upstream", ids)

    def test_a_preset_miRBase_did_not_move_stays_quiet(self):
        """Both directions -- it is about those two accessions, not all five."""
        from cas13guide.services.limitations import limitations_for_run

        spec = make_spec(targets=(MIRNA_PRESETS["hsa-miR-29c-3p"].to_target_spec(),))
        ids = {limitation.id for limitation in limitations_for_run(spec)}
        self.assertNotIn("preset-sequence-changed-upstream", ids)

    def test_a_run_on_a_pasted_sequence_gets_no_preset_note_at_all(self):
        """A note that fires unconditionally is indistinguishable from a bug."""
        from cas13guide.services.limitations import limitations_for_run

        ids = {limitation.id for limitation in limitations_for_run(make_spec())}
        self.assertNotIn("preset-sequence-unverified", ids)
        self.assertNotIn("preset-sequence-changed-upstream", ids)


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
        # PRESETS_VERIFIED is module state a human sets after reading a real
        # report; running the checker must never flip it on its own.
        from cas13guide.contract import presets as module

        self.assertIs(module.PRESETS_VERIFIED, PRESETS_VERIFIED)


class TestTheRecordedProvenanceIsReproducible(unittest.TestCase):
    """A checksum is only worth the procedure that can reproduce it.

    This is the test the original one should have been. The version it replaced
    asserted that PRESET_SOURCE['sha256'] matched ^[0-9a-f]{64}$ -- a check that
    any 64 hex characters pass, which is why it was taken as evidence the value
    was invented when it was in fact correct. Shape is not provenance.

    mature.fa is 5.7 MB and is not vendored, so this skips when it is absent and
    says how to get it. A skip that explains itself is honest; the assertion it
    guards is the only thing here that could ever prove the record right.

    Point CAS13GUIDE_MATURE_FA at a downloaded copy of
    https://www.mirbase.org/download/mature.fa to run it.
    """

    def _mature_fa(self):
        import os
        import pathlib

        override = os.environ.get("CAS13GUIDE_MATURE_FA")
        candidates = [pathlib.Path(override)] if override else []
        candidates.append(pathlib.Path(__file__).resolve().parent / "fixtures" / "mature.fa")
        for path in candidates:
            if path.is_file():
                return path
        self.skipTest(
            "mature.fa not present; set CAS13GUIDE_MATURE_FA to a copy of "
            f"{PRESET_SOURCE['url']} to check the recorded provenance"
        )

    def test_the_recorded_checksum_matches_the_real_file(self):
        digest, _checks = verify_against_mirbase(self._mature_fa())
        self.assertEqual(digest, PRESET_SOURCE["sha256"])

    def test_every_preset_matches_the_real_file(self):
        _digest, checks = verify_against_mirbase(self._mature_fa())
        for check in checks:
            with self.subTest(preset=check.preset_id):
                self.assertTrue(check.ok, check.line())

    def test_the_recorded_size_and_record_count_match(self):
        path = self._mature_fa()
        self.assertEqual(str(path.stat().st_size), PRESET_SOURCE["bytes"])
        text = path.read_text(encoding="utf-8", errors="replace")
        self.assertEqual(str(text.count(">")), PRESET_SOURCE["records"])


class TestVerifiedProvenanceIsComplete(unittest.TestCase):
    """If it claims to be verified, the record has to carry enough to re-check."""

    def test_a_verified_record_pins_a_release_and_a_checksum(self):
        if not PRESETS_VERIFIED:
            self.skipTest("presets are not marked verified")
        self.assertRegex(PRESET_SOURCE["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(PRESET_SOURCE["release"])
        self.assertTrue(PRESET_SOURCE["retrieved"])
        self.assertTrue(PRESET_SOURCE["url"].startswith("https://"))

    def test_the_release_that_moved_two_accessions_is_recorded(self):
        """Provenance without the caveat would still mislead a reader."""
        from cas13guide.contract.presets import CHANGED_IN_RELEASE_23

        self.assertEqual(len(CHANGED_IN_RELEASE_23), 2)
        accessions = {p.accession for p in MIRNA_PRESETS.values()}
        self.assertTrue(CHANGED_IN_RELEASE_23 <= accessions)
