"""Built-in targets (FR1.3), and the geometry they cannot escape."""

import re
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.effectors import EFFECTORS
from cas13guide.contract.enums import Effector, TargetClass, Universe
from cas13guide.contract.presets import (
    MIRNA_PRESETS,
    PRESET_SOURCE,
    amplicon_target,
    designability,
    get_preset,
    largest_designable_spacer_length,
    preset_ids,
)
from cas13guide.util.seq import VALID_RNA, windows

EXPECTED = {
    "hsa-miR-21-5p": "MIMAT0000076",
    "hsa-miR-23a-3p": "MIMAT0000078",
    "hsa-miR-216a-5p": "MIMAT0000273",
    "hsa-miR-23b-3p": "MIMAT0000418",
    "hsa-miR-29c-3p": "MIMAT0000681",
}


class TestPresetContents(unittest.TestCase):
    def test_all_five_required_targets_are_present(self):
        self.assertEqual(set(preset_ids()), set(EXPECTED))

    def test_accessions_match_mirbase(self):
        for preset_id, accession in EXPECTED.items():
            self.assertEqual(get_preset(preset_id).accession, accession)

    def test_sequences_are_clean_rna_of_plausible_mature_length(self):
        for preset in MIRNA_PRESETS.values():
            self.assertTrue(set(preset.sequence) <= VALID_RNA, preset.preset_id)
            self.assertNotIn("T", preset.sequence)
            self.assertNotIn("N", preset.sequence)
            # Mature miRNAs are 21-25 nt; anything outside that is a bad paste.
            self.assertGreaterEqual(preset.length, 21, preset.preset_id)
            self.assertLessEqual(preset.length, 25, preset.preset_id)

    def test_provenance_is_recorded_not_described(self):
        """NFR4/DR4: a reproducible run needs the database version on record.

        This test used to assert that a ``release``, ``sha256`` and ``retrieved``
        were *present*, and it passed against values that had been invented --
        the checksum had never been computed against any file. Asserting the
        shape of a provenance record proves only that someone typed 64 hex
        characters. So it now asserts the honest invariant instead: a checksum
        may exist only once the presets have actually been verified, and while
        they have not, the record must say so in as many words. See
        test_preset_provenance.py for the machinery that discharges it.
        """
        from cas13guide.contract.presets import PRESETS_VERIFIED

        self.assertIn("database", PRESET_SOURCE)
        self.assertIn("url", PRESET_SOURCE)
        if PRESETS_VERIFIED:
            self.assertRegex(PRESET_SOURCE["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(PRESET_SOURCE.get("release"))
        else:
            self.assertNotIn("sha256", PRESET_SOURCE)
            self.assertIn("no", PRESET_SOURCE["verified"].lower())

    def test_the_two_mir_23_paralogs_are_near_identical(self):
        """They differ only near the 3' end -- a discrimination task, not two targets."""
        a = get_preset("hsa-miR-23a-3p").sequence
        b = get_preset("hsa-miR-23b-3p").sequence
        shared = sum(1 for x, y in zip(a, b) if x == y)
        self.assertGreater(shared / min(len(a), len(b)), 0.8)
        self.assertIn("cross-react", get_preset("hsa-miR-23b-3p").note)

    def test_unknown_preset_names_the_available_ones(self):
        with self.assertRaises(KeyError) as caught:
            get_preset("hsa-miR-999")
        self.assertIn("hsa-miR-21-5p", str(caught.exception))

    def test_to_target_spec_round_trips(self):
        preset = get_preset("hsa-miR-21-5p")
        spec = preset.to_target_spec(expression=42.0)
        self.assertEqual(spec.sequence, preset.sequence)
        self.assertEqual(spec.target_class, TargetClass.REQUIRED)
        self.assertEqual(spec.biotype, Universe.NCRNA)
        self.assertEqual(spec.expression, 42.0)


class TestMatureMiRNAGeometry(unittest.TestCase):
    """A 28 nt spacer does not fit in a 24 nt target, and saying so is the job.

    This is the collision behind FR1.3 + FR1.5 + FR2.1: with LwaCas13a's
    canonical geometry the presets yield *zero* candidates. The package cannot
    make that untrue; it can only be explicit about it, offer the shorter
    spacer, and mark the result unvalidated.
    """

    def test_canonical_spacer_yields_no_windows_on_any_preset(self):
        for preset in MIRNA_PRESETS.values():
            report = designability(preset, Effector.LWA_CAS13A)   # canonical 28
            self.assertEqual(report.windows, 0, preset.preset_id)
            self.assertFalse(report.designable)
            self.assertIn("cannot hold", report.note)
            self.assertIn("amplicon", report.note)

    def test_a_shorter_tolerated_spacer_does_fit(self):
        for preset in MIRNA_PRESETS.values():
            length = largest_designable_spacer_length(preset, Effector.LWA_CAS13A)
            self.assertIsNotNone(length, preset.preset_id)
            report = designability(preset, Effector.LWA_CAS13A, spacer_length=length)
            self.assertTrue(report.designable)
            self.assertGreaterEqual(report.windows, 1)

    def test_and_is_reported_as_unvalidated_when_it_does(self):
        """Fitting is not the same as being a geometry anyone has data for.

        This asserted the note contained "out-of-domain (R3 REVIEW)". The
        first half was true and the second was not: an out-of-domain pair gets
        ``Interval.unknown``, whose lower bound is 0.0, and benefit is read at
        its lower bound -- so the coverage gate FAILs it. R3 REVIEW is what an
        *in-domain* pair with a weak calibration stratum earns. The note now
        says what happens, so the assertion checks that instead of the phrase
        that was wrong.
        """
        preset = get_preset("hsa-miR-21-5p")
        length = largest_designable_spacer_length(preset, Effector.LWA_CAS13A)
        report = designability(preset, Effector.LWA_CAS13A, spacer_length=length)
        self.assertFalse(report.canonical_length)
        self.assertIn("not narrowed at all", report.note)
        self.assertIn("refused rather than certified", report.note)
        self.assertNotIn("R3 REVIEW", report.note)

    def test_the_note_names_an_ortholog_that_does_publish_the_length(self):
        """"Unvalidated here" is only actionable with somewhere to go."""
        preset = get_preset("hsa-miR-216a-5p")        # 22 nt
        length = largest_designable_spacer_length(preset, Effector.LWA_CAS13A)
        note = designability(preset, Effector.LWA_CAS13A, spacer_length=length).note
        self.assertIn("RfxCas13d publishes 22 nt", note)

    def test_a_length_no_ortholog_publishes_says_so_rather_than_naming_none(self):
        preset = get_preset("hsa-miR-21-5p")          # 24 nt, nobody publishes it
        length = largest_designable_spacer_length(preset, Effector.LWA_CAS13A)
        note = designability(preset, Effector.LWA_CAS13A, spacer_length=length).note
        self.assertIn("No ortholog in this package publishes that length", note)

    def test_largest_fitting_length_is_the_largest_that_fits(self):
        preset = get_preset("hsa-miR-21-5p")          # 24 nt
        length = largest_designable_spacer_length(preset, Effector.LWA_CAS13A)
        self.assertEqual(length, preset.length)       # exactly one window
        self.assertEqual(len(list(windows(preset.sequence, length))), 1)
        # asking for more windows forces a shorter spacer
        shorter = largest_designable_spacer_length(preset, Effector.LWA_CAS13A, min_windows=3)
        self.assertEqual(shorter, preset.length - 2)

    def test_a_length_below_the_orthologs_floor_is_reported_as_impossible(self):
        low, _high = EFFECTORS[Effector.LWA_CAS13A].spacer_length_range
        tiny = MIRNA_PRESETS["hsa-miR-216a-5p"]
        self.assertGreaterEqual(low, 1)
        # nothing shorter than the floor is ever offered
        self.assertGreaterEqual(
            largest_designable_spacer_length(tiny, Effector.LWA_CAS13A), low
        )


class TestAmpliconPathRestoresNormalDesign(unittest.TestCase):
    """The recommended shape: detect the amplicon, not the bare mature strand."""

    CONTEXT_5 = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"
    CONTEXT_3 = "GCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUAGCUA"

    def test_amplicon_contains_the_mature_sequence(self):
        preset = get_preset("hsa-miR-21-5p")
        target = amplicon_target(preset, self.CONTEXT_5, self.CONTEXT_3)
        self.assertIn(preset.sequence, target.sequence)
        self.assertEqual(
            len(target.sequence),
            len(self.CONTEXT_5) + preset.length + len(self.CONTEXT_3),
        )

    def test_it_tiles_normally_at_the_canonical_spacer_length(self):
        preset = get_preset("hsa-miR-21-5p")
        target = amplicon_target(preset, self.CONTEXT_5, self.CONTEXT_3)
        canonical = EFFECTORS[Effector.LWA_CAS13A].preferred_spacer_length
        self.assertGreater(len(list(windows(target.sequence, canonical))), 50)

    def test_dna_context_is_transcribed(self):
        preset = get_preset("hsa-miR-29c-3p")
        target = amplicon_target(preset, "ATGCATGC", "TTTTAAAA")
        self.assertNotIn("T", target.sequence)


class TestBothWaysToATargetStillAgree(unittest.TestCase):
    """``to_target_spec`` is a second path, and it agrees by coincidence.

    Production never calls it. ``--preset`` builds a ``SequenceRecord`` and
    goes through the same record-to-target conversion a FASTA does, which
    takes ``gene`` from the record name and hardcodes ``biotype`` to NCRNA.
    The preset carries its own ``display_name`` and ``biotype``, so the two
    paths agree only because every shipped preset happens to have
    ``display_name == preset_id`` and an NCRNA biotype.

    That matters because the tests use ``to_target_spec``:
    ``test_preset_provenance`` builds its spec with it. A preset whose display
    name or biotype differed would hand those tests a target production never
    produces, and nothing would say so. This repository has removed one second
    path already -- ``fasta.records_to_target_specs``, in the commit that
    ``test_how_records_become_targets`` is named for -- and the method is kept
    here because a library caller has a use for it. What is not kept is the
    coincidence being unwatched.

    Compared against a real run rather than against a copy of the production
    construction, because a copy is a third path and would drift on its own.
    """

    def _designed_target(self, preset_id: str):
        from cas13guide.services.workflow import DesignRequest, run_request

        outcome = run_request(DesignRequest(
            preset=preset_id, task="knockdown",
            assay="cell_culture_knockdown", effector="RfxCas13d",
        ))
        return outcome.spec.contract.targets[0]

    def test_every_shipped_preset_converts_the_same_either_way(self):
        fields = ("target_id", "sequence", "target_class", "gene", "biotype",
                  "expression")
        for preset_id, preset in MIRNA_PRESETS.items():
            with self.subTest(preset=preset_id):
                designed = self._designed_target(preset_id)
                stated = preset.to_target_spec()
                for field in fields:
                    self.assertEqual(
                        getattr(designed, field), getattr(stated, field),
                        f"{preset_id}.{field}: a run and to_target_spec disagree, "
                        "so the tests that use to_target_spec are describing a "
                        "target no run produces",
                    )


if __name__ == "__main__":
    unittest.main()
