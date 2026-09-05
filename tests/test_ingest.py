"""Loading a real screen -- and refusing a bad one."""

import csv
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.services.ingest import (
    ColumnMap,
    DatasetError,
    load_screen,
    write_screen,
)
from cas13guide.simulate import simulate_screen


def screen(n_transcripts=12, transcript_length=220, step=6):
    return simulate_screen(
        seed=21, step=step, signal_scale=10, transcript_effect_sd=0.8,
        noise_sd=0.2, n_transcripts=n_transcripts, transcript_length=transcript_length,
    )[0]


class ScreenFileTestCase(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.examples = screen()

    def write(self, name, rows, header, delimiter=","):
        path = Path(self.dir.name) / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def standard(self, name="screen.csv"):
        return self.write(
            name,
            [[e.spacer, e.observed_activity, e.group] for e in self.examples],
            ["spacer", "activity", "transcript"],
        )


class TestHappyPath(ScreenFileTestCase):
    def test_loads_a_clean_screen(self):
        examples, report = load_screen(self.standard())
        self.assertTrue(report.ok, report.summary())
        self.assertEqual(len(examples), len(self.examples))
        self.assertEqual(report.groups, 12)
        self.assertEqual(report.spacer_lengths, {22: len(self.examples)})

    def test_target_site_defaults_to_the_perfect_match(self):
        from cas13guide.util.seq import revcomp

        examples, _ = load_screen(self.standard())
        for e in examples[:20]:
            self.assertEqual(e.target_site, revcomp(e.spacer))
            self.assertEqual(e.duplex().n_mismatch, 0)

    def test_accepts_dna_and_lowercase(self):
        path = self.write(
            "dna.csv",
            [[e.spacer.replace("U", "T").lower(), e.observed_activity, e.group]
             for e in self.examples],
            ["spacer", "activity", "transcript"],
        )
        examples, report = load_screen(path)
        self.assertTrue(report.ok, report.summary())
        self.assertEqual(len(examples), len(self.examples))
        for e in examples[:10]:
            self.assertNotIn("T", e.spacer)

    def test_reads_tsv_by_extension(self):
        path = self.write(
            "screen.tsv",
            [[e.spacer, e.observed_activity, e.group] for e in self.examples],
            ["spacer", "activity", "transcript"],
            delimiter="\t",
        )
        _, report = load_screen(path)
        self.assertTrue(report.ok, report.summary())

    def test_custom_column_names(self):
        path = self.write(
            "custom.csv",
            [[e.spacer, e.observed_activity, e.group] for e in self.examples],
            ["guide_seq", "score", "gene"],
        )
        _, report = load_screen(
            path, columns=ColumnMap(spacer="guide_seq", activity="score", group="gene")
        )
        self.assertTrue(report.ok, report.summary())

    def test_round_trips_through_write_screen(self):
        path = Path(self.dir.name) / "out.csv"
        write_screen(path, self.examples)
        examples, report = load_screen(path)
        self.assertTrue(report.ok)
        self.assertEqual(len(examples), len(self.examples))


class TestDirection(ScreenFileTestCase):
    def test_depletion_scores_are_flipped_exactly(self):
        """log2 fold change: more negative means better knockdown.

        Rank normalization makes the flip exact, so a screen reported as
        depletion yields identical training targets to the same screen
        reported as activity.
        """
        flipped = self.write(
            "lfc.csv",
            [[e.spacer, -e.observed_activity, e.group] for e in self.examples],
            ["spacer", "activity", "transcript"],
        )
        as_depletion, report = load_screen(
            flipped, columns=ColumnMap(higher_is_better=False)
        )
        as_activity, _ = load_screen(self.standard())

        self.assertTrue(report.ok)
        self.assertIn("depletion", " ".join(report.warnings))
        for a, b in zip(as_depletion, as_activity):
            self.assertAlmostEqual(a.observed_activity, b.observed_activity, places=9)

    def test_forgetting_the_flip_inverts_the_targets(self):
        """The failure this guards against: silently training on inverted labels."""
        flipped = self.write(
            "lfc2.csv",
            [[e.spacer, -e.observed_activity, e.group] for e in self.examples],
            ["spacer", "activity", "transcript"],
        )
        wrong, _ = load_screen(flipped)  # higher_is_better left at its default
        right, _ = load_screen(flipped, columns=ColumnMap(higher_is_better=False))
        for a, b in zip(wrong[:20], right[:20]):
            self.assertAlmostEqual(a.observed_activity + b.observed_activity, 1.0, places=9)

    def test_rank_normalization_lands_inside_the_unit_interval(self):
        examples, _ = load_screen(self.standard())
        for e in examples:
            self.assertGreater(e.observed_activity, 0.0)
            self.assertLess(e.observed_activity, 1.0)

    def test_already_normalized_passes_values_through(self):
        examples, report = load_screen(self.standard(), already_normalized=True)
        self.assertTrue(report.ok)
        for loaded, original in zip(examples, self.examples):
            self.assertAlmostEqual(
                loaded.observed_activity, original.observed_activity, places=9
            )


class TestRefusals(ScreenFileTestCase):
    def test_missing_grouping_column_is_fatal(self):
        """The methodological guard: no transcript column, no training.

        Without groups every split would be ungrouped, and the resulting score
        would be measured on near-duplicates of the training rows.
        """
        path = self.write(
            "nogroup.csv",
            [[e.spacer, e.observed_activity] for e in self.examples],
            ["spacer", "activity"],
        )
        examples, report = load_screen(path)
        self.assertFalse(report.ok)
        self.assertEqual(examples, [])
        self.assertIn("grouping column", " ".join(report.errors))

    def test_missing_activity_column_is_fatal(self):
        path = self.write(
            "noact.csv",
            [[e.spacer, e.group] for e in self.examples],
            ["spacer", "transcript"],
        )
        _, report = load_screen(path)
        self.assertFalse(report.ok)

    def test_too_few_groups_is_fatal(self):
        keep = sorted({e.group for e in self.examples})[:2]
        path = self.write(
            "few.csv",
            [[e.spacer, e.observed_activity, e.group]
             for e in self.examples if e.group in keep],
            ["spacer", "activity", "transcript"],
        )
        _, report = load_screen(path)
        self.assertFalse(report.ok)
        self.assertIn("at least 3", " ".join(report.errors))

    def test_out_of_range_values_under_already_normalized_are_fatal(self):
        path = self.write(
            "oor.csv",
            [[e.spacer, e.observed_activity * 10, e.group] for e in self.examples],
            ["spacer", "activity", "transcript"],
        )
        _, report = load_screen(path, already_normalized=True)
        self.assertFalse(report.ok)
        self.assertIn("outside [0, 1]", " ".join(report.errors))

    def test_missing_file_raises(self):
        with self.assertRaises(DatasetError):
            load_screen(Path(self.dir.name) / "nope.csv")

    def test_empty_after_cleaning_is_fatal(self):
        path = self.write(
            "junk.csv",
            [["NNNNNNNNNNNNNNNNNNNNNN", 0.5, "T1"] for _ in range(10)],
            ["spacer", "activity", "transcript"],
        )
        _, report = load_screen(path)
        self.assertFalse(report.ok)


class TestWarnings(ScreenFileTestCase):
    def test_unusable_sequences_are_dropped_and_reported(self):
        rows = [[e.spacer, e.observed_activity, e.group] for e in self.examples]
        rows.append(["NNNNNNNNNNNNNNNNNNNNNN", 0.5, "ENST00099"])
        rows.append(["not a sequence", 0.5, "ENST00099"])
        path = self.write("dirty.csv", rows, ["spacer", "activity", "transcript"])
        examples, report = load_screen(path)
        self.assertTrue(report.ok)
        self.assertEqual(len(examples), len(self.examples))
        self.assertIn("unusable sequence", " ".join(report.warnings))

    def test_non_numeric_activity_is_dropped_and_reported(self):
        rows = [[e.spacer, e.observed_activity, e.group] for e in self.examples]
        rows.append([self.examples[0].spacer, "", "ENST00099"])
        rows.append([self.examples[0].spacer, "notanumber", "ENST00099"])
        path = self.write("nan.csv", rows, ["spacer", "activity", "transcript"])
        examples, report = load_screen(path)
        self.assertEqual(len(examples), len(self.examples))
        # The wording widened when `nan` turned out to reach this branch as a
        # perfectly good float: "non-numeric" is the wrong description of a
        # value that parses. Both of this test's cases -- an empty cell and
        # "notanumber" -- are still non-numeric, and still say so.
        warnings = " ".join(report.warnings)
        self.assertIn("non-numeric", warnings)
        self.assertIn("activity", warnings)

    def test_mixed_spacer_lengths_keeps_the_dominant_design(self):
        rows = [[e.spacer, e.observed_activity, e.group] for e in self.examples]
        for e in self.examples[:20]:
            rows.append([e.spacer[:20], e.observed_activity, e.group])
        path = self.write("mixed.csv", rows, ["spacer", "activity", "transcript"])
        examples, report = load_screen(path)
        self.assertTrue(report.ok)
        self.assertEqual({len(e.spacer) for e in examples}, {22})
        self.assertIn("mixed spacer lengths", " ".join(report.warnings))

    def test_duplicate_guides_are_reported_but_kept(self):
        rows = [[e.spacer, e.observed_activity, e.group] for e in self.examples]
        rows.extend(rows[:15])
        path = self.write("dup.csv", rows, ["spacer", "activity", "transcript"])
        examples, report = load_screen(path)
        self.assertEqual(len(examples), len(rows))
        self.assertIn("duplicate spacer", " ".join(report.warnings))



class TestContradictoryOptionsAreRefused(ScreenFileTestCase):
    """``already_normalized`` takes values verbatim, so it cannot also flip them.

    Rank normalization is what makes the depletion direction *exact*, and
    ``already_normalized`` skips it entirely -- so the branch simply ignored
    ``higher_is_better=False`` and loaded a depletion screen with every label
    inverted. That is the precise failure the module docstring calls fatal, and
    it produced no warning at all.
    """

    def test_already_normalized_with_depletion_direction_is_fatal(self):
        path = self.write(
            "both.csv",
            [[e.spacer, e.observed_activity, e.group] for e in self.examples],
            ["spacer", "activity", "transcript"],
        )
        examples, report = load_screen(
            path, columns=ColumnMap(higher_is_better=False), already_normalized=True
        )
        self.assertFalse(report.ok)
        self.assertEqual(examples, [])
        joined = " ".join(report.errors)
        self.assertIn("already_normalized", joined)
        self.assertIn("higher_is_better", joined)

    def test_each_option_alone_still_works(self):
        path = self.write(
            "solo.csv",
            [[e.spacer, e.observed_activity, e.group] for e in self.examples],
            ["spacer", "activity", "transcript"],
        )
        _, normalized = load_screen(path, already_normalized=True)
        _, depletion = load_screen(path, columns=ColumnMap(higher_is_better=False))
        self.assertTrue(normalized.ok, normalized.summary())
        self.assertTrue(depletion.ok, depletion.summary())


class TestANumberThatIsNotAMeasurement(ScreenFileTestCase):
    """`nan` and `inf` parse as floats and are not activities.

    `float("nan")` succeeds, so these arrive as numbers and the guard that
    drops them is the only thing keeping them out of the calibration set. It
    had never executed in the whole suite.

    A NaN is the dangerous one because it fails quietly: every comparison
    against it is False, so it survives sorting, quantile selection and the
    coverage check without ever looking wrong -- in the data every conformal
    bound in this package is calibrated on.
    """

    def rows_with(self, value):
        rows = [[e.spacer, e.observed_activity, e.group] for e in self.examples]
        rows[0][1] = value
        return self.write("odd.csv", rows, ["spacer", "activity", "transcript"])

    def test_a_nan_activity_is_dropped(self):
        examples, report = load_screen(self.rows_with("nan"))
        self.assertEqual(len(examples), len(self.examples) - 1)

    def test_an_infinite_activity_is_dropped(self):
        for text in ("inf", "-inf", "Infinity"):
            with self.subTest(value=text):
                examples, _ = load_screen(self.rows_with(text))
                self.assertEqual(len(examples), len(self.examples) - 1)

    def test_the_reason_says_which_column_to_look_at(self):
        """This module's whole job is saying precisely why a screen cannot be
        used, and "non-numeric" is the wrong description of `nan`: it parses
        as a float perfectly well."""
        _, report = load_screen(self.rows_with("nan"))
        summary = report.summary()
        self.assertIn("activity", summary)
        self.assertIn("finite", summary)

    def test_the_rest_of_the_screen_still_loads(self):
        """One unusable row is not an unusable screen."""
        _, report = load_screen(self.rows_with("nan"))
        self.assertTrue(report.ok, report.summary())


class TestATargetSiteMustMatchItsSpacer(ScreenFileTestCase):
    """A site of a different length is not that spacer's site.

    Dropped as a *sequence* failure rather than an activity one, which is the
    distinction this module draws so a user fixes the column that is actually
    wrong. Never exercised.
    """

    def test_a_site_of_the_wrong_length_is_dropped(self):
        rows = [[e.spacer, e.target_site, e.observed_activity, e.group]
                for e in self.examples]
        rows[0][1] = rows[0][1][:-3]
        path = self.write(
            "sites.csv", rows, ["spacer", "site", "activity", "transcript"])
        examples, report = load_screen(
            path, columns=ColumnMap(spacer="spacer", target_site="site",
                                    activity="activity", group="transcript"))
        self.assertEqual(len(examples), len(self.examples) - 1)
        self.assertIn("sequence", report.summary())


class TestRankingDegenerateScreens(ScreenFileTestCase):
    """`_rank_normalize` on nothing, and on one thing.

    A single observation has no rank information, so it gets the middle of the
    scale rather than 0.0 or 1.0 -- either of which would assert an extreme
    the screen cannot support. Neither branch had run.
    """

    def test_no_values_rank_to_nothing(self):
        from cas13guide.services.ingest import _rank_normalize

        self.assertEqual(_rank_normalize([], higher_is_better=True), [])

    def test_one_value_ranks_to_the_middle(self):
        from cas13guide.services.ingest import _rank_normalize

        self.assertEqual(_rank_normalize([7.5], higher_is_better=True), [0.5])
        self.assertEqual(_rank_normalize([7.5], higher_is_better=False), [0.5])


if __name__ == "__main__":
    unittest.main()
