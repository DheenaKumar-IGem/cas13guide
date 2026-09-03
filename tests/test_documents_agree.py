"""The cheap half of ``examples/verify_the_documented_numbers.py``, as tests.

That instrument does two kinds of work. One kind is expensive -- it runs a
design to compare the landing page's quoted limitations against what the tool
actually prints, and re-measures a Monte Carlo figure over 20,000 spacers. That
belongs in ``examples/``, run deliberately, like the mutation sweep and the
coverage measurement.

The other kind is string comparison between files, costs nothing, and catches
the failure that has actually recurred: **two documents stating one fact
differently**. The test count drifted three separate times, once sitting at
1119, 1122 and 1135 simultaneously, and each time it was found by someone
running the instrument by hand rather than by the suite. A check that only runs
when remembered is a check that documents the drift instead of preventing it.

So the fast half lives here, importing the instrument rather than restating it
-- two copies of this logic would be the very defect it exists to catch.

One consequence worth knowing before it surprises someone: this file is inside
the number it checks. Adding a test here changes the count the documents have
to state, and the first run after writing it failed for exactly that reason.
That is the mechanism working, not a flaw in it -- the count is only worth
stating if adding tests is what moves it.
"""

import pathlib
import sys
import unittest

from helpers import *  # noqa: F401,F403

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))

import verify_the_documented_numbers as documents  # noqa: E402


class TestTheTestCountIsStatedOnce(unittest.TestCase):
    """Three places say how many tests there are. They must be one number."""

    def test_every_document_states_it(self):
        counts = documents.documented_test_counts()
        self.assertEqual(len(counts), 3, f"found {sorted(counts)}")

    def test_they_agree_with_each_other(self):
        counts = documents.documented_test_counts()
        self.assertEqual(len(set(counts.values())), 1,
                         "; ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    def test_and_with_the_suite_they_describe(self):
        """Collected, not passing: three tests need a miRBase file and skip
        without it, so the passing count moves with an environment variable."""
        counts = documents.documented_test_counts()
        self.assertEqual(set(counts.values()), {documents.collected_test_count()})


class TestTheReadmeCountsItsOwnProcedures(unittest.TestCase):
    """"Nine claims ... each by a procedure anyone can re-run", above a block
    that listed eight. It was wrong from the commit that wrote it."""

    def test_the_prose_matches_the_block(self):
        claimed, actual = documents.claimed_and_actual_procedures()
        self.assertIsNotNone(claimed, "the sentence stopped stating a number")
        self.assertEqual(claimed, actual)


class TestTheFlipRateIsStatedConsistently(unittest.TestCase):
    """Four files carry this figure; five carried it as a stale 4.9%.

    Only agreement is checked here. Whether they agree on the *right* number
    needs a fresh measurement, which is the instrument's job because it costs
    20,000 duplex scans.
    """

    def test_every_source_states_it(self):
        stated = documents.stated_flip_rates()
        self.assertEqual(len(stated), len(documents.FLIP_RATE_SOURCES),
                         f"found {sorted(stated)}")

    def test_they_agree_with_each_other(self):
        stated = documents.stated_flip_rates()
        self.assertEqual(len(set(stated.values())), 1,
                         "; ".join(f"{k}={v}" for k, v in sorted(stated.items())))


class TestTheMutationCountIsStatedConsistently(unittest.TestCase):
    """Three figures describe the size of the mutation sweep, and one of them
    is a denominator rather than a headline -- "10 of the 32 are caught only by
    the full run". Adding a mutation moves all three, and the denominator is
    the one a writer forgets, because it does not look like the number they
    came to change.

    Whether every mutation is *caught* is the sweep's own result and takes
    minutes to establish, so it is not asserted here; only that no document
    claims more caught than the sweep carries.
    """

    def test_every_stated_total_is_the_size_of_the_set(self):
        size = documents.mutation_set_size()
        stated = documents.stated_mutation_counts()
        totals = {k: v for k, v in stated.items() if "caught" not in k}
        self.assertTrue(totals, f"no total found among {sorted(stated)}")
        self.assertEqual(set(totals.values()), {size},
                         "; ".join(f"{k}={v}" for k, v in sorted(totals.items())))

    def test_no_document_claims_more_caught_than_exist(self):
        size = documents.mutation_set_size()
        caught = {k: v for k, v in documents.stated_mutation_counts().items()
                  if "caught" in k}
        for where, value in sorted(caught.items()):
            with self.subTest(where=where):
                self.assertLessEqual(value, size)


class TestThePagesFigureIsACountOfSomething(unittest.TestCase):
    """It said "about eighty defects", which nobody could check either way."""

    def test_the_stated_count_is_what_the_tree_holds(self):
        import re

        page = documents.PAGE.read_text(encoding="utf-8")
        stated = re.search(r"have fixed (\d+) defects", page)
        self.assertIsNotNone(stated, "the page stopped stating a countable figure")
        self.assertEqual(int(stated.group(1)), documents.regression_class_count())


class TestThePagesWorkedExampleIsInternallyConsistent(unittest.TestCase):
    """The page invites a reader to check its example by hand, so the three
    ways it states that example must not contradict each other.

    Whether the guide is one the pipeline really produces is a separate
    question, and the instrument answers it by running a design.
    """

    def setUp(self):
        self.shown = documents.page_worked_example()
        self.assertTrue(self.shown, "no worked example found on the page")

    def test_the_spacer_pairs_with_the_site_beside_it(self):
        from cas13guide.util.seq import revcomp

        self.assertEqual(revcomp(self.shown["site"]), self.shown["spacer"])

    def test_the_drawn_duplex_is_that_same_pair(self):
        top, bottom = documents.page_duplex()
        self.assertEqual(top, self.shown["spacer"])
        self.assertEqual(bottom, self.shown["site"][::-1])

    def test_the_printed_gc_is_the_spacers_gc(self):
        spacer = self.shown["spacer"]
        expected = (spacer.count("G") + spacer.count("C")) / len(spacer)
        self.assertAlmostEqual(float(self.shown["gc"]), expected, delta=5e-5)


if __name__ == "__main__":
    unittest.main()
