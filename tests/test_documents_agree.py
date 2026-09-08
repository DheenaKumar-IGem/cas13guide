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
    """Five places say how many tests there are. They must be one number.

    It was three until a fourth turned up stating 930 -- the ``unittest
    discover`` line, which the scraper's pytest-anchored pattern never saw,
    four lines above prose promising that both runners discover the same set.

    Then a fifth turned up saying 1241 while the other four said 1473: the
    mutation sweep's own header, describing what it does to a survivor. The
    instrument written to catch documentation drift had drifted, in the same
    sentence whose *other* number -- the mutation count -- was scraped and
    correct. Both are checked now.
    """

    def test_every_document_states_it(self):
        counts = documents.documented_test_counts()
        self.assertEqual(len(counts), documents.EXPECTED_TEST_SOURCES,
                         f"found {sorted(counts)}")

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


class TestTheCoverageFigureIsStatedConsistently(unittest.TestCase):
    """Three statements of one number, in one file, and nothing compared them.

    That is the shape the test count was in when it drifted across three places,
    and the shape the flip rate was in when five files agreed on a stale figure.
    Only agreement is checked here: whether they agree on the *right* number
    needs ``examples/measure_coverage.py``, which traces subprocesses through a
    full suite run. Same division of labour as the flip rate.
    """

    def test_every_statement_is_still_there(self):
        stated = documents.stated_coverage_figures()
        self.assertEqual(len(stated), documents.EXPECTED_COVERAGE_FIGURES,
                         f"found {sorted(stated.items())}")

    def test_they_agree_with_each_other(self):
        stated = documents.stated_coverage_figures()
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

    def test_no_source_has_dropped_out_of_the_comparison(self):
        """Pinned like the test count and the flip rate. A document reworded
        past its regex would otherwise leave the comparison silently, and the
        check would agree with whatever remained -- vacuous agreement being the
        failure this whole file exists to stop, applied to itself."""
        self.assertEqual(len(documents.stated_mutation_counts()),
                         documents.EXPECTED_MUTATION_SOURCES,
                         sorted(documents.stated_mutation_counts()))

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


class TestTheCountsStatedInWordsAreTheTreesCounts(unittest.TestCase):
    """Counts that move when somebody does ordinary work.

    Adding a regression test, running another review pass, or writing a test
    that needs the miRBase file each moves a number the README spells out in
    one place and derives in none. They were all correct when checked by hand;
    the point is that checking by hand is what had to happen.
    """

    def test_each_one_is_what_the_tree_holds(self):
        for label, stated, actual in documents.documented_word_counts():
            with self.subTest(count=label):
                self.assertIsNotNone(stated, "the sentence stating it is gone")
                self.assertEqual(stated, actual)


class TestTheVocabularySizeIsTheVocabularys(unittest.TestCase):
    """"52 of 52 things it can say" is two numbers with different lifetimes.

    The numerator needs a whole suite run and stays the vocabulary
    instrument's job. The denominator is the size of four vocabularies, and
    is the half that drifts: adding a reason code or a caveat is a local edit
    two files away from the figure describing it.
    """

    def test_the_denominator_is_the_size_of_the_vocabulary(self):
        _produced, of_how_many, size = documents.documented_vocabulary()
        self.assertIsNotNone(of_how_many, "the figure stating it is gone")
        self.assertEqual(of_how_many, size)

    def test_it_does_not_claim_more_produced_than_exist(self):
        produced, _of_how_many, size = documents.documented_vocabulary()
        self.assertIsNotNone(produced, "the figure stating it is gone")
        self.assertLessEqual(produced, size)


class TestEveryFileTheDocumentsNameExists(unittest.TestCase):
    """179 citations across two documents, and a rename orphans them silently.

    This is the failure the mutation sweep had when its screening hints named
    modules that no longer existed: a pointer that resolves to nothing reads
    exactly like one that resolves, until somebody follows it.
    """

    def test_no_citation_dangles(self):
        dangling = documents.unresolvable_file_citations()
        self.assertEqual(
            dangling, [],
            "; ".join(f"{name} cited {count}x" for name, count in dangling))


class TestThePageQuotesTheToolCorrectly(unittest.TestCase):
    """What the landing page prints as the tool's own words must be them.

    Every other check in this module compares two *documents*. This one
    compares a document against the package, which is the only way a
    quotation can be checked at all: two documents can agree with each other
    and both be wrong, which is exactly what happened when the page and the
    tool both named nine sequence universes for a corpus holding one.

    `verify_the_documented_numbers.py` has done this since it was written, and
    nothing ran it. Its counts are mirrored into the tests above; its
    quotations were not, so a limitation's wording could change and a full
    green suite would say nothing. That is not hypothetical -- it is how the
    page came to quote a superseded remedy for two commits.
    """

    @classmethod
    def setUpClass(cls):
        from verify_the_documented_numbers import (
            normalise,
            page_critical_quotes,
            tool_critical_limitations,
        )

        cls.normalise = staticmethod(normalise)
        cls.page = page_critical_quotes()
        cls.tool = tool_critical_limitations()

    def test_the_page_quotes_as_many_as_the_tool_prints(self):
        self.assertEqual(
            len(self.page), len(self.tool),
            f"the page quotes {len(self.page)} CRITICAL limitations and a run "
            f"prints {len(self.tool)}",
        )

    def test_every_quotation_is_whole_and_is_something_the_tool_says(self):
        """Membership of the exact string, which is both checks at once.

        An abridged quote is not a quote the tool says, so one assertion
        covers wording and completeness together -- and completeness is the
        half that matters most here: when the page last softened these, what
        went each time was the sharpest consequence and the remedy.
        """
        printed = [self.normalise(text) for text in self.tool]
        for quote in self.page:
            with self.subTest(quote=quote[:60]):
                self.assertIn(
                    self.normalise(quote), printed,
                    "the page attributes this to the tool, and the tool does "
                    "not say it -- in these words, or in this much of them",
                )


if __name__ == "__main__":
    unittest.main()
