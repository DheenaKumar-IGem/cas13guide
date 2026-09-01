"""Figures the code states about itself, recomputed.

Nine stale numbers have been corrected across the review passes -- a value
measured once, quoted in a comment, and never checked again while the code
around it moved. Unlike the vacuous-test class, nothing stopped the tenth.

Not every claim is mechanically checkable; 136 numeric claims live in these
comments and most describe a one-off experiment. What *is* checkable is a claim
about the behaviour of code still in the tree, and the self-folding proxy is the
load-bearing one: its recall justifies a hard FAIL that cannot be bought back,
and it is quoted in four places including a limitation a user reads.

Recomputing it found the quoted figures were taken on a sample too small to
support them. At 6,000 random 22-mers only ~53 sequences genuinely fold, so the
recall estimate carried a 95% interval about +/-11% wide; the stated 28.9% is
really 34.9% +/- 3.7% at ten times the sample. The precision claim was worse
than imprecise -- "roughly half its flags being sequences that barely fold at
all" implies precision near 50%, and it is 21%.
"""

import math
import random
import unittest

from cas13guide.pipeline.b_qc import (
    MAX_SELF_STEM,
    SELF_FOLD_MAX_DG,
    longest_self_complement,
)

#: Fixed, so the figures in the comments are reproducible rather than merely
#: plausible. Ten thousand keeps the test near a second while leaving ~100
#: genuine folders behind the recall estimate.
SEED = 12345
N = 10_000


def _sample():
    try:
        import RNA
    except ImportError:      # pragma: no cover - environment dependent
        return None
    rng = random.Random(SEED)
    flagged = folded = both = 0
    for _ in range(N):
        spacer = "".join(rng.choice("AUGC") for _ in range(22))
        proxy = longest_self_complement(spacer) > MAX_SELF_STEM
        real = RNA.fold(spacer)[1] <= SELF_FOLD_MAX_DG
        flagged += proxy
        folded += real
        both += proxy and real
    return flagged, folded, both


class TestTheSelfFoldProxyIsAsBadAsDocumented(unittest.TestCase):
    """The proxy decides an unbuyable-back FAIL, so its weakness is a promise.

    Asserted as intervals rather than as point equality: these are sampling
    estimates, and a test demanding three decimal places of a Monte Carlo
    quantity is a flaky test dressed as a precise one.
    """

    @classmethod
    def setUpClass(cls):
        cls.sample = _sample()

    def setUp(self):
        if self.sample is None:
            self.skipTest("ViennaRNA is not installed")

    def test_the_flag_rate_is_about_two_percent(self):
        flagged, _folded, _both = self.sample
        self.assertAlmostEqual(flagged / N, 0.018, delta=0.006)

    def test_about_one_percent_of_random_spacers_genuinely_fold(self):
        """Calibrates SELF_FOLD_MAX_DG: the threshold is meant to be rare."""
        _flagged, folded, _both = self.sample
        self.assertAlmostEqual(folded / N, 0.011, delta=0.005)

    def test_recall_is_about_a_third_and_nowhere_near_complete(self):
        _flagged, folded, both = self.sample
        recall = both / folded
        self.assertGreater(folded, 50, "too few positives to estimate recall")
        self.assertAlmostEqual(recall, 0.35, delta=0.12)
        # The point of the comment: this is a fallback, not a substitute.
        self.assertLess(recall, 0.6)

    def test_most_of_what_it_flags_does_not_actually_fold(self):
        """The claim that was not merely noisy: "roughly half its flags" implies
        precision near 0.5, and it is nearer 0.2."""
        flagged, _folded, both = self.sample
        precision = both / flagged
        self.assertLess(precision, 0.4)

    def test_the_stated_figures_are_inside_their_own_confidence_interval(self):
        """What went wrong originally: a figure quoted to three significant
        figures off a sample that could not support one."""
        _flagged, folded, both = self.sample
        recall = both / folded
        half_width = 1.96 * math.sqrt(recall * (1 - recall) / folded)
        self.assertLess(half_width, 0.15, "sample too small to quote a recall")
        self.assertLess(abs(recall - 0.349), 0.15)


class TestTheCommentsQuoteTheseFigures(unittest.TestCase):
    """A measured value and the prose beside it must not drift apart, which is
    exactly how the previous nine went stale."""

    @staticmethod
    def _source(*parts):
        from pathlib import Path

        return (Path(__file__).resolve().parents[1] / "src" / "cas13guide"
                ).joinpath(*parts).read_text(encoding="utf-8")

    def test_b_qc_states_the_larger_sample(self):
        text = self._source("pipeline", "b_qc.py")
        self.assertIn("60,000 random 22-mers", text)
        self.assertIn("34.9%", text)

    def test_no_citation_still_quotes_the_superseded_recall(self):
        """It was quoted in four places; all four had to move together."""
        for parts in (("pipeline", "b_qc.py"), ("pipeline", "orchestrator.py"),
                      ("services", "limitations.py")):
            with self.subTest(module="/".join(parts)):
                text = self._source(*parts)
                # The one permitted mention is b_qc's note recording what the
                # figure used to be and why it moved.
                mentions = text.count("28.9")
                allowed = 1 if parts == ("pipeline", "b_qc.py") else 0
                self.assertEqual(mentions, allowed)


if __name__ == "__main__":
    unittest.main()
