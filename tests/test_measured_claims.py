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
import pathlib
import random
import re
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


class TestTheSignalDegeneracyFigure(unittest.TestCase):
    """"the ratio varies by 0.159% across the unit square".

    That figure is why ``predicted_signal`` is reported and not ranked on: with
    the shipped constants the ODE sits in its linear regime, so the output is a
    constant multiple of ``activity_lower * accessibility`` -- two axes already
    in the objective set -- and adding it silently re-weights them. If the
    constants or the integrator ever move the model out of that regime, the
    figure grows and the axis should return; ``adds_information`` probes for
    exactly that, so this pins the number the probe is calibrated against.
    """

    def test_the_ratio_is_flat_to_about_a_sixth_of_a_percent(self):
        from cas13guide.services.kinetics import CollateralKineticsModel

        model = CollateralKineticsModel()
        ratios = []
        for i in range(21):
            for j in range(21):
                activity = i / 20 or 0.001
                accessibility = j / 20 or 0.001
                signal = model.predict(activity, accessibility).relative_signal
                ratios.append(signal / (activity * accessibility))
        spread = (max(ratios) - min(ratios)) / max(ratios)
        self.assertAlmostEqual(spread, 0.0016, delta=0.0008)

    def test_and_the_pipeline_therefore_declines_to_rank_on_it(self):
        """The consequence the figure exists to justify."""
        from cas13guide.services.kinetics import CollateralKineticsModel

        self.assertFalse(CollateralKineticsModel().adds_information())


class TestTheDirectRepeatIsWorthGettingRight(unittest.TestCase):
    """"the old value changed that verdict for 4.9% of random 22 nt spacers".

    The DR is data that looks like decoration, and a wrong one does not fail
    loudly -- it moves verdicts. This recomputes the figure that says how much,
    against the superseded value recovered from the first commit.
    """

    #: The corrupted RfxCas13d repeat as it shipped in 8cbe60f, before the Arc
    #: weights showed it disagreed with the published CasRx mature repeat.
    SUPERSEDED = "CAACUAAACCCUAUAAGAAACGAAAUGAGAACCCAAUUCAUCGCU"

    def test_the_verdict_flip_rate_is_about_five_percent(self):
        from cas13guide.contract.effectors import EFFECTORS
        from cas13guide.contract.enums import Effector
        from cas13guide.pipeline.b_qc import MAX_DR_STEM, longest_cross_complement

        current = EFFECTORS[Effector.RFX_CAS13D].direct_repeat
        self.assertNotEqual(current, self.SUPERSEDED)
        rng = random.Random(7)
        flips = 0
        n = 4_000
        for _ in range(n):
            spacer = "".join(rng.choice("AUGC") for _ in range(22))
            before = longest_cross_complement(spacer, self.SUPERSEDED) > MAX_DR_STEM
            after = longest_cross_complement(spacer, current) > MAX_DR_STEM
            flips += before != after
        self.assertAlmostEqual(flips / n, 0.049, delta=0.02)

    def test_the_flip_goes_in_both_directions(self):
        """"in both directions" -- a DR error is not simply more or less strict."""
        from cas13guide.contract.effectors import EFFECTORS
        from cas13guide.contract.enums import Effector
        from cas13guide.pipeline.b_qc import MAX_DR_STEM, longest_cross_complement

        current = EFFECTORS[Effector.RFX_CAS13D].direct_repeat
        rng = random.Random(11)
        gained = lost = 0
        for _ in range(4_000):
            spacer = "".join(rng.choice("AUGC") for _ in range(22))
            before = longest_cross_complement(spacer, self.SUPERSEDED) > MAX_DR_STEM
            after = longest_cross_complement(spacer, current) > MAX_DR_STEM
            gained += after and not before
            lost += before and not after
        self.assertGreater(gained, 0)
        self.assertGreater(lost, 0)


class TestTheReadmeDoesNotContradictItself(unittest.TestCase):
    """The Verification section states its test count twice, and the two drifted.

    Three times in one session the prose line said one number and the code
    block six lines below it said another -- 930 against 976, then 976 against
    1002. Each time the fix was to update one and forget the other, which is
    the signature of a figure with no guard.

    The count *itself* is deliberately not pinned. Collection depends on which
    optional dependencies are installed -- ``test_properties`` raises
    ``SkipTest`` at import without ``hypothesis``, taking forty tests with it --
    so a hard assertion would fail on a contributor's machine for a reason that
    has nothing to do with their change. That is a worse defect than a stale
    number.

    What is checkable without that risk is internal consistency: whatever the
    figure is, the document must state it once. So this compares the README to
    itself, and says plainly that it does not verify the number is right.
    """

    @staticmethod
    def _readme() -> str:
        return (pathlib.Path(__file__).resolve().parent.parent
                / "README.md").read_text(encoding="utf-8")

    def test_the_two_stated_test_counts_agree(self):
        text = self._readme()
        prose = re.search(r"^(\d[\d,]*) tests, run under", text, re.MULTILINE)
        block = re.search(r"^pytest\s+# (\d[\d,]*) tests$", text, re.MULTILINE)

        self.assertIsNotNone(prose, "the Verification section states no test count")
        self.assertIsNotNone(block, "the verification commands quote no test count")
        self.assertEqual(
            prose.group(1), block.group(1),
            "the Verification section states its test count twice and the two "
            "disagree; both should be the number `pytest` reports as passing",
        )

    def test_the_figure_is_not_wildly_out_of_date(self):
        """A floor rather than an equality, for the reason in the docstring: it
        catches a number left behind by a hundred tests, not by six."""
        import subprocess
        import sys

        text = self._readme()
        stated = int(re.search(r"^(\d+) tests, run under", text, re.MULTILINE).group(1))
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--collect-only"],
            capture_output=True, text=True,
            cwd=pathlib.Path(__file__).resolve().parent.parent,
        )
        collected = re.search(r"(\d+) tests? collected", result.stdout)
        if collected is None:
            self.skipTest("could not collect the suite to compare against")
        actual = int(collected.group(1))
        self.assertLess(
            abs(actual - stated), max(50, actual // 10),
            f"the README says {stated} tests and the suite collects {actual}",
        )


if __name__ == "__main__":
    unittest.main()
