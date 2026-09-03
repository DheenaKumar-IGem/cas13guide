"""A thirteenth pass, aimed at modules no finder had ever been pointed at.

Round twelve's lesson was that re-searching searched ground pays badly. So this
round's dimensions were chosen by coverage of *attention* rather than by
coverage of lines, and it was the richest yet -- seven confirmed.

The three fixed here share a shape worth naming. Each is a rule that was
derived correctly in one place and not applied in the place beside it:

* ``Interval.unknown`` for expression had a ceiling that was not a bound, so
  the hazard axis rewarded deleting a measurement;
* ``PFSPolicy`` exempted a flank of ``None`` as "no evidence" and did not
  exempt ``N``, which is the same thing spelled differently;
* ``longest_self_complement`` was taught G-U wobbles and
  ``longest_cross_complement``, twenty lines below it in the same file, was
  not -- so the two functions disagreed about what a base pair is, and the one
  feeding a *hard* flag was the stricter.

The third carried a second defect inside the fix, which is the more useful
lesson: a pairing rule and a helix-length threshold are one calibration. Making
the scan correct without moving the threshold traded a hazard the tool missed
for a hard flag firing on 15-48% of random spacers. Both halves are pinned
below.
"""

import random
import unittest

from helpers import *  # noqa: F401,F403


class TestMaximalIgnoranceIsActuallyABound(unittest.TestCase):
    """Deleting a measurement must never make a guide look safer.

    ``expression_interval(None)`` returned ``[0, abundance_factor(1000)]`` =
    ``[0, 0.952]``, described as maximal ignorance. It was not:
    ``abundance_factor`` is monotone with supremum 1, so every locus above
    ~800 TPM has an upper bound *above* that ceiling -- 0.9955 at 9,000 TPM.

    Hazard reads expression at ``.upper``, so the consequence ran the one
    direction a hazard bound must never move. A guide with an off-target on a
    highly expressed locus failed the hazard budget when the abundance was
    known and survived as REVIEW when it was not. The shipped demo world alone
    carries rRNA at 9,000 TPM and mito at 2,500, the two biotypes with the
    highest hazard weights.
    """

    GRID = [0.0, 1.0, 12.5, 50.0, 200.0, 800.0, 2_500.0, 9_000.0, 1e5, 1e6]

    def test_the_unknown_interval_contains_every_measurement(self):
        from cas13guide.services.context import expression_interval

        unknown = expression_interval(None)
        for tpm in self.GRID:
            measured = expression_interval(tpm)
            with self.subTest(tpm=tpm):
                self.assertGreaterEqual(unknown.upper, measured.upper)
                self.assertLessEqual(unknown.lower, measured.lower)

    def test_ignorance_is_never_cheaper_than_knowledge(self):
        """The property stated as the pipeline reads it: hazard takes .upper,
        so no measurement may exceed the unmeasured ceiling."""
        from cas13guide.services.context import expression_interval

        ceiling = expression_interval(None).upper
        worst = max(expression_interval(tpm).upper for tpm in self.GRID)
        self.assertGreaterEqual(ceiling, worst)

    def test_a_measured_locus_is_still_narrower_than_an_unmeasured_one(self):
        """The fix must not make every locus maximally hazardous -- then the
        measurement would buy nothing and the axis would be decoration."""
        from cas13guide.services.context import expression_interval

        for tpm in self.GRID:
            with self.subTest(tpm=tpm):
                measured = expression_interval(tpm)
                self.assertLess(measured.upper - measured.lower, 1.0)

    def test_the_unmeasured_interval_is_flagged_as_missing_evidence(self):
        """What tells a user this is an evidence gap rather than a measured
        hazard, now that it reads at the top of the range."""
        from cas13guide.services.context import expression_interval

        self.assertIn("unmeasured", expression_interval(None).stratum)


class TestAnUnreadBaseIsNotADisfavouredBase(unittest.TestCase):
    """``None`` was exempt from the PFS penalty and ``N`` was not.

    ``preference()``'s own docstring said "A flank of None ... is not counted
    against the candidate. Absence of evidence is not a disfavoured base." An
    ``N`` is the same situation -- an assembly gap or a masked base -- and it
    fell through the branch a genuinely disfavoured G takes: on the default
    effector, a fully synthesisable spacer took a PFS_DISFAVOURED warning and a
    0.55 factor because the base beyond its window had not been sequenced.

    On the hard axis it was worse: PspCas13b's required PFS made ``check()``
    return False for an unread flank, which is a deterministic incompatibility
    claim about a base nobody has read.

    ``N`` is the only ambiguity code that can reach here -- the FASTA loader
    normalises unreadable characters to ``N`` and refuses every other code --
    so this is the complete rule, not a special case for one letter.
    """

    def _policies(self):
        from cas13guide.contract.effectors import EFFECTORS

        return [(name.value, eff.pfs) for name, eff in EFFECTORS.items()]

    def test_an_unread_flank_costs_nothing_on_the_soft_axis(self):
        for name, policy in self._policies():
            for five, three in ((None, "N"), ("N", None), ("N", "N")):
                with self.subTest(effector=name, flanks=(five, three)):
                    self.assertEqual(policy.preference(five, three), 1.0)
                    self.assertTrue(policy.is_preferred(five, three))

    def test_an_unread_flank_is_not_a_hard_incompatibility(self):
        for name, policy in self._policies():
            with self.subTest(effector=name):
                self.assertTrue(policy.check("N", "N"))

    def test_it_scores_exactly_as_no_flank_at_all(self):
        """The two are the same claim, so they must produce the same number."""
        for name, policy in self._policies():
            with self.subTest(effector=name):
                self.assertEqual(
                    policy.preference(None, "N"), policy.preference(None, None))
                self.assertEqual(policy.check(None, "N"), policy.check(None, None))

    def test_a_measured_disfavoured_base_is_still_penalised(self):
        """The exemption must not disable the penalty it is carved out of."""
        from cas13guide.contract.effectors import DISFAVOURED_PFS_FACTOR, EFFECTORS
        from cas13guide.contract.enums import Effector

        lwa = EFFECTORS[Effector.LWA_CAS13A].pfs
        self.assertEqual(lwa.preference(None, "G"), DISFAVOURED_PFS_FACTOR)
        self.assertEqual(lwa.preference(None, "A"), 1.0)

    def test_lowercase_is_read_the_same_way(self):
        from cas13guide.contract.effectors import EFFECTORS
        from cas13guide.contract.enums import Effector

        lwa = EFFECTORS[Effector.LWA_CAS13A].pfs
        self.assertEqual(lwa.preference(None, "n"), 1.0)


class TestBothScansAgreeWhatABasePairIs(unittest.TestCase):
    """``longest_cross_complement`` was Watson-Crick while its neighbour was not.

    ``HELIX_PAIRS`` carries the comment "the rule is derived here once and used
    everywhere", and it was true of every module except the function twenty
    lines below the constant. A single G-U wobble inside a spacer:DR helix
    split it into two shorter ones, so a spacer that really does sequester its
    own direct repeat measured as not doing so -- under-detecting a hazard that
    fires a *hard* flag.
    """

    COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}

    def _dr(self) -> str:
        from cas13guide.contract.effectors import EFFECTORS
        from cas13guide.contract.enums import Effector

        return EFFECTORS[Effector.RFX_CAS13D].direct_repeat

    def test_a_wobble_does_not_split_one_helix_into_two(self):
        from cas13guide.pipeline.b_qc import longest_cross_complement

        # A perfect 12 nt antiparallel partner of the repeat's first 12 nt,
        # with one C->U making a U-G wobble where a C-G pair was.
        partner = "".join(self.COMPLEMENT[b] for b in reversed(self._dr()[:12]))
        index = partner.index("C")
        wobbled = partner[:index] + "U" + partner[index + 1:]

        self.assertEqual(longest_cross_complement(partner, self._dr()), 12)
        self.assertEqual(longest_cross_complement(wobbled, self._dr()), 12)

    def test_the_two_scans_in_this_module_use_one_pairing_rule(self):
        """The structural invariant. Both must accept G-U, or the file
        contains two definitions of a base pair again."""
        from cas13guide.pipeline.b_qc import (
            longest_cross_complement,
            longest_self_complement,
        )

        # GGGG against UUUU is a helix only if wobbles count.
        self.assertGreaterEqual(longest_cross_complement("GGGG", "UUUU"), 4)
        self.assertGreaterEqual(longest_self_complement("GGGG" + "AAA" + "UUUU"), 4)

    def test_it_pairs_antiparallel(self):
        """A parallel match is not a duplex: GGAA pairs with UUCC only when the
        second strand is read the other way round."""
        from cas13guide.pipeline.b_qc import longest_cross_complement

        self.assertEqual(longest_cross_complement("GGAA", "UUCC"), 4)
        self.assertLess(longest_cross_complement("GGAA", "CCUU"), 4)

    def test_dna_and_case_are_normalised(self):
        from cas13guide.pipeline.b_qc import longest_cross_complement

        self.assertEqual(longest_cross_complement("ggaa", "TTCC"), 4)


class TestTheThresholdIsCalibratedToThePairingRule(unittest.TestCase):
    """The defect that was inside the fix for the defect above.

    ``MAX_DR_STEM`` was derived when the scan was Watson-Crick only, where it
    flagged 2.1-2.3% of random 22-mers against all three shipped repeats.
    Teaching the scan wobbles roughly doubles the helix a random spacer reaches
    by chance, so at the old threshold of six the same constant flagged 39.3%
    of random spacers against the RfxCas13d repeat and 47.5% against PspCas13b.

    ``DR_SPACER_INTERACTION`` is a hard flag -- the candidate is dropped, not
    ranked down -- so that would have deleted a third of every candidate set,
    for no reason, by an amount depending mostly on which repeat the ortholog
    happens to carry. It is what ``contract/effectors.py`` describes at length
    about hard-vs-soft PFS, reached from the other side.

    This is the guard: if the pairing rule is ever widened again without the
    threshold moving with it, the false-positive budget blows and these fail.
    """

    #: The Watson-Crick rule at its own threshold flagged 2.1-2.3%. Six times
    #: that is far tighter than the 15-48% the mismatch produced, and loose
    #: enough not to fail on sampling noise at this n.
    BUDGET = 0.06

    COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}

    def _rate(self, repeat: str, seed: int, n: int = 4_000) -> float:
        from cas13guide.pipeline.b_qc import MAX_DR_STEM, longest_cross_complement

        rng = random.Random(seed)
        hits = 0
        for _ in range(n):
            spacer = "".join(rng.choice("AUGC") for _ in range(22))
            hits += longest_cross_complement(spacer, repeat) > MAX_DR_STEM
        return hits / n

    def test_a_random_spacer_is_rarely_flagged_against_any_shipped_repeat(self):
        from cas13guide.contract.effectors import EFFECTORS

        for name, params in EFFECTORS.items():
            with self.subTest(effector=name.value):
                rate = self._rate(params.direct_repeat, seed=12345)
                self.assertLess(rate, self.BUDGET, f"{name.value} flags {rate:.1%}")

    def test_a_spacer_built_to_sequester_its_repeat_is_still_flagged(self):
        """The other half. A threshold high enough to be quiet on noise must
        still catch the thing the flag exists for -- including through a
        wobble, which is what the Watson-Crick scan could not do at any
        threshold."""
        from cas13guide.contract.effectors import EFFECTORS
        from cas13guide.pipeline.b_qc import MAX_DR_STEM, longest_cross_complement

        for name, params in EFFECTORS.items():
            repeat = params.direct_repeat
            for start in range(0, len(repeat) - 12, 4):
                segment = repeat[start:start + 12]
                partner = "".join(self.COMPLEMENT[b] for b in reversed(segment))
                if "C" not in partner:
                    continue
                index = partner.index("C")
                wobbled = partner[:index] + "U" + partner[index + 1:]
                with self.subTest(effector=name.value, start=start):
                    self.assertGreater(
                        longest_cross_complement(wobbled, repeat), MAX_DR_STEM)

    def test_the_threshold_is_documented_as_a_rate_not_a_length(self):
        """A constant whose derivation is not written down is the thing that
        went stale here. If the number moves, the reason must move with it."""
        import inspect

        from cas13guide.pipeline import b_qc

        source = inspect.getsource(b_qc)
        head = source[: source.index("MAX_DR_STEM = ")]
        self.assertIn("one calibration", head)
        self.assertIn("MAX_SELF_STEM", source)


if __name__ == "__main__":
    unittest.main()
