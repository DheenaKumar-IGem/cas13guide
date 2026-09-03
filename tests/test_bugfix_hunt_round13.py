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


class TestProvenanceIdentifiesTheWorldItSearched(unittest.TestCase):
    """The ledger could not tell two off-target worlds apart.

    The only entry naming the sequence world recorded
    ``ctx.transcriptome.version``, which is ``f"{build}/{annotation}"`` built
    from ``--genome-build``/``--annotation``. Those default to GRCh38/v44 and
    are never compared against the file ``--background`` points at. So the
    world was identified by a caller-asserted string that is constant by
    default, plus a locus count -- and two different corpora holding the same
    number of records produced byte-identical provenance, under a genome build
    nothing had checked, republished verbatim in the HTML report and
    report.json.

    "A guide is specific *against what you gave it*" is the tool's central
    caveat. The record of what you gave it was a number.
    """

    def _world(self, pairs, expression=None):
        from cas13guide.contract.enums import Universe
        from cas13guide.services.transcriptome import Locus, Transcriptome

        world = Transcriptome("GRCh38", "v44")
        for locus_id, sequence in pairs:
            world.add(Locus(locus_id, sequence, Universe.CODING,
                            gene=locus_id, expression=expression))
        return world

    PAIRS = [("L0", "ACGUACGUACGUACGUACGU"), ("L1", "GGGGCCCCAAAAUUUUGGGG")]
    OTHER = [("L0", "UUUUAAAACCCCGGGGUUUU"), ("L1", "AGAGAGAGCUCUCUCUAGAG")]

    def test_two_corpora_of_the_same_size_are_distinguishable(self):
        """The defect exactly: same count, different sequences."""
        one, two = self._world(self.PAIRS), self._world(self.OTHER)
        self.assertEqual(len(one), len(two))
        self.assertEqual(one.version, two.version)
        self.assertNotEqual(one.digest, two.digest)

    def test_load_order_does_not_change_the_digest(self):
        """Otherwise the same corpus would look like two, and a digest that
        changes for no reason is one nobody can use."""
        self.assertEqual(
            self._world(self.PAIRS).digest,
            self._world(list(reversed(self.PAIRS))).digest,
        )

    def test_abundance_is_part_of_the_world(self):
        """The hazard product reads expression, so two worlds that differ only
        in abundance give different answers and must not share an identity."""
        self.assertNotEqual(
            self._world(self.PAIRS).digest,
            self._world(self.PAIRS, expression=12.0).digest,
        )

    def test_an_added_locus_changes_it(self):
        extra = self.PAIRS + [("L2", "ACGUACGUACGUACGUACGU")]
        self.assertNotEqual(self._world(self.PAIRS).digest,
                            self._world(extra).digest)

    def test_the_asserted_build_is_still_recorded_as_asserted(self):
        """The build is what the caller said; the digest is what was loaded.
        Both are kept, and only one of them is a measurement."""
        world = self._world(self.PAIRS)
        self.assertEqual(world.version, "GRCh38/v44")
        self.assertNotEqual(world.version, world.digest)


class TestTheExportHeaderSaysWhatWasSearched(unittest.TestCase):
    """``provenance_lines`` filtered the ledger to model/calibration/search.

    So the ``transcriptome`` records -- the only entries naming the world, its
    locus count and which index ran over it -- were silently dropped, and even
    where the ledger *could* distinguish two runs the exported header could
    not. A CSV outlives the session that made it, and this is the file whose
    whole job is to say how it was produced.

    ``exclusion`` records were dropped the same way, and ``provenance.py``'s
    own docstring says a silent exclusion is the bug.
    """

    def _report(self, digest: str, loci: int = 5) -> dict:
        return {
            "spec_fingerprint": "abc123",
            "provenance": {"records": [
                {"kind": "model", "name": "activity", "version": "act/v1",
                 "detail": {}},
                {"kind": "transcriptome", "name": "reference",
                 "version": "GRCh38/v44",
                 "detail": {"loci": loci, "digest": digest,
                            "expression": "expression/static/v1",
                            "build_verified": False,
                            "index": "index/exhaustive-scan/v1",
                            "index_choice": "scan beats FM here"}},
                {"kind": "exclusion", "name": "collateral cleavage",
                 "version": "-", "detail": {"why": "not predicted for this assay"}},
            ]},
        }

    def _lines(self, *args, **kwargs) -> list[str]:
        from cas13guide.services.fasta import provenance_lines

        return provenance_lines(self._report(*args, **kwargs))

    def test_the_header_names_the_world(self):
        joined = "\n".join(self._lines("93ff8484e3b0"))
        self.assertIn("transcriptome: reference = GRCh38/v44", joined)

    def test_it_carries_the_digest_that_tells_two_worlds_apart(self):
        joined = "\n".join(self._lines("93ff8484e3b0"))
        self.assertIn("93ff8484e3b0", joined)

    def test_two_worlds_give_two_different_headers(self):
        """The point of the whole fix, at the surface a user keeps."""
        self.assertNotEqual(self._lines("93ff8484e3b0"), self._lines("c637e04d9095"))

    def test_it_names_the_expression_source(self):
        self.assertIn("expression/static/v1", "\n".join(self._lines("d")))

    def test_a_deliberate_exclusion_travels_with_the_file(self):
        joined = "\n".join(self._lines("d"))
        self.assertIn("excluded: collateral cleavage", joined)
        self.assertIn("not predicted for this assay", joined)

    def test_the_model_lines_are_unchanged(self):
        self.assertIn("model: activity = act/v1", "\n".join(self._lines("d")))

    def test_no_line_carries_an_embedded_newline(self):
        """The header is comment lines a parser skips by prefix, so a newline
        inside a value puts the rest on an unprefixed line and it reads as
        data. Adding record kinds must not reintroduce that."""
        for line in self._lines("93ff8484e3b0"):
            with self.subTest(line=line[:40]):
                self.assertNotIn("\n", line)


class TestTheLedgerRecordsTheAbundanceEvidence(unittest.TestCase):
    """``ExpressionSource.version`` was declared and never read.

    ``ModelRegistry.versions()`` has no expression slot and the per-context
    record wrote transcriptome version, locus count, index version and index
    rationale -- not the expression source. Every other seam is recorded:
    activity, accessibility, energy, kinetics, calibration and its source dict,
    hazard weights, qc_self_fold, the search bound. This was the one input to
    the hazard product that left no trace, against provenance.py's opening
    claim that every parameter is recorded so a result can be reproduced and
    diffed.
    """

    TARGET = (
        "GGACAUCGAAUCGCGGAACUUGGCAUCAAGCUAGCUAGGCAUCGAUCGAUCAGCUAGCAUCGAUCGGAUCC"
        "AUGGAUUCCAGGCACUUCUCGAUCAUAGGACUCGGUAAUGCCAUGAGUCGAUGGACUUCAGUGCCAUCGAU"
    )

    def _records(self):
        import pathlib
        import tempfile

        from cas13guide.services.workflow import DesignRequest, run_request

        with tempfile.TemporaryDirectory() as directory:
            fasta = pathlib.Path(directory) / "t.fa"
            fasta.write_text(f">T1\n{self.TARGET}\n", encoding="utf-8")
            outcome = run_request(DesignRequest(
                fasta_path=str(fasta), task="knockdown",
                assay="cell_culture_knockdown", effector="RfxCas13d",
                use_structure=False, use_signal=False,
            ))
        return [record for record in outcome.report["provenance"]["records"]
                if record.get("kind") == "transcriptome"]

    def test_the_expression_source_version_is_recorded(self):
        records = self._records()
        self.assertTrue(records, "a run must record its world")
        for record in records:
            with self.subTest(context=record["name"]):
                self.assertTrue(record["detail"].get("expression"))

    def test_the_world_digest_is_recorded(self):
        for record in self._records():
            with self.subTest(context=record["name"]):
                self.assertRegex(record["detail"].get("digest", ""), r"^[0-9a-f]{12}$")

    def test_the_ledger_says_the_build_was_never_verified(self):
        """The claim GRCh38/v44 is the caller's, not the tool's. Recording it
        without saying so is how an unchecked assertion gets republished as a
        finding."""
        for record in self._records():
            with self.subTest(context=record["name"]):
                self.assertIs(record["detail"].get("build_verified"), False)


if __name__ == "__main__":
    unittest.main()
