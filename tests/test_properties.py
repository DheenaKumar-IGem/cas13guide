"""Property tests: invariants stated once, checked against generated input.

``hypothesis`` was a declared dev dependency that nothing imported -- and was
not even installed, so it had never run. That is a gap worth closing rather than
a formality: almost every load-bearing rule in this package is a *property*
rather than a value, and the example-based suite pins each one at the handful of
points somebody thought of.

The properties here are the ones where a generator genuinely outperforms an
example. They are deliberately about **relationships that must hold for every
input** -- an inverse, a bound, a monotonicity, an ordering, a conservation law
-- rather than about particular numbers, which the rest of the suite already
covers.

Each test names the invariant it defends and, where the invariant exists because
something went wrong once, says so. A property test that nobody can trace back
to a reason is a test nobody will dare delete when it becomes wrong.
"""

import math
import unittest

from helpers import *  # noqa: F401,F403

try:
    from hypothesis import HealthCheck, assume, given, settings
    from hypothesis import strategies as st
except ImportError:  # pragma: no cover - the dev extra is not installed
    raise unittest.SkipTest("hypothesis is not installed; pip install -e '.[dev]'")

from cas13guide.metrics.bounds import Interval, benefit_lower, hazard_product
from cas13guide.util.duplex import profile_duplex
from cas13guide.util.seq import gc_fraction, max_homopolymer, normalize, revcomp, windows

#: Sequences over the alphabet the package actually accepts. ``N`` is included
#: deliberately: it is a legal input character, and every pass that forgot it
#: found a defect there.
RNA = st.text(alphabet="ACGUN", min_size=1, max_size=120)
RNA_NO_N = st.text(alphabet="ACGU", min_size=1, max_size=120)
UNIT = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


class TestSequencePrimitives(unittest.TestCase):
    """The rules every layer above these silently assumes."""

    @given(RNA)
    @SETTINGS
    def test_revcomp_is_its_own_inverse(self, sequence):
        self.assertEqual(revcomp(revcomp(sequence)), sequence)

    @given(RNA)
    @SETTINGS
    def test_revcomp_preserves_length(self, sequence):
        self.assertEqual(len(revcomp(sequence)), len(sequence))

    @given(RNA)
    @SETTINGS
    def test_normalize_is_idempotent(self, sequence):
        once = normalize(sequence)
        self.assertEqual(normalize(once), once)

    @given(st.text(alphabet="ACGUTacgut", min_size=1, max_size=80))
    @SETTINGS
    def test_normalize_transcribes_and_uppercases_and_does_nothing_else(self, sequence):
        result = normalize(sequence)
        self.assertEqual(len(result), len(sequence))
        self.assertNotIn("T", result)
        self.assertEqual(result, sequence.upper().replace("T", "U"))

    @given(RNA)
    @SETTINGS
    def test_gc_fraction_is_a_fraction(self, sequence):
        self.assertGreaterEqual(gc_fraction(sequence), 0.0)
        self.assertLessEqual(gc_fraction(sequence), 1.0)

    @given(RNA)
    @SETTINGS
    def test_max_homopolymer_is_bounded_by_length_and_at_least_one(self, sequence):
        run = max_homopolymer(sequence)
        self.assertGreaterEqual(run, 1)
        self.assertLessEqual(run, len(sequence))

    @given(st.text(alphabet="A", min_size=1, max_size=40))
    @SETTINGS
    def test_a_pure_run_is_its_own_length(self, sequence):
        self.assertEqual(max_homopolymer(sequence), len(sequence))

    @given(RNA, st.integers(min_value=1, max_value=30), st.integers(min_value=1, max_value=5))
    @SETTINGS
    def test_windows_are_in_range_contiguous_and_correctly_counted(self, sequence, size, step):
        got = list(windows(sequence, size, step))
        expected = (
            len(range(0, len(sequence) - size + 1, step)) if len(sequence) >= size else 0
        )
        self.assertEqual(len(got), expected)
        for start, window in got:
            self.assertEqual(len(window), size)
            self.assertEqual(window, sequence[start : start + size])

    @given(RNA, st.integers(min_value=1, max_value=30))
    @SETTINGS
    def test_windows_span_the_whole_sequence_at_step_one(self, sequence, size):
        assume(len(sequence) >= size)
        got = list(windows(sequence, size, 1))
        self.assertEqual(got[0][0], 0)
        self.assertEqual(got[-1][0] + size, len(sequence))


class TestDuplexProfile(unittest.TestCase):
    """A duplex profile is what every activity number is computed from."""

    @given(RNA_NO_N)
    @SETTINGS
    def test_a_spacer_against_its_own_target_is_a_perfect_duplex(self, spacer):
        profile = profile_duplex(spacer, revcomp(spacer))
        self.assertTrue(profile.is_perfect)
        self.assertEqual(profile.n_mismatch, 0)
        self.assertEqual(profile.n_wobble, 0)

    @given(RNA_NO_N)
    @SETTINGS
    def test_counts_never_exceed_the_length(self, spacer):
        profile = profile_duplex(spacer, revcomp(spacer)[::-1])
        self.assertEqual(profile.length, len(spacer))
        self.assertLessEqual(profile.n_mismatch + profile.n_wobble, profile.length)

    @given(RNA_NO_N)
    @SETTINGS
    def test_the_cigar_describes_the_same_duplex_it_summarises(self, spacer):
        profile = profile_duplex(spacer, revcomp(spacer))
        self.assertEqual(
            sum(int(part) for part in "".join(
                character if character.isdigit() else " " for character in profile.cigar
            ).split()),
            profile.length,
        )

    @given(RNA_NO_N, st.integers(min_value=0, max_value=119))
    @SETTINGS
    def test_one_substitution_costs_at_most_one_position(self, spacer, index):
        """A single changed base cannot disturb a second position.

        Cheap to state and worth stating: the wobble rule means a substitution
        may be scored as a mismatch *or* as a wobble, and an implementation that
        mis-indexed would show up here rather than in a hand-picked example.
        """
        assume(len(spacer) >= 1)
        position = index % len(spacer)
        site = list(revcomp(spacer))
        site[position] = {"A": "C", "C": "A", "G": "U", "U": "G"}[site[position]]
        profile = profile_duplex(spacer, "".join(site))
        self.assertLessEqual(profile.n_mismatch + profile.n_wobble, 1)


class TestTheBoundAlgebra(unittest.TestCase):
    """The asymmetry is the whole design: benefit at LCB, hazard at UCB."""

    @staticmethod
    def _interval(draw_lower, draw_point, draw_upper):
        low, mid, high = sorted((draw_lower, draw_point, draw_upper))
        return Interval(point=mid, lower=low, upper=high)

    @given(UNIT, UNIT, UNIT)
    @SETTINGS
    def test_an_interval_never_reports_a_benefit_above_its_point(self, a, b, c):
        interval = self._interval(a, b, c)
        self.assertLessEqual(benefit_lower(interval), interval.point)

    @given(UNIT, UNIT, UNIT)
    @SETTINGS
    def test_hazard_is_never_below_the_point_estimate(self, a, b, c):
        interval = self._interval(a, b, c)
        self.assertGreaterEqual(hazard_product(interval), interval.point)

    @given(UNIT, UNIT, UNIT, UNIT, UNIT, UNIT)
    @SETTINGS
    def test_hazard_product_is_monotone_in_every_factor(self, a, b, c, d, e, f):
        """Raising any factor's upper bound cannot lower the hazard.

        The defect this defends against is fix #6, recorded in the README: the
        original bounded activity and passed expression through as a point
        estimate, which understated hazard on the axis with the most
        uncertainty.
        """
        first = self._interval(a, b, c)
        second = self._interval(d, e, f)
        wider = Interval(point=second.point, lower=second.lower, upper=1.0)
        self.assertGreaterEqual(
            hazard_product(first, wider), hazard_product(first, second) - 1e-12
        )

    @given(UNIT, UNIT, UNIT)
    @SETTINGS
    def test_clamping_preserves_ordering_and_stays_inside_the_box(self, a, b, c):
        clamped = self._interval(a, b, c).clamped(0.2, 0.8)
        self.assertLessEqual(clamped.lower, clamped.point)
        self.assertLessEqual(clamped.point, clamped.upper)
        for value in (clamped.lower, clamped.point, clamped.upper):
            self.assertGreaterEqual(value, 0.2)
            self.assertLessEqual(value, 0.8)

    @given(UNIT, UNIT, UNIT)
    @SETTINGS
    def test_an_inverted_interval_is_refused(self, a, b, c):
        low, _mid, high = sorted((a, b, c))
        assume(high > low)
        with self.assertRaises(ValueError):
            Interval(point=low, lower=high, upper=low)

    @given(UNIT, UNIT)
    @SETTINGS
    def test_width_is_never_negative(self, a, b):
        low, high = sorted((a, b))
        self.assertGreaterEqual(Interval(point=low, lower=low, upper=high).width, 0.0)


class TestTheConformalQuantile(unittest.TestCase):
    """Split conformal's guarantee is only as good as the quantile under it."""

    @given(st.lists(UNIT, min_size=25, max_size=200),
           st.floats(min_value=0.01, max_value=0.99, allow_nan=False))
    @SETTINGS
    def test_the_quantile_lies_within_the_observed_residuals(self, observations, alpha):
        # At least MIN_STRATUM_N rows, or the service declines to resolve the
        # stratum at all -- which is correct behaviour and simply not what this
        # property is about.
        from cas13guide.services.calibration import CalibrationService

        service = CalibrationService()
        key = CalibrationService.stratum_key("RfxCas13d", "cell_culture_knockdown", "mm0")
        for index, value in enumerate(observations):
            service.add_observation(key, 0.5, value, group=f"T{index % 9}")
        stratum = service.resolve(key)
        self.assertIsNotNone(stratum)
        quantile, _certified = stratum.quantile(alpha)
        residuals = stratum.sorted_residuals()
        self.assertGreaterEqual(quantile, min(residuals))
        self.assertLessEqual(quantile, max(residuals))

    @given(st.lists(UNIT, min_size=30, max_size=120))
    @SETTINGS
    def test_a_smaller_alpha_never_gives_a_narrower_interval(self, observations):
        """Asking for more confidence cannot buy a tighter bound."""
        from cas13guide.services.calibration import CalibrationService

        service = CalibrationService()
        key = CalibrationService.stratum_key("RfxCas13d", "cell_culture_knockdown", "mm0")
        for index, value in enumerate(observations):
            service.add_observation(key, 0.5, value, group=f"T{index % 9}")
        stratum = service.resolve(key)
        self.assertIsNotNone(stratum)
        strict, _ = stratum.quantile(0.01)
        loose, _ = stratum.quantile(0.20)
        self.assertGreaterEqual(strict, loose)


class TestThePanelPosterior(unittest.TestCase):
    """FR6.3's arithmetic, over the whole parameter range rather than at a point."""

    @given(st.floats(min_value=0.5, max_value=400.0, allow_nan=False),
           st.floats(min_value=0.5, max_value=400.0, allow_nan=False),
           st.floats(min_value=0.01, max_value=0.99, allow_nan=False))
    @SETTINGS
    def test_the_beta_quantile_inverts_its_own_cdf(self, a, b, p):
        from cas13guide.services.panel_confidence import (
            beta_quantile,
            regularized_incomplete_beta,
        )

        x = beta_quantile(a, b, p)
        self.assertGreaterEqual(x, 0.0)
        self.assertLessEqual(x, 1.0)
        self.assertAlmostEqual(regularized_incomplete_beta(a, b, x), p, places=6)

    @given(st.floats(min_value=0.5, max_value=200.0, allow_nan=False),
           st.floats(min_value=0.5, max_value=200.0, allow_nan=False),
           UNIT, UNIT)
    @SETTINGS
    def test_the_beta_cdf_is_monotone(self, a, b, x, y):
        from cas13guide.services.panel_confidence import regularized_incomplete_beta

        low, high = sorted((x, y))
        self.assertLessEqual(
            regularized_incomplete_beta(a, b, low),
            regularized_incomplete_beta(a, b, high) + 1e-12,
        )

    @given(st.lists(st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
                    min_size=1, max_size=6),
           st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @SETTINGS
    def test_the_point_estimate_stays_inside_its_own_interval(self, probabilities, beta):
        """The clamp exists for this, and the property is the reason.

        A point estimate outside the bounds it is reported beside is wrong
        wherever it is read, and the beta-factor form can leave them for panels
        whose members are likely to fail.
        """
        from cas13guide.services.panel_confidence import (
            _combine_beta_factor,
            _combine_independent,
        )

        point = _combine_beta_factor(list(probabilities), beta)
        self.assertGreaterEqual(point, max(probabilities) - 1e-12)
        self.assertLessEqual(point, _combine_independent(list(probabilities)) + 1e-12)

    @given(st.lists(st.floats(min_value=0.05, max_value=0.95, allow_nan=False),
                    min_size=2, max_size=5))
    @SETTINGS
    def test_more_shared_failure_never_makes_a_panel_look_better(self, probabilities):
        from cas13guide.services.panel_confidence import _combine_beta_factor

        values = [
            _combine_beta_factor(list(probabilities), beta / 10.0) for beta in range(11)
        ]
        for earlier, later in zip(values, values[1:]):
            self.assertLessEqual(later, earlier + 1e-12)

    @given(st.lists(st.floats(min_value=0.01, max_value=0.99, allow_nan=False),
                    min_size=1, max_size=6))
    @SETTINGS
    def test_the_independent_ceiling_is_never_below_the_best_member(self, probabilities):
        from cas13guide.services.panel_confidence import _combine_independent

        self.assertGreaterEqual(
            _combine_independent(list(probabilities)), max(probabilities) - 1e-12
        )


class TestSearchIndexesAgree(unittest.TestCase):
    """The exhaustive indexes must return the same hit set, on any world.

    A differential check rather than a fixture: the two implementations share
    no code, so agreeing on generated input is real evidence and agreeing on a
    hand-written example is not.
    """

    @given(st.lists(RNA_NO_N.filter(lambda s: len(s) >= 25), min_size=1, max_size=3),
           st.integers(min_value=0, max_value=4))
    @settings(max_examples=40, deadline=None,
              suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
    def test_the_scanner_and_the_fm_index_find_the_same_sites(self, sequences, mismatches):
        from cas13guide.contract.enums import Universe
        from cas13guide.services.fmindex import FMIndexSearchIndex
        from cas13guide.services.indexes import ExhaustiveScanIndex
        from cas13guide.services.transcriptome import Locus, Transcriptome

        world = Transcriptome("GRCh38", "v44")
        for index, sequence in enumerate(sequences):
            world.add(Locus(f"L{index}", sequence, Universe.CODING, expression=1.0))
        universes = frozenset({Universe.CODING})
        spacer = revcomp(sequences[0][:22])
        assume(len(spacer) == 22)

        scan = ExhaustiveScanIndex(world, universes).search("g", spacer, mismatches)
        fm = FMIndexSearchIndex(world, universes).search("g", spacer, mismatches)
        self.assertEqual(
            sorted((h.locus_id, h.start, h.duplex.n_mismatch) for h in scan.hits),
            sorted((h.locus_id, h.start, h.duplex.n_mismatch) for h in fm.hits),
        )


class TestDistinctSites(unittest.TestCase):
    """Marking, not filtering -- so the marked set is always a subset."""

    @given(st.lists(st.integers(min_value=0, max_value=400), min_size=1, max_size=25),
           st.integers(min_value=1, max_value=40))
    @SETTINGS
    def test_the_first_guide_is_always_distinct_and_the_set_is_a_subset(
        self, starts, spacing
    ):
        import dataclasses

        from cas13guide.contract.enums import ReasonCode, TargetClass, Verdict
        from cas13guide.metrics.bounds import Interval as Iv
        from cas13guide.models.evidence import (
            Candidate,
            GateResult,
            GuideAssessment,
            MetricVector,
            PairEvidence,
        )
        from cas13guide.pipeline.f_decide import RankedGuide, distinct_site_ids

        ranked = []
        for rank, start in enumerate(starts, 1):
            guide_id = f"g{rank}"
            hit = dataclasses.replace(
                make_hit(guide_id, "L_t1", "t1", TargetClass.REQUIRED), start=start
            )
            assessment = GuideAssessment(
                guide_id=guide_id,
                candidate=Candidate(guide_id=guide_id, spacer="A" * 22,
                                    source_target_id="t1", start=start),
                qc=clean_qc(guide_id),
                metrics=MetricVector(0.9, 0.0, 0.0, 0.0, 0.9, 1.0),
                gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
                evidence=(
                    PairEvidence(
                        guide_id=guide_id, locus_id="L_t1", context="reference", hit=hit,
                        activity=Iv(0.9, 0.9, 0.95), expression=Iv(0.5, 0.5, 0.5),
                        hazard_weight=0.0,
                    ),
                ),
            )
            ranked.append(
                RankedGuide(rank=rank, assessment=assessment, score=1.0,
                            rationale=(), sensitivity={})
            )

        distinct = distinct_site_ids(ranked, spacing)
        all_ids = {entry.assessment.guide_id for entry in ranked}
        self.assertTrue(distinct <= all_ids)
        self.assertIn("g1", distinct)

    @given(st.lists(st.integers(min_value=0, max_value=400), min_size=1, max_size=20))
    @SETTINGS
    def test_zero_spacing_marks_everything_distinct(self, starts):
        """With no spacing rule there are no shifted copies to collapse."""
        import dataclasses

        from cas13guide.contract.enums import ReasonCode, TargetClass, Verdict
        from cas13guide.metrics.bounds import Interval as Iv
        from cas13guide.models.evidence import (
            Candidate,
            GateResult,
            GuideAssessment,
            MetricVector,
            PairEvidence,
        )
        from cas13guide.pipeline.f_decide import RankedGuide, distinct_site_ids

        ranked = []
        for rank, start in enumerate(starts, 1):
            guide_id = f"g{rank}"
            hit = dataclasses.replace(
                make_hit(guide_id, "L_t1", "t1", TargetClass.REQUIRED), start=start
            )
            ranked.append(
                RankedGuide(
                    rank=rank,
                    assessment=GuideAssessment(
                        guide_id=guide_id,
                        candidate=Candidate(guide_id=guide_id, spacer="A" * 22,
                                            source_target_id="t1", start=start),
                        qc=clean_qc(guide_id),
                        metrics=MetricVector(0.9, 0.0, 0.0, 0.0, 0.9, 1.0),
                        gate=GateResult(Verdict.PASS, (ReasonCode.CONSTRAINTS_SATISFIED,)),
                        evidence=(
                            PairEvidence(
                                guide_id=guide_id, locus_id="L_t1", context="reference",
                                hit=hit, activity=Iv(0.9, 0.9, 0.95),
                                expression=Iv(0.5, 0.5, 0.5), hazard_weight=0.0,
                            ),
                        ),
                    ),
                    score=1.0, rationale=(), sensitivity={},
                )
            )
        self.assertEqual(
            distinct_site_ids(ranked, 0),
            {entry.assessment.guide_id for entry in ranked},
        )


class TestInputIsNeverSilentlyAltered(unittest.TestCase):
    """The module's headline rule, checked against generated input.

    Seven passes found four separate ways a sequence could be altered or
    dropped without a word. This states the rule once, for everything.
    """

    @given(RNA)
    @SETTINGS
    def test_an_accepted_sequence_comes_back_unchanged(self, sequence):
        from cas13guide.services.fasta import parse_sequence_input

        [record] = parse_sequence_input(sequence)
        self.assertEqual(record.sequence, sequence)

    @given(st.text(alphabet="ACGUN \t\n", min_size=1, max_size=80))
    @SETTINGS
    def test_only_whitespace_is_ever_removed(self, sequence):
        from cas13guide.services.fasta import SequenceInputError, parse_sequence_input

        expected = "".join(character for character in sequence if not character.isspace())
        try:
            [record] = parse_sequence_input(sequence)
        except SequenceInputError:
            self.assertEqual(expected, "")   # nothing left is the only legal refusal
            return
        self.assertEqual(record.sequence, expected)

    @given(st.text(alphabet="BDHVRYKMSWXZ0123456789", min_size=1, max_size=20))
    @SETTINGS
    def test_anything_outside_the_alphabet_is_refused_rather_than_dropped(self, junk):
        from cas13guide.services.fasta import SequenceInputError, parse_sequence_input

        with self.assertRaises(SequenceInputError):
            parse_sequence_input(junk)

    @given(RNA)
    @SETTINGS
    def test_a_fasta_round_trips_through_its_own_writer(self, sequence):
        from cas13guide.services.fasta import (
            SequenceRecord,
            format_fasta,
            parse_fasta,
        )

        [record] = parse_fasta(format_fasta([SequenceRecord("r1", sequence)]))
        self.assertEqual(record.name, "r1")
        self.assertEqual(record.sequence, sequence)


if __name__ == "__main__":
    unittest.main()
