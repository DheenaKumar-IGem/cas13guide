"""The only check in this repository against something that is not it.

Every other verification here is a *consistency* method: it finds two
descriptions of a run disagreeing. Round eleven's critic named the limit of
that, and `examples/verify_the_maths_against_a_referent.py` was written to
close the arithmetic half of it -- the regularized incomplete beta against
closed forms and the binomial-tail identity, the beta quantile against
closed-form inverses, split-conformal coverage against its own theorem by
simulation, and the Frechet bounds against exhaustive enumeration.

Nothing ran it. It is the most consequential instrument in the repository by
its own account -- the four numbers every guarantee rests on -- and a full
green suite said nothing about any of them, because it lived in `examples/`
and only a person typing its name ever executed it. It takes 0.4 seconds.

This runs it. It deliberately does not reimplement any referent: a second
implementation of a closed form is a second thing to drift, and the whole
point of a referent is that there is exactly one and it is not ours.
"""

import contextlib
import io
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "examples"))

import verify_the_maths_against_a_referent as referent  # noqa: E402


class TestTheMathsReproducesAgainstAnExternalReferent(unittest.TestCase):
    def test_every_number_the_guarantees_rest_on_reproduces(self):
        """Its `main` names each failure, so the message is the diagnosis."""
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = referent.main()
        self.assertEqual(code, 0, chr(10) + captured.getvalue())


class TestTheReferentCheckCannotPassVacuously(unittest.TestCase):
    """Its comparisons are maxima over lists, and max over empty clears any
    tolerance.

    The instrument guards this internally with two derived counts, and the
    guard is worth having in the suite for the same reason the checks are: a
    generator that stopped producing cases would report "largest disagreement
    0.00e+00 over 0 exact cases: ok". That is the vacuous pass the vocabulary
    scraper had, in different arithmetic.
    """

    def test_the_case_generators_still_produce_what_is_expected(self):
        self.assertEqual(
            len(referent.exact_beta_cdf_cases()), referent.EXPECTED_CDF_CASES
        )
        self.assertEqual(
            len(referent.exact_beta_quantile_cases()),
            referent.EXPECTED_QUANTILE_CASES,
        )

    def test_those_counts_are_not_zero(self):
        """Or the check above would be satisfied by generating nothing."""
        self.assertGreater(referent.EXPECTED_CDF_CASES, 0)
        self.assertGreater(referent.EXPECTED_QUANTILE_CASES, 0)


class TestNothingIsComputedForNobody(unittest.TestCase):
    """The dead-value sweep, run rather than described.

    `verify_nothing_is_computed_for_nobody.py` finds values this package
    computes that nothing reads, and exits non-zero naming them. The suite
    referred to it -- one test parses its source to check it can see
    properties -- but nothing executed it, so a field could go dead and every
    run stay green.

    That is not hypothetical either. Making the universe gate read what an
    index holds left `ContextScope.missing_universes`, which compared the
    assay against the *declaration*, with no caller anywhere; eight green
    suites said nothing and a manual run of the instrument found it in
    0.4 seconds.

    A value with no reader is allowed, but only on the record: the instrument
    has an ALLOWED list that carries the reason for each, which is what turns
    "dead" into "deliberate".
    """

    def test_every_value_reaches_something(self):
        import contextlib
        import io

        import verify_nothing_is_computed_for_nobody as sweep

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = sweep.main()
        self.assertEqual(code, 0, chr(10) + captured.getvalue())

    def test_the_sweep_still_has_something_to_look_at(self):
        """Or "everything reaches something" would be true of nothing.

        The same vacuous pass the other instruments guard against: a walker
        that stopped finding declarations would report no dead values because
        it found no values.
        """
        import contextlib
        import io

        import verify_nothing_is_computed_for_nobody as sweep

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            sweep.main()
        text = captured.getvalue()
        self.assertRegex(text, r"[1-9][0-9]* (public )?(field|value|name)")


if __name__ == "__main__":
    unittest.main()
