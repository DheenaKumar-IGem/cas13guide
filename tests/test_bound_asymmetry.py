"""Benefit is never read at the wrong end of its interval.

The asymmetry is the load-bearing idea of this package: a benefit is read at its
*lower* confidence bound and a hazard at its *upper* one, so that a wide
interval can never be mistaken for evidence. The README claimed that
``metrics/bounds.py`` being "the only place intervals are constructed" is what
keeps it -- which was false (they are built in three modules) and describes the
wrong mechanism anyway, since constructing an ``Interval`` elsewhere cannot drop
an asymmetry that lives in how it is *read*.

What actually holds is narrower and checkable: every benefit read goes through
``benefit_lower``, so no site exists at which someone could read a benefit at
the upper end of a wide interval and call it evidence. Hazards read ``.upper``
directly in a handful of places, which is the safe direction -- reading a hazard
high is conservative, and reading it low is the mistake, which is why only the
``.lower`` side needs guarding.

Checked by parsing rather than grepping, because ``.lower`` is also a string
method and ``str.lower()`` outnumbers interval reads in this tree.
"""

import ast
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403

from cas13guide.metrics.bounds import Interval

SRC = Path(__file__).resolve().parents[1] / "src" / "cas13guide"

#: Attribute chains whose ``.lower`` would be a benefit read. ``Interval`` is
#: also used for coverage checks and for this module's own dataclass fields,
#: neither of which is a benefit.
BENEFIT_OWNERS = ("activity", "expression")


def bare_bound_reads(bound: str) -> list[str]:
    """``x.<owner>.<bound>`` read as a value, not called."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Attribute) and node.attr == bound):
                continue
            if id(node) in called:            # str.lower() and friends
                continue
            owner = node.value
            if isinstance(owner, ast.Attribute) and owner.attr in BENEFIT_OWNERS:
                found.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno} "
                             f"{ast.unparse(node)}")
    return sorted(found)


class TestNoBenefitIsReadDirectly(unittest.TestCase):
    def test_nothing_reads_lower_off_an_activity_or_expression(self):
        """The dangerous direction, and the one with no legitimate site.

        Any benefit this package reports goes through ``benefit_lower``. A
        direct ``.lower`` here would not be wrong *today* -- it would be a place
        where the next edit could quietly become wrong.
        """
        self.assertEqual(bare_bound_reads("lower"), [])

    def test_hazards_do_read_upper_directly_and_that_is_the_point(self):
        """Stops the test above from passing because the parse found nothing.

        If this returns empty too, the AST walk is broken rather than the code
        being pure -- these reads exist and are correct.
        """
        upper = bare_bound_reads("upper")
        self.assertGreaterEqual(len(upper), 4, upper)

    def test_benefit_lower_is_actually_used(self):
        """A helper nobody calls would satisfy the first test vacuously."""
        callers = [
            path.relative_to(SRC).as_posix()
            for path in SRC.rglob("*.py")
            if "benefit_lower(" in path.read_text(encoding="utf-8")
            and path.name != "bounds.py"
        ]
        self.assertGreaterEqual(len(callers), 3, callers)

    def test_the_readme_no_longer_claims_the_construction_mechanism(self):
        readme = (SRC.parents[1] / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("the only place intervals are constructed", readme)
        self.assertIn("every benefit read goes through", readme)


class TestAnIntervalContainsItsOwnEstimate(unittest.TestCase):
    """The two halves of an Interval must describe the same quantity.

    ``point`` is the number a report prints; ``lower``/``upper`` are what the
    gate reads. The class refused an inverted interval and said nothing about
    where the point sat, so a guide could have been shown an activity its own
    bound excluded -- two surfaces disagreeing about one quantity, which is
    the failure this repository keeps finding in other forms.

    Nothing was wrong when this was written: 119,762 intervals built across
    the whole suite, including the trained model, kinetics, variant contexts
    and BLAST, already satisfied it. What was missing is anything that would
    say so if it stopped being true.
    """

    def test_a_point_below_its_lower_bound_is_refused(self):
        with self.assertRaises(ValueError):
            Interval(point=0.1, lower=0.2, upper=0.8)

    def test_a_point_above_its_upper_bound_is_refused(self):
        with self.assertRaises(ValueError):
            Interval(point=0.9, lower=0.2, upper=0.8)

    def test_the_bounds_themselves_are_still_valid_points(self):
        """Closed interval: sitting exactly on an edge is not a violation."""
        for point in (0.2, 0.5, 0.8):
            with self.subTest(point=point):
                self.assertEqual(
                    Interval(point=point, lower=0.2, upper=0.8).point, point)

    def test_an_inverted_interval_is_still_refused(self):
        """The check it was added beside, so neither replaces the other."""
        with self.assertRaises(ValueError):
            Interval(point=0.5, lower=0.8, upper=0.2)


class TestTheNonNegativityGuardFiresOnWhatItNames(unittest.TestCase):
    """`hazard_product` requires non-negative factors and did not check for them.

    It tested `iv.upper < 0`, which rejects only an interval lying entirely
    below zero. An interval straddling zero -- negative lower, non-negative
    upper -- passed, and for those inputs multiplying the uppers does not
    bound the product: the extreme is reached at the two *lower* ends when
    both are negative.

    The factors in a real run cannot be negative, so this never produced a
    wrong hazard. It is pinned because the guard is the thing that says so.
    """

    @staticmethod
    def _interval(lower, upper):
        from cas13guide.models.evidence import Interval

        point = min(max(0.0, lower), upper)
        return Interval(point=point, lower=lower, upper=upper)

    def test_an_interval_straddling_zero_is_refused(self):
        from cas13guide.metrics.bounds import hazard_product

        with self.assertRaises(ValueError):
            hazard_product(self._interval(-5.0, 2.0), self._interval(-4.0, 3.0))

    def test_the_value_it_used_to_return_was_not_a_bound(self):
        """The reason the guard matters, stated as arithmetic.

        Had it not raised, multiplying the uppers gives 6.0 while the product
        of the two lower ends is 20.0 -- so the returned figure would have
        been below the maximum it claimed to bound.
        """
        import itertools

        a, b = (-5.0, 2.0), (-4.0, 3.0)
        multiplying_uppers = a[1] * b[1]
        true_max = max(x * y for x, y in itertools.product(a, b))
        self.assertLess(multiplying_uppers, true_max)

    def test_an_entirely_negative_interval_is_still_refused(self):
        """The case the old check did catch, kept."""
        from cas13guide.metrics.bounds import hazard_product

        with self.assertRaises(ValueError):
            hazard_product(self._interval(-5.0, -2.0))

    def test_non_negative_factors_are_unaffected(self):
        """The only shape a real run produces, so nothing moves."""
        from cas13guide.metrics.bounds import hazard_product

        got = hazard_product(self._interval(0.2, 0.6),
                             self._interval(0.3, 0.9), scale=2.0)
        self.assertAlmostEqual(got, 0.6 * 0.9 * 2.0, places=12)

    def test_a_zero_lower_bound_is_allowed(self):
        """`>= 0` is the precondition, not `> 0` -- an unknown factor reads
        [0, 1] and must stay usable."""
        from cas13guide.metrics.bounds import hazard_product

        self.assertAlmostEqual(hazard_product(self._interval(0.0, 1.0)), 1.0)


if __name__ == "__main__":
    unittest.main()
