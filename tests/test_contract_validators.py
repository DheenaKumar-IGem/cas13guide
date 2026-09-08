"""The three refusals `TargetContract` makes, none of which had ever run.

`contract/spec.py` opens by claiming that everything downstream reads this
object and nothing else for intent. These validators are what make that claim
safe, and each protects an assumption something downstream makes without
re-checking it:

* `required_coverage` takes a minimum over the REQUIRED targets. Its own
  `if not required:` guard is unreachable *because* the third validator holds,
  so pinning that validator is what makes the guard defence in depth rather
  than dead code.
* `get` answers with the first target matching an id and `by_class` with every
  match, so two targets sharing an id would be scored twice under one name
  while the other was never scored at all — a confident number about the wrong
  thing, which is the failure mode this package is built against.
* an empty contract has nothing to design against, and every stage downstream
  would read it as a run with no work rather than as a malformed request.

Found by subprocess-traced branch coverage: all three raise-paths were
unexecuted across the whole suite.
"""

import unittest

from pydantic import ValidationError

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import Task, TargetClass
from cas13guide.contract.spec import TargetContract, TargetSpec


def target(target_id: str, target_class: TargetClass) -> TargetSpec:
    return TargetSpec(
        target_id=target_id, sequence=SEQ_A, target_class=target_class,
    )


class TestTheContractRefusesWhatDownstreamCannotHandle(unittest.TestCase):
    def test_a_contract_needs_at_least_one_target(self):
        with self.assertRaises(ValidationError) as caught:
            TargetContract(task=Task.KNOCKDOWN, targets=())
        self.assertIn("at least one target", str(caught.exception))

    def test_two_targets_may_not_share_an_id(self):
        """`get` would answer with the first and `by_class` with both."""
        with self.assertRaises(ValidationError) as caught:
            TargetContract(task=Task.KNOCKDOWN, targets=(
                target("t1", TargetClass.REQUIRED),
                target("t1", TargetClass.DESIRED),
            ))
        self.assertIn("duplicate target_id", str(caught.exception))

    def test_a_contract_needs_something_it_is_required_to_hit(self):
        with self.assertRaises(ValidationError) as caught:
            TargetContract(task=Task.KNOCKDOWN, targets=(
                target("t1", TargetClass.DESIRED),
                target("t2", TargetClass.FORBIDDEN),
            ))
        self.assertIn("at least one REQUIRED", str(caught.exception))

    def test_a_well_formed_contract_is_accepted(self):
        """Or the three refusals above would be satisfied by refusing
        everything, which is the shape of a check that cannot pass."""
        contract = TargetContract(task=Task.KNOCKDOWN, targets=(
            target("t1", TargetClass.REQUIRED),
            target("t2", TargetClass.DESIRED),
        ))
        self.assertEqual(len(contract.targets), 2)
        self.assertEqual(
            [t.target_id for t in contract.by_class(TargetClass.REQUIRED)],
            ["t1"],
        )
        self.assertIsNotNone(contract.get("t2"))
        self.assertIsNone(contract.get("nope"))


class TestTheMetricsGuardIsDefenceInDepthNotDeadCode(unittest.TestCase):
    """`required_coverage` guards against a contract with no REQUIRED target.

    That branch is unreachable through the contract, which is the point: the
    guard exists in case the invariant is ever reached another way. Calling
    the metric directly is how to tell "cannot happen" from "does nothing",
    and the two look identical in a coverage report.
    """

    def test_it_answers_the_vacuous_one_point_zero(self):
        """And that is why the contract must never produce this state.

        `min` over no requirements is "nothing required is unmet", so the
        metric returns 1.0 -- perfect coverage, from no evidence at all. It is
        the same vacuous pass this package has found in four other places:
        `all([])` is True, a maximum over an empty list clears any tolerance,
        an empty held-out set covers everything.

        Here it is unreachable, and the third validator above is the only
        reason. That is the pairing worth pinning: the guard is defence in
        depth, its value is the *unsafe* direction, and what keeps it harmless
        lives in another module.
        """
        from cas13guide.metrics.aggregate import required_coverage

        spec = make_spec()
        self.assertEqual(required_coverage((), _SpecWithNoRequired(spec)), 1.0)

    def test_the_contract_is_what_makes_that_unreachable(self):
        """Not a comment claiming it: the refusal, exercised."""
        with self.assertRaises(ValidationError):
            TargetContract(task=Task.KNOCKDOWN,
                           targets=(target("t1", TargetClass.DESIRED),))

    def test_a_real_spec_never_reaches_it(self):
        spec = make_spec()
        self.assertTrue(spec.contract.by_class(TargetClass.REQUIRED))


class _SpecWithNoRequired:
    """A spec-shaped object whose contract reports no REQUIRED targets.

    Deliberately not a `ResolvedDesignSpec`: the point is that the metric
    holds even if something that is not the contract hands it an empty
    required set, which is the only way that branch can ever be reached.
    """

    def __init__(self, spec):
        self._spec = spec
        self.thresholds = spec.thresholds
        self.profile = spec.profile
        self.context = spec.context
        self.contract = _ContractWithNoRequired(spec.contract)


class _ContractWithNoRequired:
    def __init__(self, contract):
        self._contract = contract
        self.task = contract.task
        self.targets = contract.targets

    def by_class(self, cls):
        return ()

    def get(self, target_id):
        return self._contract.get(target_id)


if __name__ == "__main__":
    unittest.main()
