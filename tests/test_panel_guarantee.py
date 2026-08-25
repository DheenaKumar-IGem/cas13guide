"""If the report prints an approximation bound, the bound has to hold.

`Panel.guarantee` is not a comment. It is computed in select_panel, emitted
into the report JSON by g_output, and read by someone deciding whether the
panel they are about to synthesise is good enough. For most of this file's life
it said:

    greedy >= 1/2 of optimum (matroid-constrained submodular maximization)

and that is false. The 1/2 greedy bound needs the feasible family to be a
matroid. "No two members within min_spacing, and none sharing an off-target
locus" is the independent-set family of a conflict graph, which has no exchange
property: one pan-variant guide in a conserved window is feasible, three
variant-specific guides packed around it are feasible together, and the first
set cannot be extended into the second. Greedy takes the pan guide for the
bigger first-round gain and forfeits all three.

The only test of the string asserted `assertIn("1/2", panel.guarantee)`. It
checked the text and never the property -- the same shape as the provenance
test that asserted a sha256 was 64 hex characters. So these tests brute-force
the true optimum over feasible sets, using the module's own `_compatible` and
`_objective` so "feasible" means what the selector means, and assert that
whatever bound the string claims is actually achieved.
"""

import itertools
import random
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.enums import TargetClass
from cas13guide.contract.spec import TargetSpec
from cas13guide.pipeline.optimizer import (
    _compatible,
    _objective,
    _profile,
    select_panel,
)
from test_optimizer import guide


def brute_force_optimum(pool, spec, size, min_spacing, max_shared_off=0):
    """The best feasible set, by the selector's own definition of feasible."""
    profiles = {a.guide_id: _profile(a) for a in pool}
    weights = {
        t.target_id: max(t.weight, 1.0)
        for t in spec.contract.by_class(TargetClass.REQUIRED)
    }
    best, best_value = (), 0.0
    for k in range(1, size + 1):
        for combo in itertools.combinations(pool, k):
            chosen = []
            for assessment in combo:
                if not _compatible(
                    profiles[assessment.guide_id], chosen, min_spacing, max_shared_off
                ):
                    break
                chosen.append(profiles[assessment.guide_id])
            else:
                value = _objective(chosen, weights)
                if value > best_value:
                    best, best_value = tuple(a.guide_id for a in combo), value
    return best, best_value


def three_required_targets():
    return tuple(
        TargetSpec(
            target_id=f"t{i}", sequence="A" * 300, target_class=TargetClass.REQUIRED
        )
        for i in (1, 2, 3)
    )


class TestTheGuaranteeIsNotOverclaimed(unittest.TestCase):
    def test_the_constrained_branch_claims_no_bound(self):
        """The specific regression: it must not say 1/2, or any ratio."""
        spec = make_spec()
        panel = select_panel([], spec, size=3)
        self.assertNotIn("1/2", panel.guarantee)
        self.assertIn("NO approximation guarantee", panel.guarantee)

    def test_the_cardinality_branch_keeps_its_real_bound(self):
        """1 - 1/e is a genuine theorem and must survive the correction."""
        spec = make_spec()
        panel = select_panel([], spec, size=3, min_spacing=0, max_shared_off=10**9)
        self.assertIn("1 - 1/e", panel.guarantee)

    def test_the_instance_that_refuted_the_old_claim(self):
        """A pan-variant guide blocking its own specialists.

        Not contrived: required targets that are aligned variants share
        coordinates, so a conserved-window guide at 100 and variant-specific
        guides at 105/110/115 conflict at a 22 nt spacer.
        """
        spec = make_spec(targets=three_required_targets())
        pool = [
            guide("gPan", {"t1": 0.40, "t2": 0.40, "t3": 0.40}, start=100),
            guide("gVar1", {"t1": 0.99, "t2": 0.00, "t3": 0.00}, start=105),
            guide("gVar2", {"t1": 0.00, "t2": 0.99, "t3": 0.00}, start=110),
            guide("gVar3", {"t1": 0.00, "t2": 0.00, "t3": 0.99}, start=115),
        ]
        panel = select_panel(pool, spec, size=3)
        _best, optimum = brute_force_optimum(pool, spec, 3, spec.spacer_length)

        self.assertGreater(optimum, 0.0)
        ratio = panel.objective / optimum
        # greedy really is this bad here -- that is the point
        self.assertLess(ratio, 0.5)
        # and the report must not be claiming otherwise
        self.assertNotIn("1/2", panel.guarantee)


class TestAnyClaimedRatioActuallyHolds(unittest.TestCase):
    """The general property, over random instances.

    This is the test that would have caught the original defect without anyone
    having to think of the pan-variant construction: if the string names a
    ratio, achieve it.
    """

    def claimed_ratio(self, guarantee):
        if "1 - 1/e" in guarantee:
            return 0.6321
        if "1/2" in guarantee:
            return 0.5
        return None  # claims no bound, so nothing to check

    def random_pool(self, rng, n=6):
        return [
            guide(
                f"g{i}",
                {f"t{t}": round(rng.random(), 3) for t in (1, 2, 3)},
                start=rng.randrange(0, 200),
            )
            for i in range(n)
        ]

    def test_random_instances_never_violate_a_claimed_ratio(self):
        spec = make_spec(targets=three_required_targets())
        rng = random.Random(20260824)
        checked = 0
        for trial in range(40):
            pool = self.random_pool(rng)
            panel = select_panel(pool, spec, size=3)
            claimed = self.claimed_ratio(panel.guarantee)
            if claimed is None:
                continue
            _best, optimum = brute_force_optimum(pool, spec, 3, spec.spacer_length)
            if optimum <= 0:
                continue
            checked += 1
            with self.subTest(trial=trial):
                self.assertGreaterEqual(
                    panel.objective / optimum,
                    claimed - 1e-9,
                    f"claimed {claimed} but achieved "
                    f"{panel.objective / optimum:.4f}: {panel.guarantee}",
                )
        # the constrained branch claims nothing, so nothing should be checked;
        # if that ever changes, this test starts doing real work automatically
        self.assertEqual(checked, 0, "a ratio is being claimed again -- verify it")

    def test_the_cardinality_branch_meets_one_minus_one_over_e(self):
        """Where a bound IS claimed, hold it to the bound."""
        spec = make_spec(targets=three_required_targets())
        rng = random.Random(11)
        for trial in range(25):
            pool = self.random_pool(rng)
            panel = select_panel(pool, spec, size=3, min_spacing=0, max_shared_off=10**9)
            _best, optimum = brute_force_optimum(pool, spec, 3, 0, 10**9)
            if optimum <= 0:
                continue
            with self.subTest(trial=trial):
                self.assertGreaterEqual(panel.objective / optimum, 0.6321 - 1e-9)
