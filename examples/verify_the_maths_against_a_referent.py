"""Check the numbers against something that is not this package.

Eleven review rounds have been *consistency* work: they find two descriptions of
a run disagreeing. Round eleven's critic named the limit of that plainly --

    This is a consistency method, not a ground-truth method. It is structurally
    unable to find a defect where every surface agrees and all of them are
    wrong: a formula that is internally coherent, documented as exactly what it
    does, tested against its own output, and scientifically incorrect. Nothing
    in this repo compares a number to an external referent.

Part of that gap cannot be closed from inside a repository. Whether the activity
heuristic predicts real Cas13 knockdown is a question for a wet-lab screen, and
this file does not pretend otherwise.

But the *mathematics* is not in that category. A regularized incomplete beta has
a definition independent of any implementation of it; split-conformal prediction
has a finite-sample coverage theorem; Fréchet bounds are the tightest possible
bounds on a joint given its marginals, which for small cases can be found by
brute force. Each of those is an external referent, and each is checkable here.

So this file re-derives, by a different method, the four numbers the package's
guarantees rest on:

1. ``regularized_incomplete_beta`` against closed forms and the binomial-tail
   identity ``I_x(k, n-k+1) = P(Bin(n, x) >= k)``, plus the symmetry relation --
   referents that need no integrator, no tolerance and no shared assumptions;
2. ``beta_quantile`` against closed-form inverse CDFs, because inverting the
   package's own CDF would prove only that the two agree with each other;
3. the split-conformal quantile against its **coverage theorem**, by simulation:
   an interval at alpha must cover the truth about ``1 - alpha`` of the time, and
   the finite-sample correction is what makes that hold at small n. This is the
   one that would catch a formula everything else agrees about;
4. the Fréchet bounds against exhaustive enumeration of joint distributions.

Usage::

    python examples/verify_the_maths_against_a_referent.py

Exits non-zero if any number fails to reproduce.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cas13guide.services.calibration import CalibrationService  # noqa: E402
from cas13guide.services.panel_confidence import (  # noqa: E402
    beta_quantile,
    regularized_incomplete_beta,
)


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def binomial_tail(n: int, k: int, x: float) -> float:
    """``P(Bin(n, x) >= k)``, summed directly from the pmf.

    This is the referent that settled the question. For integer parameters the
    regularized incomplete beta *is* a binomial tail --
    ``I_x(k, n - k + 1) = P(Bin(n, x) >= k)`` -- an identity that needs no
    integrator, no substitution and no tolerance. It is exact, it covers a wide
    grid of ``(a, b, x)``, and it shares nothing with a continued fraction.
    """
    return sum(math.comb(n, i) * x ** i * (1 - x) ** (n - i) for i in range(k, n + 1))


def exact_beta_cdf_cases() -> list[tuple[tuple[float, float, float], float, str]]:
    """Every case here has a closed form, so none of them can be argued with."""
    cases: list[tuple[tuple[float, float, float], float, str]] = []
    for x in (0.1, 0.37, 0.5, 0.9):
        cases.append(((1.0, 1.0, x), x, "uniform"))
        for power in (2, 3, 7):
            cases.append(((float(power), 1.0, x), x ** power, f"x^{power}"))
            cases.append(((1.0, float(power), x), 1 - (1 - x) ** power,
                          f"1-(1-x)^{power}"))
        # Beta(1/2, 1/2) is the arcsine distribution.
        cases.append(((0.5, 0.5, x), (2 / math.pi) * math.asin(math.sqrt(x)),
                      "arcsine"))
    # And the binomial identity over a grid of integer parameters.
    for n in (5, 20, 61):
        for k in (1, n // 3, n // 2, n - 1, n):
            for x in (0.15, 0.5, 0.75):
                cases.append(((float(k), float(n - k + 1), x),
                              binomial_tail(n, k, x), f"Bin({n},{x})>={k}"))
    return cases


def exact_beta_quantile_cases() -> list[tuple[tuple[float, float, float], float, str]]:
    """Families whose inverse CDF is closed-form, so the check is not circular.

    Inverting the package's own CDF to test its own quantile would prove only
    that the two agree with each other, which is the failure mode this whole
    file exists to avoid.
    """
    cases: list[tuple[tuple[float, float, float], float, str]] = []
    for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.99):
        cases.append(((1.0, 1.0, p), p, "uniform"))
        for power in (2.0, 3.0, 8.0):
            cases.append(((power, 1.0, p), p ** (1 / power), f"x^{power:g}"))
            cases.append(((1.0, power, p), 1 - (1 - p) ** (1 / power),
                          f"1-(1-x)^{power:g}"))
        cases.append(((0.5, 0.5, p), math.sin(p * math.pi / 2) ** 2, "arcsine"))
    return cases


def frechet_by_enumeration(marginals: list[float], grid: int = 24) -> tuple[float, float]:
    """Tightest bounds on P(at least one succeeds), found by search.

    For two members the joint is determined by one free parameter -- P(both) --
    constrained by the Fréchet-Hoeffding inequalities. Sweeping it and taking
    the extremes of the union probability is a brute-force referent for the
    closed form the package uses.
    """
    if len(marginals) != 2:
        raise ValueError("enumeration referent is written for two members")
    p, q = marginals
    lo_both, hi_both = max(0.0, p + q - 1.0), min(p, q)
    union = []
    for i in range(grid + 1):
        both = lo_both + (hi_both - lo_both) * i / grid
        union.append(p + q - both)
    return min(union), max(union)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if not ok else ""))
    return ok


def main() -> int:
    failures: list[str] = []
    rng = random.Random(20260902)

    # ---- 1. the incomplete beta, against exact identities ---------------
    print()
    print("regularized_incomplete_beta vs closed forms and the binomial identity")
    cases = exact_beta_cdf_cases()
    worst, worst_at = 0.0, None
    for (a, b, x), want, label in cases:
        got = regularized_incomplete_beta(a, b, x)
        if abs(got - want) > worst:
            worst, worst_at = abs(got - want), (a, b, x, want, got, label)
    if not check(f"largest disagreement {worst:.2e} over {len(cases)} exact cases",
                 worst < 1e-12,
                 f"{worst_at[5]} at a={worst_at[0]} b={worst_at[1]} x={worst_at[2]}: "
                 f"want {worst_at[3]:.12f}, got {worst_at[4]:.12f}"):
        failures.append("incomplete beta")

    # The symmetry relation is a constraint on the function itself, and holds
    # for every (a, b, x) rather than the ones somebody chose.
    worst = 0.0
    for a, b in ((12.5, 7.5), (2.0, 5.0), (0.5, 0.5), (300.5, 0.5), (1.0, 40.0),
                 (0.5, 300.5), (7.0, 7.0)):
        for x in (0.05, 0.25, 0.5, 0.75, 0.95):
            worst = max(worst, abs(
                regularized_incomplete_beta(a, b, x)
                + regularized_incomplete_beta(b, a, 1 - x) - 1.0))
    if not check(f"I_x(a,b) + I_(1-x)(b,a) == 1 to {worst:.1e} over 35 points",
                 worst < 1e-12):
        failures.append("incomplete beta symmetry")

    # ---- 2. the quantile, against closed-form inverses -----------------
    print()
    print("beta_quantile vs closed-form inverse CDFs")
    cases = exact_beta_quantile_cases()
    worst, worst_at = 0.0, None
    for (a, b, p), want, label in cases:
        got = beta_quantile(a, b, p)
        if abs(got - want) > worst:
            worst, worst_at = abs(got - want), (a, b, p, want, got, label)
    if not check(f"largest disagreement {worst:.2e} over {len(cases)} exact cases",
                 worst < 1e-6,
                 f"{worst_at[5]} at a={worst_at[0]} b={worst_at[1]} p={worst_at[2]}: "
                 f"want {worst_at[3]:.9f}, got {worst_at[4]:.9f}"):
        failures.append("beta quantile")

    # ---- 3. conformal coverage, against its theorem --------------------
    print("\nsplit-conformal coverage vs the 1 - alpha guarantee (simulation)")
    for alpha, n_cal in ((0.10, 40), (0.10, 200), (0.20, 25)):
        covered = trials = 0
        for trial in range(400):
            truth = [rng.random() for _ in range(n_cal)]
            # A biased, noisy predictor: the theorem must hold regardless.
            noise = [rng.gauss(0.0, 0.18) for _ in range(n_cal)]
            service = CalibrationService()
            key = CalibrationService.stratum_key("LwaCas13a", "in_vitro_detection", "mm0")
            for t, e in zip(truth, noise):
                service.add_observation(key, min(1.0, max(0.0, t + e)), t,
                                        group=f"T{trial % 12:02d}")
            # One fresh draw from the same distribution.
            new_truth = rng.random()
            new_pred = min(1.0, max(0.0, new_truth + rng.gauss(0.0, 0.18)))
            interval = service.interval(new_pred, key, alpha)
            trials += 1
            if interval.lower <= new_truth <= interval.upper:
                covered += 1
        empirical = covered / trials
        # Split conformal guarantees >= 1 - alpha marginally; the (n+1)
        # correction is what makes that hold at small n. Allow simulation noise
        # of ~3 standard errors, but a systematic shortfall must fail.
        se = math.sqrt(max(empirical * (1 - empirical), 1e-6) / trials)
        ok = empirical >= (1 - alpha) - 3 * se
        if not check(f"alpha={alpha}, n={n_cal}: covered {empirical:.3f} "
                     f"(guarantee >= {1 - alpha:.2f})", ok,
                     f"undercovers by {(1 - alpha) - empirical:.4f}"):
            failures.append(f"conformal coverage alpha={alpha} n={n_cal}")

    # ---- 4. Frechet bounds, against enumeration ------------------------
    print("\nFrechet bounds vs exhaustive enumeration of the joint")
    from cas13guide.services.panel_confidence import _combine_independent

    worst_lo = worst_hi = 0.0
    for _ in range(200):
        p, q = rng.random(), rng.random()
        # The package's floor is the comonotone case (the panel is worth its
        # best member) and its ceiling is the independent noisy-OR.
        floor, ceiling = max(p, q), _combine_independent([p, q])
        enum_lo, enum_hi = frechet_by_enumeration([p, q])
        worst_lo = max(worst_lo, abs(floor - enum_lo))
        worst_hi = max(worst_hi, ceiling - enum_hi)
    if not check(f"comonotone floor matches enumeration (max gap {worst_lo:.2e})",
                 worst_lo < 1e-12):
        failures.append("frechet floor")
    if not check("independent ceiling never exceeds the attainable maximum",
                 worst_hi <= 1e-12, f"exceeds by {worst_hi:.2e}"):
        failures.append("frechet ceiling")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print("   " + f)
        return 1
    print("Every number reproduces against a referent that is not this package.")
    print("This closes the arithmetic half of the gap round eleven's critic")
    print("named. The other half -- whether the activity model predicts real")
    print("knockdown -- needs a wet-lab screen, and no file here can stand in")
    print("for one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
