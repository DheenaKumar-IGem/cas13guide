"""Bayesian panel confidence (FR6.3).

What this answers, and what the number next to it does not
----------------------------------------------------------
``Panel.coverage`` is a noisy-OR of per-guide conformal lower bounds:
``1 - prod(1 - lcb)``. It reads like "the probability the panel detects the
target" and it is not one, for two separate reasons, and this module addresses
them separately because they are separate mistakes.

**One: a calibrated activity bound is not a success probability.** ``lcb`` is a
bound on a *continuous* predicted activity. Whether a guide works is a
*binary* event -- observed activity clears the assay's threshold, or it does
not. Turning the first into the second is an inference, and it needs data
about how often predictions at that level actually come true. That data
already exists in this package: the calibration strata hold ``(predicted,
observed)`` pairs. So per guide we take the calibration rows whose prediction
sits near this guide's, count how many cleared the threshold, and update a
Jeffreys ``Beta(0.5, 0.5)`` prior with that binomial likelihood. The result is
a posterior on "a guide predicted this well succeeds", with a credible
interval that is wide exactly when the calibration set is thin.

That is the Bayesian half, and it is deliberately built on counts rather than
on the conformal interval. An earlier attempt at FR6.3 derived its scale as
``q / z`` -- a distribution-free conformal quantile divided by a Gaussian one
-- which imports the normality assumption conformal exists to avoid. Nothing
here reads a conformal quantile.

**Two: panel members do not fail independently.** They share a transcript, a
prep, and one activity model whose errors are correlated; the spacing
constraint only stops them sharing a *site*. Noisy-OR assumes independence, so
it is not a probability but a *ceiling* -- and specifically a ceiling under
positive dependence, which is the only kind on offer here. The matching floor
is the comonotone case, where the members fail together as far as their
marginals allow, and the panel is worth exactly its best member:

    max_g p_g   <=   P(panel detects t)   <=   1 - prod_g (1 - p_g)

Both ends are Frechet-Hoeffding bounds, they need no parameter nobody
measured, and the gap between them is the honest width of the claim. (Under
*arbitrary* dependence the upper bound is the looser ``min(1, sum p_g)``;
positive dependence is what buys the noisy-OR back as a ceiling, and that
assumption is stated rather than hidden.) This package reads benefit at its
lower bound everywhere else, so the headline number here is the floor: a panel
is credited with what its best member can do, and the extra members have to be
*shown* to be independent before they count.

A point estimate, only if you measured one
------------------------------------------
Collapsing that interval to a single number takes a dependence parameter --
the common-cause fraction ``beta`` of the standard beta-factor reliability
model (IEC 61508-6), where each member's failure probability ``q_g`` splits
into a shared part ``beta * q_g`` and an independent part ``(1 - beta) q_g``:

    Q_panel = beta * mean(q_g) + prod_g ((1 - beta) * q_g)

At ``beta = 0`` that is the noisy-OR; at ``beta = 1`` the panel is one guide
wearing three hats. ``beta`` cannot be estimated from the calibration set this
package holds: it is a property of replicate *panel* experiments, and nothing
here has any. So **no default is shipped**. Supply one measured for your assay
via :meth:`PanelDependence.from_json` and :meth:`absolute_confidence` unlocks;
without one it raises, and the sound interval above is what you get. That is
the same bargain :mod:`cas13guide.services.kinetics` strikes over ``k_cat``,
for the same reason: a plausible number in the place where a measured one goes
is worse than a hole, because it does not look like one.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..contract.enums import TargetClass
from ..contract.spec import ResolvedDesignSpec
from ..metrics.bounds import benefit_lower

#: Calibration rows within this much of a guide's prediction count as evidence
#: about it. Wide enough that a stratum of a few hundred rows leaves a usable
#: window, narrow enough that the posterior still depends on the prediction.
DEFAULT_BANDWIDTH = 0.10

#: Below this many rows in the window the posterior is reported but flagged
#: thin: a handful of trials cannot separate 0.6 from 0.9.
MIN_WINDOW_N = 20


class UnmeasuredDependenceError(RuntimeError):
    """A single panel-detection probability was asked for without a measured
    common-cause fraction to compute it with."""


@dataclass(frozen=True, slots=True)
class PanelDependence:
    """How much of a panel's failure is common-cause (DR4).

    ``common_cause_beta`` is the fraction of each member's failure probability
    that is shared with every other member -- bad RNA, a mistaken structure
    call, a model bias the whole panel inherits. It is a property of your
    assay, measured from replicate panel runs, and this package ships no value
    for it.
    """

    common_cause_beta: float
    #: Where it came from. Anything but a real measurement should stay
    #: provisional, and provisional values are refused the same as absent ones.
    source: str = "PLACEHOLDER -- not measured"
    provisional: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.common_cause_beta <= 1.0:
            raise ValueError(
                "common_cause_beta is a fraction of failure that is shared; "
                f"got {self.common_cause_beta}, which is outside [0, 1]"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "PanelDependence":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {"common_cause_beta", "source", "provisional"}
        if unknown:
            raise ValueError(
                f"unknown key(s) in {path}: {', '.join(sorted(unknown))}. "
                "Expected common_cause_beta, source, provisional."
            )
        if "common_cause_beta" not in data:
            raise ValueError(f"{path} has no 'common_cause_beta'")
        return cls(**data)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @property
    def usable(self) -> bool:
        return not self.provisional

    @property
    def version(self) -> str:
        tag = "provisional" if self.provisional else "measured"
        return f"panel-dependence/beta-factor/{tag}/v1"


# ---------------------------------------------------------------------------
# Beta distribution, to the precision a credible interval needs.
#
# scipy would supply these, and scipy is not a dependency of this package for
# one HTML report and one posterior. The continued fraction below is the
# standard Lentz evaluation of the regularized incomplete beta; the quantile
# is a bisection on it, which is slow per call and irrelevant at panel sizes.
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    max_iterations, eps, tiny = 300, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iterations + 1):
        m2 = 2 * m
        num = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        num = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + num * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + num / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` -- the Beta(a, b) CDF at ``x``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def beta_quantile(a: float, b: float, p: float) -> float:
    """Inverse Beta CDF by bisection. Monotone, so bisection is exact enough."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if regularized_incomplete_beta(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Per-guide success posterior
# ---------------------------------------------------------------------------

#: Jeffreys prior for a binomial rate. Chosen over Beta(1, 1) because a
#: uniform prior on a rate is not uninformative under reparameterisation, and
#: over Beta(0, 0) because that is improper and returns nothing usable when a
#: window happens to hold no successes.
PRIOR_A = 0.5
PRIOR_B = 0.5


@dataclass(frozen=True, slots=True)
class GuideSuccess:
    """Posterior on "a guide predicted this well clears the threshold"."""

    guide_id: str
    target_id: str
    #: What the activity model said, read at its lower bound like every other
    #: benefit in this package.
    prediction: float
    #: Calibration rows near ``prediction``, and how many cleared threshold.
    trials: int
    successes: int
    posterior_a: float
    posterior_b: float
    credible_lower: float
    credible_upper: float
    #: Which calibration stratum the window came from, or why there was none.
    stratum: str = "none"
    #: True when the window held too few rows to say much, or none at all. The
    #: posterior is still returned; it is simply the prior plus a shrug.
    thin: bool = True

    @property
    def mean(self) -> float:
        return self.posterior_a / (self.posterior_a + self.posterior_b)

    def to_dict(self) -> dict[str, Any]:
        return {
            "guide_id": self.guide_id,
            "target_id": self.target_id,
            "prediction": round(self.prediction, 4),
            "trials": self.trials,
            "successes": self.successes,
            "success_mean": round(self.mean, 4),
            "credible_lower": round(self.credible_lower, 4),
            "credible_upper": round(self.credible_upper, 4),
            "stratum": self.stratum,
            "thin": self.thin,
        }


@dataclass(frozen=True, slots=True)
class TargetConfidence:
    """What can be said about one REQUIRED target, and how wide it is."""

    target_id: str
    members: tuple[GuideSuccess, ...]
    #: Comonotone floor: the panel is worth its best member. This is the
    #: number to quote, because it is the one that does not assume anything
    #: about the members failing separately.
    lower: float
    #: Independent ceiling: the noisy-OR. Reachable only if the members really
    #: are independent, which they are not.
    upper: float
    #: The floor and ceiling recomputed at the members' marginal credible
    #: bounds, so posterior uncertainty is visible alongside dependence
    #: uncertainty. A monotone envelope over marginals, not a joint credible
    #: set -- detection is increasing in every ``p_g``, so the endpoints are
    #: bounds, but their joint level is not 1 - alpha.
    credible_lower: float
    credible_upper: float
    #: Only present when a *measured* common-cause fraction was supplied.
    point: float | None = None
    #: True when any member's posterior rests on a thin or absent window.
    thin: bool = False

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "credible_lower": round(self.credible_lower, 4),
            "credible_upper": round(self.credible_upper, 4),
            "point": None if self.point is None else round(self.point, 4),
            "thin": self.thin,
            "members": [m.to_dict() for m in self.members],
        }


@dataclass(frozen=True, slots=True)
class PanelConfidence:
    """FR6.3 for one panel: a bounded detection probability per target."""

    version: str
    targets: tuple[TargetConfidence, ...] = ()
    #: The dependence model used for ``point``, or ``None`` when none was
    #: supplied and only the interval is available.
    dependence_version: str | None = None
    #: Plain-language statements of what the numbers assume. These reach the
    #: report; a caveat only the source states is a caveat nobody reads.
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_point_estimates(self) -> bool:
        return any(t.point is not None for t in self.targets)

    def by_target(self) -> dict[str, TargetConfidence]:
        return {t.target_id: t for t in self.targets}

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dependence": self.dependence_version,
            "targets": [t.to_dict() for t in self.targets],
            "notes": list(self.notes),
        }


def _combine_independent(probabilities: list[float]) -> float:
    miss = 1.0
    for p in probabilities:
        miss *= 1.0 - p
    return 1.0 - miss


def _combine_beta_factor(probabilities: list[float], beta: float) -> float:
    """Beta-factor common-cause combination, clamped into the sound interval.

    The clamp is not cosmetic, and it binds in the opposite regime to the one
    an earlier version of this comment named. The textbook form adds a
    common-cause term to a product of thinned independent terms; as members
    become nearly certain to *succeed* (``q -> 0``) the common-cause term is
    ``O(q)`` and the thinned product is ``O(q**n)``, so the raw value stays
    below the ceiling and the clamp never fires. It fires when members are
    nearly certain to **fail**: for two equal members the condition reduces to
    ``q > 1 / (2 - beta)``, i.e. roughly ``p < 0.5`` -- which is routine with a
    thin calibration window, where the Jeffreys posterior sits near 0.5. And
    the excursion is not a hair: at ``p = 0.10`` it leaves the interval by tens
    of percent of the ceiling. A point estimate outside its own bounds is a bug
    wherever it is read, so it is corrected here rather than downstream.
    """
    failures = [1.0 - p for p in probabilities]
    mean_failure = sum(failures) / len(failures)
    independent = 1.0
    for q in failures:
        independent *= (1.0 - beta) * q
    detection = 1.0 - (beta * mean_failure + independent)
    floor, ceiling = max(probabilities), _combine_independent(probabilities)
    return min(max(detection, floor), ceiling)


class BayesianPanelConfidence:
    """FR6.3 -- the Bayesian panel confidence model.

    Constructed per run because it reads the run's own calibration service:
    the posterior is only as good as the screen behind it, and a service built
    on synthetic calibration produces a posterior that is exercisable and
    nothing more. The run's limitations say so.
    """

    version = "panel-confidence/beta-binomial-common-cause/v1"

    def __init__(
        self,
        calibration: Any,
        spec: ResolvedDesignSpec,
        dependence: PanelDependence | None = None,
        bandwidth: float = DEFAULT_BANDWIDTH,
        min_window: int = MIN_WINDOW_N,
    ) -> None:
        self.calibration = calibration
        self.spec = spec
        self.dependence = dependence
        self.bandwidth = bandwidth
        self.min_window = min_window
        #: Observed activity at or above this counts as the guide having
        #: worked. The gate already uses it as the bar a REQUIRED target must
        #: clear, so success means the same thing in both places.
        self.threshold = float(spec.thresholds.required_coverage_min)
        self.alpha = float(spec.thresholds.alpha)

    # -- per guide ------------------------------------------------------
    def guide_success(
        self, guide_id: str, target_id: str, prediction: float, n_mismatch: int = 0
    ) -> GuideSuccess:
        """Beta-Binomial posterior for one guide against one target."""
        from .calibration import CalibrationService

        key = CalibrationService.stratum_key(
            self.spec.profile.effector.value,
            self.spec.profile.assay.value,
            CalibrationService.mismatch_regime(n_mismatch),
        )
        stratum = None
        resolve = getattr(self.calibration, "resolve", None)
        if resolve is not None:
            stratum = resolve(key)

        trials = successes = 0
        label = f"{key}->none"
        if stratum is not None:
            label = stratum.key
            pairs = [
                (p, o)
                for p, o in zip(stratum.predicted, stratum.observed)
                if abs(p - prediction) <= self.bandwidth
            ]
            if len(pairs) < self.min_window:
                # Too few neighbours to condition on the prediction level. Fall
                # back to the whole stratum rather than to nothing: an
                # unconditional success rate is weaker evidence, not absent
                # evidence, and the ``thin`` flag says which one this is.
                pairs = list(zip(stratum.predicted, stratum.observed))
                label = f"{stratum.key}->unconditional"
            trials = len(pairs)
            successes = sum(1 for _, o in pairs if o >= self.threshold)

        a = PRIOR_A + successes
        b = PRIOR_B + (trials - successes)
        return GuideSuccess(
            guide_id=guide_id,
            target_id=target_id,
            prediction=prediction,
            trials=trials,
            successes=successes,
            posterior_a=a,
            posterior_b=b,
            credible_lower=beta_quantile(a, b, self.alpha / 2.0),
            credible_upper=beta_quantile(a, b, 1.0 - self.alpha / 2.0),
            stratum=label,
            thin=trials < self.min_window or label.endswith("unconditional"),
        )

    # -- per panel ------------------------------------------------------
    def for_panel(self, panel: Any) -> PanelConfidence:
        """Bounded detection probability for every REQUIRED target.

        A target no panel member reaches gets no entry rather than a zero: the
        panel does not cover it at all, which ``Panel.coverage`` and the
        feasibility check already report, and a confidence of 0.0 alongside
        them reads as a computed result rather than an absence.
        """
        required = [t.target_id for t in self.spec.contract.by_class(TargetClass.REQUIRED)]
        per_target: dict[str, list[GuideSuccess]] = {t: [] for t in required}

        for assessment in panel.guides:
            best: dict[str, tuple[float, int]] = {}
            for evidence in assessment.evidence:
                target_id = evidence.hit.target_id
                if evidence.target_class is not TargetClass.REQUIRED or not target_id:
                    continue
                lower = benefit_lower(evidence.activity)
                if target_id not in best or lower > best[target_id][0]:
                    best[target_id] = (lower, evidence.hit.duplex.n_mismatch)
            for target_id, (lower, n_mismatch) in best.items():
                if target_id not in per_target:
                    continue
                per_target[target_id].append(
                    self.guide_success(assessment.guide_id, target_id, lower, n_mismatch)
                )

        targets: list[TargetConfidence] = []
        for target_id in required:
            members = per_target[target_id]
            if not members:
                continue
            means = [m.mean for m in members]
            lows = [m.credible_lower for m in members]
            highs = [m.credible_upper for m in members]
            point = None
            if self.dependence is not None and self.dependence.usable:
                point = _combine_beta_factor(means, self.dependence.common_cause_beta)
            targets.append(
                TargetConfidence(
                    target_id=target_id,
                    members=tuple(members),
                    lower=max(means),
                    upper=_combine_independent(means),
                    credible_lower=max(lows),
                    credible_upper=_combine_independent(highs),
                    point=point,
                    thin=any(m.thin for m in members),
                )
            )

        return PanelConfidence(
            version=self.version,
            targets=tuple(targets),
            dependence_version=(
                self.dependence.version
                if self.dependence is not None and self.dependence.usable
                else None
            ),
            notes=self._notes(targets),
        )

    def _notes(self, targets: list[TargetConfidence]) -> tuple[str, ...]:
        # Chosen by the same predicate that gates the third note, because the
        # first two were unconditional and therefore false in the one
        # configuration this module exists to unlock: supply a measured beta
        # and the report drew a point marker inside the band while the text
        # under it said "reported as an interval, not a number" and "nothing
        # here can tell you where between them your assay sits". The measured
        # beta is exactly what located it. `not targets` stays on the
        # interval-only branch: any([]) is False, and a run covering no target
        # has no point estimate to describe.
        located = bool(targets) and all(t.point is not None for t in targets)
        if located:
            notes = [
                "Detection probability is a point estimate located inside the "
                "[fail together, fail independently] interval by the measured "
                "common-cause fraction you supplied. The interval is still "
                "reported, and is still what the evidence bounds on its own.",
                "The point is only as good as that fraction. It came from your "
                "replicate panel runs, not from anything this package measured.",
            ]
        else:
            notes = [
                "Detection probability is reported as an interval, not a number. "
                "The lower end assumes the panel's guides fail together as far as "
                "their individual rates allow, the upper end assumes they fail "
                "independently. Nothing here can tell you where between them your "
                "assay sits.",
                "Read the lower end. It is what the panel is worth if its members "
                "share a failure mode, which guides on one transcript scored by one "
                "model generally do.",
            ]
        if targets and any(t.point is None for t in targets):
            notes.append(
                "No measured common-cause fraction was supplied, so no single "
                "detection probability is offered. Measure one from replicate "
                "panel runs and pass it with --panel-dependence."
            )
        if any(t.thin for t in targets):
            notes.append(
                "At least one member's success posterior rests on a thin or "
                "unconditional calibration window, so its credible interval is "
                "wide by construction rather than by chance."
            )
        return tuple(notes)

    # -- the refusal ----------------------------------------------------
    def absolute_confidence(self, panel: Any) -> dict[str, float]:
        """A single detection probability per target, or a refusal.

        Mirrors ``kinetics.absolute_signal``: the machinery is here and works,
        and it will not manufacture the one input nobody measured.
        """
        if self.dependence is None:
            raise UnmeasuredDependenceError(
                "A single panel detection probability needs a measured "
                "common-cause fraction, and none was supplied. Without it the "
                "answer is the interval from for_panel(): [best member, "
                "noisy-OR]. Measure beta from replicate panel runs and pass it "
                "with --panel-dependence FILE.json."
            )
        if not self.dependence.usable:
            raise UnmeasuredDependenceError(
                "The supplied common-cause fraction is flagged provisional "
                f"({self.dependence.source!r}). A guessed dependence parameter "
                "produces a confident-looking number whose confidence is the "
                "guess. Set provisional=false only when the value is measured."
            )
        confidence = self.for_panel(panel)
        return {t.target_id: t.point for t in confidence.targets if t.point is not None}
