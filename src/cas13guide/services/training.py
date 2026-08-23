"""Fitting an activity model, and doing the splits honestly.

The reference :class:`~cas13guide.services.registry.HeuristicCas13Model` exists
so the pipeline runs; this module is how you replace it with something learned
from a tiling screen.

The methodological point
------------------------
Guide-design datasets are tiling screens: overlapping windows walked along a
handful of transcripts. Two windows one nucleotide apart share 21 of 22 bases
and sit in the same local structural and expression context, so rows within a
transcript are correlated beyond their sequence.

Every split here is grouped by transcript. The reason is the *estimand*, not a
folk warning about leakage: grouped CV answers "how well will this model do on
a transcript it has never seen", which is the deployment condition when you
design guides against a new gene. Ungrouped CV answers "how well will it do on
another window of a transcript it already trained on" -- a question nobody
designing guides is asking.

That distinction matters because the size and even the *sign* of the ungrouped
bias depends on the data regime. Measured on simulated screens with a
per-transcript offset (``simulate.py``, 156 features, ridge-logistic):

==========  ==========  =========  =========
rows        grouped     ungrouped  gap
==========  ==========  =========  =========
156         0.540       0.703      **+0.163**
276         0.812       0.871      +0.059
594         0.727       0.734      +0.006
1432        0.743       0.697      -0.046
2628        0.828       0.782      -0.046
==========  ==========  =========  =========

Two effects compete:

* **Leakage** inflates ungrouped scores. The model partly memorizes a
  transcript's offset and is then rewarded for it on held-out windows of the
  same transcript. It dominates when the model has capacity to spare --
  roughly ``n <= 2p`` for this linear model, and at far larger ``n`` for the
  high-capacity models real screens are fit with.
* **Test-set composition** inflates *grouped* scores. A grouped test fold
  holds two or three whole transcripts, so its observations carry less
  unpredictable between-transcript offset spread than a random fold drawn from
  all of them, and within-fold Spearman rises.

With the transcript offset removed entirely the gap is exactly ``0.000``,
which pins the mechanism: neither effect comes from sequence overlap on its
own. So "grouped is always the more conservative number" is false, and this
module does not claim it. Grouped is the number that answers the right
question. :func:`compare_splitting` runs the probe on your own data so the
regime can be measured rather than assumed.

The three-way split
-------------------
Split conformal needs a calibration set disjoint from training, and coverage
has to be checked on data used for neither. So: **train** fits the weights,
**calibrate** fits the conformal quantiles, **test** measures skill and
coverage. All three are grouped by transcript.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..contract.enums import Effector
from ..models.features import FeatureSpec
from ..util.duplex import DuplexProfile, profile_duplex
from .benchmark import BenchmarkCase, BenchmarkReport, run_benchmark, spearman
from .calibration import CalibrationService
from .registry import ModelDomain, ModelRegistry, NullAccessibilityModel


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One measured guide/target observation from a screen."""

    spacer: str
    target_site: str
    observed_activity: float
    #: The leakage unit. Overlapping windows from one transcript share this,
    #: and must never straddle a split boundary.
    group: str
    effector: str
    assay: str
    accessibility: float | None = None

    def duplex(self) -> DuplexProfile:
        return profile_duplex(self.spacer, self.target_site)

    def stratum(self) -> str:
        """The calibration stratum this observation belongs to."""
        return CalibrationService.stratum_key(
            self.effector,
            self.assay,
            CalibrationService.mismatch_regime(self.duplex().n_mismatch),
        )


# ---------------------------------------------------------------------------
# Ridge logistic regression (IRLS)
# ---------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def fit_ridge_logistic(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Fit ``P(active) = sigmoid(Xw + b)`` with an L2 penalty, by IRLS.

    ``y`` may be continuous in ``[0, 1]`` -- this is a fractional-response
    model, and the cross-entropy loss is well defined for fractional targets.
    Newton steps converge in a handful of iterations, so no learning rate has
    to be tuned and the result is deterministic.

    The intercept is not penalized.
    """
    n, p = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(p + 1)

    penalty = np.eye(p + 1) * (2.0 * alpha)
    penalty[p, p] = 0.0  # never shrink the intercept

    for _ in range(max_iter):
        eta = Xa @ w
        mu = _sigmoid(eta)
        weights = np.clip(mu * (1.0 - mu), 1e-6, None)
        z = eta + (y - mu) / weights

        WX = Xa * weights[:, None]
        hessian = Xa.T @ WX + penalty
        gradient = Xa.T @ (weights * z)
        try:
            w_new = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            w_new = np.linalg.lstsq(hessian, gradient, rcond=None)[0]

        if np.max(np.abs(w_new - w)) < tol:
            w = w_new
            break
        w = w_new

    return w[:p], float(w[p])


# ---------------------------------------------------------------------------
# The trained model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TrainedActivityModel:
    """An ``ActivityModel`` fit from data, with a domain read off that data."""

    spec: FeatureSpec
    weights: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    domain: ModelDomain
    version: str = "model/ridge-logistic/v1"

    def predict(
        self, spacer: str, duplex: DuplexProfile, accessibility: float | None
    ) -> float:
        x = self.spec.transform(spacer, duplex, accessibility)
        z = (x - self.mean) / self.scale
        return float(_sigmoid(np.array([z @ self.weights + self.intercept]))[0])

    def top_features(self, k: int = 10) -> list[tuple[str, float]]:
        """Largest-magnitude standardized coefficients, for inspection.

        Worth reading after any refit: a positional one-hot feature dominating
        the list usually means the model found the transcript, not the biology.
        """
        order = np.argsort(-np.abs(self.weights))[:k]
        return [(self.spec.names[i], float(self.weights[i])) for i in order]


def _design_matrix(examples: list[TrainingExample], spec: FeatureSpec) -> np.ndarray:
    return spec.transform_many(
        (e.spacer, e.duplex(), e.accessibility) for e in examples
    )


def _standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0  # constant columns pass through unchanged
    return mean, scale


def _derive_domain(examples: list[TrainingExample]) -> ModelDomain:
    """A model's domain is exactly what it was shown -- nothing wider."""
    effectors = set()
    for e in examples:
        try:
            effectors.add(Effector(e.effector))
        except ValueError:
            continue
    return ModelDomain(
        effectors=frozenset(effectors),
        spacer_lengths=frozenset(len(e.spacer) for e in examples),
        max_mismatches=max((e.duplex().n_mismatch for e in examples), default=0),
    )


def fit_model(examples: list[TrainingExample], alpha: float = 1.0) -> TrainedActivityModel:
    if not examples:
        raise ValueError("cannot fit a model with no examples")
    lengths = {len(e.spacer) for e in examples}
    if len(lengths) != 1:
        raise ValueError(f"all examples must share a spacer length; saw {sorted(lengths)}")

    spec = FeatureSpec.for_length(lengths.pop())
    X = _design_matrix(examples, spec)
    y = np.array([e.observed_activity for e in examples], dtype=np.float64)
    mean, scale = _standardizer(X)
    weights, intercept = fit_ridge_logistic((X - mean) / scale, y, alpha=alpha)

    return TrainedActivityModel(
        spec=spec,
        weights=weights,
        intercept=intercept,
        mean=mean,
        scale=scale,
        domain=_derive_domain(examples),
    )


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def group_kfold(groups: list[str], k: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """Deterministic k-fold where a group never straddles a fold boundary."""
    unique = sorted(set(groups))
    if k > len(unique):
        raise ValueError(f"cannot make {k} folds from {len(unique)} groups")

    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(unique)))
    assignment = {unique[g]: i % k for i, g in enumerate(order)}

    folds: list[tuple[list[int], list[int]]] = []
    for fold in range(k):
        test = [i for i, g in enumerate(groups) if assignment[g] == fold]
        train = [i for i, g in enumerate(groups) if assignment[g] != fold]
        folds.append((train, test))
    return folds


def _random_kfold(n: int, k: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    """Ungrouped k-fold. Present only so leakage can be measured."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    folds = []
    for fold in range(k):
        test = sorted(int(i) for i in order[fold::k])
        train = sorted(set(range(n)) - set(test))
        folds.append((train, test))
    return folds


@dataclass(frozen=True, slots=True)
class CVReport:
    grouped: bool
    folds: int
    spearman_mean: float
    spearman_std: float
    mae_mean: float
    per_fold: tuple[float, ...] = field(default_factory=tuple)
    #: Folds asked for, as opposed to :attr:`folds`, which counts the ones that
    #: actually ran. They differ when a fold had an empty train or test side --
    #: which happens for real, e.g. a three-group screen leaves the training
    #: split with a single group and every fold is skipped.
    requested_folds: int = 0

    @property
    def measured(self) -> bool:
        """False when no fold ran, so the numbers below mean nothing."""
        return self.folds > 0

    def summary(self) -> str:
        kind = "grouped" if self.grouped else "ungrouped"
        if not self.measured:
            # 0.000 formatted like a measurement is worse than saying nothing:
            # it reads as "no skill" rather than "not measured".
            return (
                f"{kind} {self.requested_folds}-fold: NOT MEASURED "
                "(every fold had an empty train or test split -- too few groups)"
            )
        return (
            f"{kind} {self.folds}-fold: spearman "
            f"{self.spearman_mean:.3f} +/- {self.spearman_std:.3f}, "
            f"mae {self.mae_mean:.3f}"
        )


def cross_validate(
    examples: list[TrainingExample],
    alpha: float = 1.0,
    k: int = 5,
    seed: int = 0,
    grouped: bool = True,
) -> CVReport:
    """Cross-validated skill.

    ``grouped=False`` reproduces the standard mistake, and is here so the
    inflation it produces can be measured instead of asserted.
    """
    groups = [e.group for e in examples]
    folds = (
        group_kfold(groups, k, seed) if grouped else _random_kfold(len(examples), k, seed)
    )

    scores: list[float] = []
    maes: list[float] = []
    for train_idx, test_idx in folds:
        if not train_idx or not test_idx:
            continue
        model = fit_model([examples[i] for i in train_idx], alpha=alpha)
        preds, obs = [], []
        for i in test_idx:
            e = examples[i]
            preds.append(model.predict(e.spacer, e.duplex(), e.accessibility))
            obs.append(e.observed_activity)
        scores.append(spearman(preds, obs))
        maes.append(sum(abs(p - o) for p, o in zip(preds, obs)) / len(preds))

    return CVReport(
        grouped=grouped,
        folds=len(scores),
        spearman_mean=float(np.mean(scores)) if scores else 0.0,
        spearman_std=float(np.std(scores)) if scores else 0.0,
        mae_mean=float(np.mean(maes)) if maes else 0.0,
        per_fold=tuple(scores),
        requested_folds=k,
    )


@dataclass(frozen=True, slots=True)
class SplittingComparison:
    """Grouped vs ungrouped CV on the same data, and the gap between them."""

    grouped: CVReport
    ungrouped: CVReport

    @property
    def gap(self) -> float:
        """Ungrouped minus grouped. Positive means ungrouped looks better."""
        return self.ungrouped.spearman_mean - self.grouped.spearman_mean

    @property
    def interpretation(self) -> str:
        if self.gap > 0.02:
            return (
                f"ungrouped CV is inflated by {self.gap:+.3f} spearman -- the model has "
                "capacity to memorize transcript-level effects; trust the grouped number"
            )
        if self.gap < -0.02:
            return (
                f"ungrouped CV is {self.gap:+.3f} lower, a test-set composition effect "
                "rather than leakage; the grouped number is still the one that answers "
                "the deployment question"
            )
        return (
            f"gap {self.gap:+.3f} is negligible in this regime; grouped remains correct "
            "on principle"
        )

    def summary(self) -> str:
        return f"{self.grouped.summary()}\n{self.ungrouped.summary()}\n{self.interpretation}"


def compare_splitting(
    examples: list[TrainingExample], alpha: float = 1.0, k: int = 5, seed: int = 0
) -> SplittingComparison:
    """Measure the ungrouped-split bias on this dataset instead of assuming it."""
    return SplittingComparison(
        grouped=cross_validate(examples, alpha=alpha, k=k, seed=seed, grouped=True),
        ungrouped=cross_validate(examples, alpha=alpha, k=k, seed=seed, grouped=False),
    )


# ---------------------------------------------------------------------------
# Train -> calibrate -> evaluate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: TrainedActivityModel
    calibration: CalibrationService
    benchmark: BenchmarkReport
    cv: CVReport
    split_sizes: dict[str, int]
    split_groups: dict[str, tuple[str, ...]]

    def summary(self) -> str:
        return (
            f"{self.cv.summary()} | held-out {self.benchmark.summary()} | "
            f"split {self.split_sizes}"
        )


def three_way_group_split(
    groups: list[str],
    seed: int = 0,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[list[int], list[int], list[int]]:
    """Grouped train / calibrate / test partition.

    Conformal calibration must not see training data, and coverage must be
    measured on data used for neither -- so this is a three-way split, all of
    it grouped.
    """
    unique = sorted(set(groups))
    if len(unique) < 3:
        raise ValueError("need at least three groups for a three-way split")

    rng = np.random.default_rng(seed)
    shuffled = [unique[i] for i in rng.permutation(len(unique))]

    n = len(shuffled)
    n_train = max(1, int(round(fractions[0] * n)))
    n_cal = max(1, int(round(fractions[1] * n)))
    if n_train + n_cal >= n:
        n_train = max(1, n - 2)
        n_cal = 1

    assign = {}
    for g in shuffled[:n_train]:
        assign[g] = 0
    for g in shuffled[n_train : n_train + n_cal]:
        assign[g] = 1
    for g in shuffled[n_train + n_cal :]:
        assign[g] = 2

    buckets: tuple[list[int], list[int], list[int]] = ([], [], [])
    for i, g in enumerate(groups):
        buckets[assign[g]].append(i)
    return buckets


def train_and_calibrate(
    examples: list[TrainingExample],
    alpha: float = 1.0,
    seed: int = 0,
    cv_folds: int = 5,
    calibration_alpha: float = 0.10,
) -> TrainingResult:
    """The full honest protocol: grouped split, fit, calibrate, evaluate.

    Returns everything the regression gate needs, so a refit can be compared
    against a stored baseline before it is allowed to ship.
    """
    groups = [e.group for e in examples]
    train_idx, cal_idx, test_idx = three_way_group_split(groups, seed=seed)

    train = [examples[i] for i in train_idx]
    calibrate = [examples[i] for i in cal_idx]
    test = [examples[i] for i in test_idx]

    model = fit_model(train, alpha=alpha)

    calibration = CalibrationService()
    for e in calibrate:
        calibration.add_observation(
            e.stratum(),
            model.predict(e.spacer, e.duplex(), e.accessibility),
            e.observed_activity,
            group=e.group,
        )

    registry = ModelRegistry(model, NullAccessibilityModel())
    cases = [
        BenchmarkCase(
            spacer=e.spacer,
            target_site=e.target_site,
            observed_activity=e.observed_activity,
            stratum=e.stratum(),
            accessibility=e.accessibility,
        )
        for e in test
    ]
    benchmark = run_benchmark(registry, calibration, cases, alpha=calibration_alpha)
    cv = cross_validate(train, alpha=alpha, k=min(cv_folds, len(set(e.group for e in train))), seed=seed)

    def group_names(idx: list[int]) -> tuple[str, ...]:
        return tuple(sorted({groups[i] for i in idx}))

    return TrainingResult(
        model=model,
        calibration=calibration,
        benchmark=benchmark,
        cv=cv,
        split_sizes={"train": len(train), "calibrate": len(calibrate), "test": len(test)},
        split_groups={
            "train": group_names(train_idx),
            "calibrate": group_names(cal_idx),
            "test": group_names(test_idx),
        },
    )
