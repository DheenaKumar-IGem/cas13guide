"""Calibration service -- promoted to first class in design revision v2 (fix #1).

Every ``LCB``/``UCB`` in this system is only meaningful if the model producing
it is calibrated *for the case at hand*. For Cas13 that caveat bites hard:
activity models are fit on particular effectors, tiling screens and cell
lines, so out-of-domain is the normal case rather than the exception.

Method
------
Split conformal prediction. For a stratum with calibration pairs
``(pred_i, obs_i)`` and nonconformity scores ``s_i = |obs_i - pred_i|``, the
interval ``[pred - q, pred + q]`` with

    q = Quantile_{ceil((n+1)(1-alpha)) / n} (s)

has marginal coverage ``>= 1 - alpha`` under exchangeability, distribution-free.

Strata are tried most-specific first and fall back down a chain; if no stratum
has enough calibration data, the service returns a maximally wide interval
flagged ``ood`` rather than a confident-looking narrow one. That flag
propagates to the gate as an R3 REVIEW.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..metrics.bounds import Interval

#: Below this many calibration points a stratum is not trusted on its own.
MIN_STRATUM_N = 20

#: Below this many *independent groups* (transcripts) a stratum's quantile is
#: too variable to certify, however many rows it holds -- see the effective
#: sample size note in :class:`CalibrationStratum`. Chosen where the spread of
#: empirical coverage across seeds starts to tighten, not where the mean moves.
MIN_STRATUM_GROUPS = 8


@dataclass(slots=True)
class CalibrationStratum:
    """Held-out calibration pairs for one (effector, assay, regime) cell.

    Effective sample size, and what grouping costs you
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Split conformal's coverage guarantee assumes calibration and test points
    are exchangeable. Under grouped splitting they are not: calibration rows
    come from a handful of transcripts, rows within a transcript are strongly
    correlated, and the test set is a *different* set of transcripts. So the
    effective number of independent samples is the number of transcripts, not
    the number of rows, and the marginal guarantee is approximate.

    Measured on simulated screens, nominal 0.90, mean over 9 seeds:

    ============  =========  =======  =========
    calibration   mean       sd       worst
    transcripts   coverage            seed
    ============  =========  =======  =========
    2             0.881      0.124    0.585
    4             0.852      0.113    0.667
    8             0.840      0.096    0.609
    21            0.889      0.057    0.808
    40            0.876      0.048    0.773
    ============  =========  =======  =========

    The honest reading is not "few groups means low coverage" -- the *mean* is
    roughly flat, and slightly under nominal throughout, which is the
    exchangeability violation showing up as a small optimistic bias. What
    collapses with few groups is **reliability**: with two calibration
    transcripts a single screen can land at 0.585 actual coverage; with forty
    the worst case seen was 0.773.

    Two consequences, both implemented:

    * a stratum backed by too few independent groups still returns its bound,
      but flagged ``ood`` -- the number is usable, it is just not certified;
    * coverage is never assumed, only measured. ``services/benchmark.py``
      checks it on held-out groups and fails CI when it breaks, which is the
      protection that actually holds.
    """

    key: str
    predicted: list[float] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)
    groups: set[str] = field(default_factory=set)

    def add(self, predicted: float, observed: float, group: str | None = None) -> None:
        self.predicted.append(predicted)
        self.observed.append(observed)
        if group is not None:
            self.groups.add(group)

    @property
    def n(self) -> int:
        return len(self.predicted)

    @property
    def n_groups(self) -> int:
        """Independent units backing this stratum.

        ``0`` means the caller supplied no group labels, in which case group
        adequacy cannot be judged and only the row count is checked.
        """
        return len(self.groups)

    def residuals(self) -> list[float]:
        return [abs(o - p) for p, o in zip(self.predicted, self.observed)]

    def quantile(self, alpha: float) -> tuple[float, bool]:
        """Conformal quantile with the finite-sample ``(n+1)`` correction.

        Returns ``(q, certified)``. Coverage ``>= 1 - alpha`` holds only when
        ``certified`` is True; that requires ``ceil((n+1)(1-alpha)) <= n``,
        i.e. ``alpha >= 1/(n+1)``. Below that the stratum simply does not have
        enough points to certify the level, and the widest observed residual
        caps coverage at ``n/(n+1)``. Callers must surface the uncertified
        case rather than presenting a quietly-too-narrow interval.
        """
        n = self.n
        if n == 0:
            return 1.0, False
        scores = sorted(self.residuals())
        rank = math.ceil((n + 1) * (1.0 - alpha))
        if rank > n:
            return max(scores), False
        return scores[rank - 1], True


@dataclass(frozen=True, slots=True)
class CoverageReport:
    stratum: str
    n: int
    nominal: float
    empirical: float

    @property
    def ok(self) -> bool:
        # Small tolerance for finite-sample slack.
        return self.empirical >= self.nominal - 0.05


class CalibrationService:
    """Stratified conformal bounds with an explicit fallback chain."""

    version = "calibration/split-conformal/v1"

    def __init__(
        self,
        min_stratum_n: int = MIN_STRATUM_N,
        min_stratum_groups: int = MIN_STRATUM_GROUPS,
    ) -> None:
        self._strata: dict[str, CalibrationStratum] = {}
        self.min_stratum_n = min_stratum_n
        self.min_stratum_groups = min_stratum_groups

    # -- fitting -------------------------------------------------------
    @staticmethod
    def mismatch_regime(n_mismatch: int) -> str:
        """Bucket a duplex by how hard it is for the model to predict.

        Stratification only helps when the strata have genuinely different
        error distributions. Mismatch depth does: a perfect duplex sits in the
        dense middle of any training screen, while a 4-mismatch duplex is
        sparse and the model is extrapolating. Intent labels (REQUIRED,
        FORBIDDEN, ...) do *not* -- they describe what the designer wants, not
        how the model behaves, and keying on them fragments the calibration
        set along a boundary the error does not respect.
        """
        if n_mismatch <= 0:
            return "mm0"
        if n_mismatch <= 2:
            return "mm1-2"
        return "mm3+"

    @staticmethod
    def stratum_key(effector: str, assay: str, regime: str) -> str:
        """``effector|assay|regime`` -- see :meth:`mismatch_regime`."""
        return f"{effector}|{assay}|{regime}"

    def add_observation(
        self, key: str, predicted: float, observed: float, group: str | None = None
    ) -> None:
        """Record one calibration pair.

        ``group`` is the independent unit this observation came from -- a
        transcript, for a tiling screen. Supplying it lets the service notice
        when a stratum is built from too few independent units to certify.
        """
        self._strata.setdefault(key, CalibrationStratum(key)).add(predicted, observed, group)

    def add_many(self, key: str, pairs, group: str | None = None) -> None:
        for predicted, observed in pairs:
            self.add_observation(key, predicted, observed, group)

    # -- fallback chain ------------------------------------------------
    @staticmethod
    def _fallback_chain(key: str) -> list[str]:
        """Most specific first: exact cell, then regime-agnostic, then global."""
        effector, assay, regime = key.split("|")
        return [
            key,
            f"{effector}|{assay}|*",
            f"{effector}|*|*",
            "*|*|*",
        ]

    def _resolve(self, key: str) -> CalibrationStratum | None:
        for candidate in self._fallback_chain(key):
            stratum = self._strata.get(candidate)
            if stratum is not None and stratum.n >= self.min_stratum_n:
                return stratum
        return None

    def _groups_adequate(self, stratum: CalibrationStratum) -> bool:
        """False when too few independent units back this stratum.

        A stratum with no group labels at all cannot be judged, so it passes;
        the caller simply gets no protection from this check.
        """
        if stratum.n_groups == 0:
            return True
        return stratum.n_groups >= self.min_stratum_groups

    # -- use -----------------------------------------------------------
    def interval(
        self,
        prediction: float,
        key: str,
        alpha: float,
        lo: float = 0.0,
        hi: float = 1.0,
    ) -> Interval:
        """Calibrated interval around ``prediction`` for stratum ``key``.

        Returns a maximally wide, ``ood``-flagged interval when no stratum in
        the fallback chain has enough data. That is the honest answer, and the
        gate is built to surface it rather than average it away.
        """
        stratum = self._resolve(key)
        if stratum is None:
            return Interval.unknown(lo=lo, hi=hi, stratum=f"{key}->none")

        q, certified = stratum.quantile(alpha)
        exact = stratum.key == key
        groups_ok = self._groups_adequate(stratum)

        label = stratum.key
        if not certified:
            label = f"{stratum.key}->uncertified@alpha={alpha}"
        elif not groups_ok:
            label = f"{stratum.key}->only-{stratum.n_groups}-groups"

        return Interval(
            point=prediction,
            lower=prediction - q,
            upper=prediction + q,
            # A fallback stratum, an uncertifiable level, or too few
            # independent groups all mean this bound is not the guarantee the
            # caller asked for.
            ood=not exact or not certified or not groups_ok,
            stratum=label,
        ).clamped(lo, hi)

    # -- validation (CI gate) ------------------------------------------
    def validate_coverage(self, key: str, held_out, alpha: float) -> CoverageReport:
        """Empirical coverage of the conformal interval on held-out pairs.

        Wired into the benchmark harness so a model or calibration change that
        breaks coverage fails CI instead of silently degrading every bound
        downstream.
        """
        pairs = list(held_out)
        if not pairs:
            return CoverageReport(key, 0, 1 - alpha, 0.0)
        covered = 0
        for predicted, observed in pairs:
            iv = self.interval(predicted, key, alpha)
            if iv.lower <= observed <= iv.upper:
                covered += 1
        return CoverageReport(
            stratum=key, n=len(pairs), nominal=1 - alpha, empirical=covered / len(pairs)
        )

    def describe(self) -> dict[str, int]:
        return {k: s.n for k, s in sorted(self._strata.items())}

    def describe_groups(self) -> dict[str, int]:
        """Independent units per stratum -- the number that actually governs
        how tight a certified bound can be."""
        return {k: s.n_groups for k, s in sorted(self._strata.items())}
