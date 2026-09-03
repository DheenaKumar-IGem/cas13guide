"""What a given run did *not* check, as data rather than prose.

A README paragraph drifts from the code; a hardcoded banner says the same thing
whether or not it is true this time. So limitations are values, and they are
**derived from the run that produced them** -- the models actually configured,
the bound the search actually proved, the calibration actually supplied.

That derivation is the part worth insisting on. A static list has two failure
modes and this package would hit both: it warns about placeholder kinetic
constants after you have supplied real ones, and it stays silent about a
heuristic off-target index because nobody remembered to add the entry. A
limitation nobody can trust is worse than none, because it trains the reader to
skim.

Everything that leaves the tool carries these: the HTML report renders them
above the guide table, and the CSV and FASTA exports carry them in the
provenance header, so a file that outlives the session still says what it did
not check.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

from ..contract.presets import (
    CHANGED_IN_RELEASE_23,
    MIRNA_PRESETS,
    PRESET_SOURCE,
    PRESETS_VERIFIED,
)
from ..contract.spec import ResolvedDesignSpec
from .indexes import ExhaustivenessBound


class Severity(str, Enum):
    """How much a limitation should change what you do next."""

    #: The result may be actively misleading until this is addressed.
    CRITICAL = "critical"
    #: The result is usable; this bounds how far you should read it.
    WARNING = "warning"
    #: Worth knowing, does not change the reading.
    INFO = "info"


#: Order limitations are presented in. Critical first: the reader who stops
#: after one line should have read the one that matters.
_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}


@dataclass(frozen=True, slots=True)
class Limitation:
    id: str
    severity: Severity
    text: str
    #: Where the claim comes from, when it is a claim about biology rather than
    #: about this code. Empty for the latter.
    source: str = ""

    def line(self) -> str:
        suffix = f" [{self.source}]" if self.source else ""
        return f"{self.severity.value.upper()}: {self.text}{suffix}"


def _calibration_is_synthetic(calibration: Any) -> bool:
    """True when the calibration carries the marker the workflow stamps on it.

    Duck-typed rather than isinstance-checked so a caller can hand in their own
    calibration object; the absence of the flag means "not marked synthetic",
    which is the right default for a service somebody built themselves.
    """
    return bool(getattr(calibration, "synthetic", False))


def _self_fold_is_proxy(provenance: Any) -> bool:
    """True when QC judged self-folding by stem count rather than by folding.

    Duck-typed over the ledger's serialised form, so a caller can hand in
    ``report.provenance`` (a dict) or anything else exposing ``records``.
    """
    if provenance is None:
        return False
    records = (
        provenance.get("records", ())
        if isinstance(provenance, dict)
        else getattr(provenance, "records", ())
    )
    for record in records:
        get = record.get if isinstance(record, dict) else lambda k: getattr(record, k, None)
        if get("name") == "qc_self_fold":
            return str(get("version") or "").startswith("qc/stem-proxy")
    return False


def limitations_for_run(
    spec: ResolvedDesignSpec,
    registry: Any = None,
    calibration: Any = None,
    bound: ExhaustivenessBound | None = None,
    universe_gaps: frozenset | None = None,
    panel_size: int = 0,
    panel_confidence: Any = None,
    provenance: Any = None,
) -> tuple[Limitation, ...]:
    """Everything this particular run could not establish.

    Every argument is optional so a caller with only a spec still gets the
    limitations that follow from the spec alone.
    """
    found: list[Limitation] = []

    # ---- calibration -------------------------------------------------
    if _calibration_is_synthetic(calibration):
        found.append(
            Limitation(
                "synthetic-calibration",
                Severity.CRITICAL,
                "Conformal bounds were calibrated on SYNTHETIC data. Every "
                "interval, and therefore every PASS, is exercisable rather than "
                "earned. Fit a real screen with `cas13guide train` before "
                "treating a bound as a guarantee.",
            )
        )

    # ---- panel independence --------------------------------------------
    if panel_size >= 2:
        found.append(
            Limitation(
                "panel-coverage-assumes-independence",
                Severity.WARNING,
                f"Panel coverage combines the {panel_size} members with a "
                "noisy-OR, which assumes they fail independently. They do not: "
                "they share a transcript, and they share one activity model "
                "whose errors are correlated. The spacing constraint only stops "
                "them sharing a site. Treat the number as a ranking aid -- this "
                "panel against another -- and not as the probability that the "
                "panel detects the target, which it overstates. The panel "
                "confidence interval (FR6.3) is the number that does not make "
                "that assumption; the noisy-OR is its upper end.",
            )
        )

    # ---- panel confidence (FR6.3) ---------------------------------------
    if panel_confidence is not None and getattr(panel_confidence, "targets", None):
        if not getattr(panel_confidence, "has_point_estimates", False):
            found.append(
                Limitation(
                    "panel-dependence-unmeasured",
                    Severity.WARNING,
                    "Panel detection probability is reported as an interval "
                    "because how much of a panel's failure is shared between "
                    "its members was never measured for this assay. The lower "
                    "end is what the panel is worth if the members fail "
                    "together, the upper end if they fail independently; the "
                    "truth is somewhere between and nothing here can locate "
                    "it. Measure the common-cause fraction from replicate "
                    "panel runs and supply it with --panel-dependence to get a "
                    "single number.",
                )
            )
        if any(getattr(t, "thin", False) for t in panel_confidence.targets):
            found.append(
                Limitation(
                    "panel-confidence-thin-calibration",
                    Severity.WARNING,
                    "At least one panel member's success probability was "
                    "estimated from a thin or unconditional slice of the "
                    "calibration set, so its credible interval is wide because "
                    "of how little data backs it rather than because the guide "
                    "is doubtful. A larger screen narrows it; nothing else "
                    "does.",
                )
            )

    # ---- preset provenance --------------------------------------------
    used = sorted(
        target.target_id
        for target in spec.contract.targets
        if target.target_id in MIRNA_PRESETS
    )
    if used and PRESETS_VERIFIED:
        moved = sorted(
            preset_id for preset_id in used
            if MIRNA_PRESETS[preset_id].accession in CHANGED_IN_RELEASE_23
        )
        if moved:
            found.append(
                Limitation(
                    "preset-sequence-changed-upstream",
                    Severity.INFO,
                    f"{', '.join(moved)} matched miRBase release "
                    f"{PRESET_SOURCE.get('release', '?')} exactly, but miRBase "
                    "changed the mature sequence of this accession in that "
                    "release. A paper written earlier quotes a different 3' end, "
                    "so a guide designed here will not line up with one "
                    "published against the older form. Neither is a typo; check "
                    "which release your comparison used.",
                    source="miRBase miRNA.diff",
                )
            )
    if used and not PRESETS_VERIFIED:
        found.append(
            Limitation(
                "preset-sequence-unverified",
                Severity.WARNING,
                "The target sequence came from a built-in preset "
                f"({', '.join(used)}). Each was checked against its miRBase "
                "per-accession page on 2026-08-23 and matched, but this "
                "package pins no release and no checksum, so it cannot say "
                "WHICH release it agrees with -- and mature 3' ends do get "
                "revised between releases. Databases also disagree with each "
                "other on where a mature miRNA ends. Run "
                "`presets.verify_against_mirbase(path_to_mature_fa)` against "
                "the release you intend to cite before ordering anything: a "
                "guide designed against the wrong target is wrong in a way "
                "no score on this page can show.",
                source="miRBase",
            )
        )

    # ---- activity model ----------------------------------------------
    activity = getattr(registry, "activity", None)
    version = getattr(activity, "version", "") or ""
    if version.startswith("model/heuristic"):
        found.append(
            Limitation(
                "heuristic-activity-model",
                Severity.CRITICAL,
                "Activity came from the built-in heuristic, which is a shaped "
                "guess and not a trained model. It saturates: a perfect duplex "
                "scores about 0.93 regardless of sequence, so the ranking among "
                "well-matched guides carries little information. Fit a model on "
                "a real screen, or plug a published one into the ActivityModel "
                "seam.",
            )
        )

    # ---- QC self-folding -----------------------------------------------
    # Read off the run's own provenance rather than off the registry: the QC
    # fold source is chosen in the orchestrator and is NOT
    # registry.accessibility, which is a different model behind a different
    # flag. Keying off the wrong one is how a limitation fires on the wrong
    # runs, which is the failure this module exists to avoid.
    if _self_fold_is_proxy(provenance):
        found.append(
            Limitation(
                "self-folding-is-a-proxy",
                Severity.WARNING,
                "Spacer self-folding was judged by counting the longest "
                "self-complementary stretch, not by folding the spacer -- "
                "ViennaRNA was not importable. That proxy recovers about a "
                "third of the guides a real fold rejects, and four in five of "
                "the ones it does reject do not fold tightly enough to meet "
                "the -8 kcal/mol criterion it stands in for -- so some "
                "self-structured spacers passed QC, and rather more were "
                "failed for a stem that is real but not disqualifying. A "
                "self-folding FAIL cannot be bought back, so install ViennaRNA "
                '(`pip install "cas13guide[structure]"`) before reading these '
                "verdicts as final.",
            )
        )

    # ---- structure ----------------------------------------------------
    accessibility = getattr(registry, "accessibility", None)
    access_version = getattr(accessibility, "version", "") or ""
    if access_version.startswith("accessibility/none"):
        # A floor the user set explicitly is a different situation from one
        # they never asked for. The gate skips TARGET_INACCESSIBLE when the
        # metric is None -- correctly, since absent evidence is not bad
        # evidence -- so `--min-accessibility 0.9` with no folding engine
        # installed asks for a gate that cannot fire, and the generic warning
        # never connected the two. Say which number went unused.
        floor = spec.thresholds.min_accessibility
        if floor > 0.0:
            found.append(
                Limitation(
                    "accessibility-floor-inert",
                    Severity.CRITICAL,
                    f"An accessibility floor of {floor:.2f} was requested, and "
                    "no accessibility was computed, so the floor was NOT "
                    "applied -- no guide was held back for being buried. "
                    'Install ViennaRNA (`pip install "cas13guide[structure]"`) '
                    "or drop the floor, but do not read this run as having "
                    "enforced it.",
                )
            )
        else:
            found.append(
                Limitation(
                    "no-accessibility",
                    Severity.WARNING,
                    "No structural accessibility was computed, so a site buried in a "
                    "stable hairpin is scored the same as an open one. Install "
                    'ViennaRNA (`pip install "cas13guide[structure]"`) to enable it.',
                )
            )
    elif not getattr(accessibility, "is_independent_evidence", False):
        found.append(
            Limitation(
                "accessibility-is-predicted",
                Severity.INFO,
                "Accessibility is a thermodynamic fold, not a measurement. It "
                "ranks candidates but deliberately does not tighten any "
                "calibrated bound, because it is a prediction from the same "
                "sequence the activity model already sees. A SHAPE, DMS-seq or "
                "icSHAPE track is independent evidence and does reach the model.",
            )
        )

    energy_version = getattr(getattr(registry, "energy", None), "version", "") or ""
    window = (spec.thresholds.duplex_dg_per_nt_min, spec.thresholds.duplex_dg_per_nt_max)
    if energy_version.startswith("thermodynamics/none") and any(
        bound is not None for bound in window
    ):
        low, high = window
        found.append(
            Limitation(
                "duplex-window-inert",
                Severity.CRITICAL,
                f"A duplex energy window of [{low}, {high}] kcal/mol/nt was "
                "requested, and no duplex energy was computed, so the window "
                "was NOT applied -- no guide was flagged for binding too "
                "tightly or too weakly. Install ViennaRNA "
                '(`pip install "cas13guide[structure]"`) or drop the window.',
            )
        )

    # ---- kinetics / signal --------------------------------------------
    kinetics = getattr(registry, "kinetics", None)
    constants = getattr(kinetics, "constants", None)
    if constants is not None and getattr(constants, "provisional", False):
        if not spec.profile.collateral_is_hazard:
            found.append(
                Limitation(
                    "provisional-kinetics",
                    Severity.CRITICAL,
                    "Predicted signal uses PLACEHOLDER kinetic constants. The "
                    "ordering among guides is meaningful because every candidate "
                    "goes through the same model, but the values are relative to "
                    "this run and are not fluorescence. Absolute predictions are "
                    "refused until measured constants are supplied.",
                )
            )
    if (
        kinetics is not None
        and not spec.profile.collateral_is_hazard
        and not getattr(kinetics, "adds_information", lambda: True)()
    ):
        found.append(
            Limitation(
                "signal-adds-no-information",
                Severity.WARNING,
                "Predicted signal is reported but is NOT ranked on, because with "
                "these constants it is a rescaled copy of two numbers already in "
                "the ranking: it comes out as a constant multiple of "
                "activity_lower x accessibility, so it cannot separate two guides "
                "that those do not already separate. Left in the objective set it "
                "would silently re-weight them rather than add an axis. The ODE "
                "stops being proportional once effective target approaches the "
                "effector concentration, and the axis returns on its own when "
                "constants that reach that regime are supplied.",
            )
        )
    if spec.profile.collateral_is_hazard:
        found.append(
            Limitation(
                "collateral-not-predicted",
                Severity.CRITICAL,
                "Collateral (trans) cleavage is not predicted for this assay. "
                "Once activated, Cas13 degrades bystander RNA, and that scales "
                "with target abundance rather than with off-target homology -- "
                "so no specificity score on this page bounds it. Assay it "
                "directly.",
                source="Wessels et al., Nat Biotechnol 2020",
            )
        )

    # ---- search scope --------------------------------------------------
    if bound is not None:
        if not bound.exhaustive:
            found.append(
                Limitation(
                    "search-not-exhaustive",
                    Severity.CRITICAL,
                    "The off-target search was HEURISTIC, so a guide reported as "
                    f"specific may simply have had its off-targets missed. {bound.note}",
                )
            )
        if not bound.wobble_aware:
            found.append(
                Limitation(
                    "search-not-wobble-aware",
                    Severity.WARNING,
                    "The off-target search scored G-U pairs as mismatches while "
                    "seeding. For an RNA:RNA duplex a wobble is a real base "
                    "pair, so wobble-rich off-targets are under-found.",
                )
            )
        found.append(
            Limitation(
                "search-scope-is-what-you-loaded",
                Severity.CRITICAL,
                "Off-targets were searched over "
                + (", ".join(sorted(u.value for u in bound.universes)) or "nothing")
                + " in the sequences supplied to this run. Anything not loaded is "
                "invisible: a guide is specific *against what you gave it*.",
            )
        )
    if universe_gaps:
        found.append(
            Limitation(
                "universe-gap",
                Severity.CRITICAL,
                "The assay exposes the guide to universes the search did not "
                "cover: " + ", ".join(sorted(u.value for u in universe_gaps))
                + ". A missing universe does not produce a warning-shaped result, "
                "it produces a suspiciously clean one.",
            )
        )

    # ---- spec-derived ---------------------------------------------------
    if not spec.profile.spacer_length_is_canonical:
        params = spec.profile.params
        found.append(
            Limitation(
                "non-canonical-spacer-length",
                Severity.WARNING,
                f"{spec.spacer_length} nt is outside "
                f"{spec.profile.effector.value}'s published lengths "
                f"{params.spacer_lengths}. The design is expressible and is "
                "reported out-of-domain, so bounds widen rather than narrow -- "
                "but no model here was validated at this length.",
            )
        )

    found.append(
        Limitation(
            "predictions-not-measurements",
            Severity.INFO,
            "Every score on this page is a prediction. A high one makes a guide "
            "a good candidate, not a certainty. Order several and test them.",
        )
    )
    found.sort(key=lambda item: (_SEVERITY_ORDER[item.severity], item.id))
    return tuple(found)


def as_lines(limitations: Sequence[Limitation]) -> list[str]:
    return [item.line() for item in limitations]


def worst_severity(limitations: Sequence[Limitation]) -> Severity | None:
    if not limitations:
        return None
    return min(limitations, key=lambda item: _SEVERITY_ORDER[item.severity]).severity
