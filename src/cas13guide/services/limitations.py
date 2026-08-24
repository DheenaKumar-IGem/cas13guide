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


def limitations_for_run(
    spec: ResolvedDesignSpec,
    registry: Any = None,
    calibration: Any = None,
    bound: ExhaustivenessBound | None = None,
    universe_gaps: frozenset | None = None,
    panel_size: int = 0,
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
                "panel detects the target, which it overstates.",
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
        if used:
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

    # ---- structure ----------------------------------------------------
    accessibility = getattr(registry, "accessibility", None)
    access_version = getattr(accessibility, "version", "") or ""
    if access_version.startswith("accessibility/none"):
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
