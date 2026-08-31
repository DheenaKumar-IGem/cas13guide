"""Structural accessibility from a thermodynamic fold (FR3.1-FR3.4).

A Cas13 guide has to invade whatever secondary structure its target already
has, so a site buried in a stem is a worse site than an equally-matched one in
a loop. This module scores that, using ViennaRNA's local folding.

Local, not global
-----------------
Folding a whole transcript once and reading pair probabilities off it is both
slow (O(n^3)) and wrong for long RNAs: base pairs spanning kilobases dominate
the partition function while, in a cell, co-transcriptional folding and
helicases make long-range pairs far less relevant than local ones.
``RNAplfold`` instead folds in a sliding window and reports the probability
that each nucleotide is unpaired given only pairs within that window, which is
what FR3.2 asks for and is O(n * W^2).

Why this is *not* fed to the activity model
-------------------------------------------
``pipeline/d_pair_evidence`` credits accessibility to the activity model only
when it comes from a source independent of that model -- SHAPE, DMS-seq,
icSHAPE. A ViennaRNA fold is not independent evidence: it is a prediction from
the same sequence the model already sees, and the model's feature vector
already carries composition and duplex terms derived from it. Feeding it in
would let one computation vouch for another and narrow a bound that nothing new
has actually constrained.

So a thermodynamic accessibility carries ``is_independent_evidence = False``
and is used the way the PFS preference is used: it ranks candidates and it can
raise a review, but it never tightens a calibrated interval. A real probing
dataset dropped in behind the same Protocol sets the flag True and *does* reach
the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Sliding window, in nucleotides, that local folding is computed over.
#: FR3.2 asks for 70; it is also ViennaRNA's own default for RNAplfold.
DEFAULT_WINDOW = 70

#: Longest base pair span considered inside a window. Equal to the window by
#: default, i.e. no extra restriction beyond the window itself.
DEFAULT_MAX_BP_SPAN = 70

#: Mean unpaired probability below which a site counts as "highly structured"
#: for FR3.4 reporting. Advisory: the gate threshold is a spec decision and
#: lives in ``Thresholds.min_accessibility``, defaulting to off.
STRUCTURED_SITE_THRESHOLD = 0.25

#: Shortest sequence that can contain a base pair: ViennaRNA's minimum hairpin
#: loop is 3 nt, plus the two closing bases. Below this nothing pairs, which is
#: why the answer is computable without folding -- and it has to be, because
#: ``RNA.pfl_fold_up`` aborts the process at 3 and 4 nt.
MIN_PAIRABLE_LENGTH = 5


class AccessibilityUnavailable(RuntimeError):
    """ViennaRNA is not importable, so no fold-based accessibility is possible."""


@dataclass(slots=True)
class ViennaRNAAccessibilityModel:
    """Per-nucleotide unpaired probability from ``RNAplfold`` (FR3.1-FR3.3).

    One profile is computed per locus and cached, because a design run asks
    about hundreds of windows on the same handful of transcripts and the fold
    is by far the most expensive thing in the pipeline.
    """

    window: int = DEFAULT_WINDOW
    max_bp_span: int = DEFAULT_MAX_BP_SPAN
    version: str = "accessibility/vienna-rnaplfold/v1"
    #: A computed fold, not a measurement. See the module docstring.
    is_independent_evidence: bool = False
    _profiles: dict[str, tuple[float, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        try:
            import RNA  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise AccessibilityUnavailable(
                "ViennaRNA is not installed. `pip install ViennaRNA`, or use "
                "NullAccessibilityModel to run without structural evidence."
            ) from exc
        self.version = (
            f"accessibility/vienna-rnaplfold/w{self.window}/span{self.max_bp_span}/v1"
        )

    def unpaired_profile(self, sequence: str) -> tuple[float, ...]:
        """Probability that each nucleotide is unpaired, 5'->3'.

        Sequences shorter than the window are folded whole -- ``RNAplfold``
        handles that, and a short transcript has no long-range pairs to exclude
        anyway. *Except* at 3 and 4 nt, where ``RNA.pfl_fold_up`` aborts the
        process: not an exception, a native abort that kills the interpreter
        with no traceback and, under ``serve``, would take every in-flight
        design down with it. Those two lengths are answered without folding --
        see below.
        """
        cached = self._profiles.get(sequence)
        if cached is not None:
            return cached

        import RNA

        if not sequence:
            profile: tuple[float, ...] = ()
        elif len(sequence) < MIN_PAIRABLE_LENGTH:
            # Below a hairpin's minimum loop plus its two closing bases nothing
            # can pair, so every position is unpaired with probability 1. This
            # is the answer the partition function gives for these lengths
            # (checked against RNA.fold_compound().pf() for 1-5 nt), computed
            # rather than obtained, because obtaining it crashes.
            profile = (1.0,) * len(sequence)
        else:
            window = min(self.window, len(sequence))
            span = min(self.max_bp_span, window)
            # pfl_fold_up returns a 1-based table: up[i][u] is the probability
            # that the u-long stretch ending at position i is unpaired.
            table = RNA.pfl_fold_up(sequence, 1, window, span)
            profile = tuple(float(table[i][1]) for i in range(1, len(sequence) + 1))
        self._profiles[sequence] = profile
        return profile

    def accessibility(self, sequence: str, start: int, length: int) -> float | None:
        """Mean unpaired probability across ``sequence[start:start+length]`` (FR3.3).

        ``None`` when the window does not lie inside the sequence, which is the
        contract's way of saying "no evidence" rather than guessing a value.
        """
        if length <= 0 or start < 0 or start + length > len(sequence):
            return None
        profile = self.unpaired_profile(sequence)
        if not profile:
            return None
        window = profile[start : start + length]
        return sum(window) / len(window)

    def is_structured(self, sequence: str, start: int, length: int) -> bool | None:
        """FR3.4: does this site sit in a highly structured region?"""
        score = self.accessibility(sequence, start, length)
        if score is None:
            return None
        return score < STRUCTURED_SITE_THRESHOLD


@dataclass(frozen=True, slots=True)
class StaticAccessibilityModel:
    """Accessibility supplied from outside -- a probing experiment, typically.

    The seam a real SHAPE/DMS-seq/icSHAPE track plugs into. Unlike a fold this
    *is* independent evidence, so it flows into the activity model.
    """

    profiles: dict[str, tuple[float, ...]]
    version: str = "accessibility/static-probing/v1"
    is_independent_evidence: bool = True

    def accessibility(self, sequence: str, start: int, length: int) -> float | None:
        profile = self.profiles.get(sequence)
        if profile is None or length <= 0 or start < 0 or start + length > len(profile):
            return None
        window = profile[start : start + length]
        return sum(window) / len(window)
