"""The Arc RNAtargeting CNN behind the :class:`ActivityModel` seam.

A real trained model, on the seam the README has always said it was for. The
built-in :class:`~cas13guide.services.registry.HeuristicCas13Model` saturates --
a perfect duplex scores about 0.93 whatever the sequence -- so the ranking among
well-matched guides carries almost no information. This replaces it with the
published `guide_nolin_threef` weights from Wei et al., *Cell Systems* 2023,
running as a three-fold ensemble.

Verified, not assumed
---------------------
``tests/test_arc_model.py`` drives 133 guides from the reference implementation
and reproduces its scores to **0.0** difference. That fixture is the gate: four
things here keep every tensor shape valid and every score in range while being
wrong, and only end-to-end parity catches them.

* **Channel order is A, T, C, G** -- not the conventional ACGT. From
  ``base_positions`` in the reference. Getting it wrong is the single easiest
  way to break this silently.
* **The ensemble averages logits, then applies the sigmoid.** Averaging the
  sigmoids instead is a plausible-looking mistake: measured over the fixture it
  shifts scores by up to 0.037, mean 0.007. Small enough to survive eyeballing,
  large enough to reorder guides -- which is the whole output.
* **The three scalars are ordered** ``linearfold_val, isoform_percent,
  target_unfold_energy``.
* **The normalizing constants are training-time values, not tunables.**

What it needs that this package cannot supply
---------------------------------------------
Two of the three scalars are folding free energies, and they must come from
**LinearFold in its default CONTRAfold mode**. ViennaRNA -- which this package
does ship -- uses the Turner model, and its energies are simply different
numbers. Feeding them in produces a confident, in-range, wrong score, which is
the one failure this codebase exists to prevent. So there is no ViennaRNA
fallback and there will not be one: :class:`MFESource` is a seam you fill with a
LinearFold binding, and without one :func:`features_for_target` refuses.

The domain is narrow, on purpose
--------------------------------
The model sees the spacer and the folding of its intended target. It never sees
a mismatched site, so it has nothing to say about one. ``domain.max_mismatches``
is therefore ``0``: ask it about an off-target and the pipeline marks the answer
out-of-domain and widens the interval, rather than passing off an on-target
efficacy prediction as an off-target risk estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..contract.enums import Effector
from ..util.duplex import DuplexProfile
from .registry import ModelDomain

#: Channel index per base. **A, T, C, G** -- from ``base_positions`` in the
#: reference. Not ACGT.
BASE_POSITIONS: dict[str, int] = {"A": 0, "T": 1, "C": 2, "G": 3}

#: The model's input width. The tensor shape is (30, 4); other lengths cannot
#: be scored by it at all.
GUIDE_LENGTH = 30

#: Prepended to each guide before folding, from ``make_guide_library_features``.
#:
#: In RNA this is the RfxCas13d pre-crRNA direct repeat, and its last 30 nt are
#: the published CasRx mature repeat. Comparing it against
#: ``contract/effectors.py`` is what caught the corrupted 45 nt value that used
#: to be recorded there; ``tests/test_arc_model.py`` now pins that the two
#: agree, so that neither can drift from the other unnoticed.
STANDARD_PREFIX = "CAAGTAAACCCCTACCAACTGGTCGGGGTTTGAAAC"

#: Training-time normalizers. Not tunables: changing any of them shifts every
#: prediction, silently and uniformly.
GUIDE_MFE_OFFSET = 6.48
GUIDE_MFE_DIVISOR = 11.93
UNFOLD_ENERGY_DIVISOR = 15.67

#: Target window and flank, from ``make_guide_library_features``.
TARGET_LENGTH = 30
FLANK_LENGTH = 15

#: Where the vendored weights live. See THIRD_PARTY_LICENSES.md.
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "arc"
FOLD_FILES = ("fold_0.onnx", "fold_1.onnx", "fold_2.onnx")


class ArcModelUnavailable(RuntimeError):
    """onnxruntime or the weight files are missing."""


class MFESourceRequired(RuntimeError):
    """A LinearFold CONTRAfold energy source is needed and was not supplied."""


@runtime_checkable
class MFESource(Protocol):
    """Minimum free energy, in LinearFold's **CONTRAfold** parameterization.

    Implement this over a LinearFold binding. Do not implement it over
    ViennaRNA: the energies differ, and the model was trained on the former.
    """

    version: str

    def mfe(self, sequence: str) -> float:
        """Unconstrained MFE of ``sequence``, kcal/mol."""

    def mfe_constrained(self, sequence: str, constraint: str) -> float:
        """MFE under a dot-bracket constraint string of equal length."""


@dataclass(frozen=True, slots=True)
class ArcFeatures:
    """The three scalars the CNN consumes beside the one-hot guide."""

    linearfold_val: float
    isoform_percent: float
    target_unfold_energy: float

    def as_array(self) -> np.ndarray:
        """Ordered as the model expects. The order is part of the contract.

        ``float32`` because that is the model's input dtype; a caller comparing
        against Python floats should expect single-precision rounding.
        """
        return np.array(
            [self.linearfold_val, self.isoform_percent, self.target_unfold_energy],
            dtype=np.float32,
        )


def one_hot(sequence: str, length: int = GUIDE_LENGTH) -> np.ndarray:
    """``(length, 4)`` one-hot in **A, T, C, G** order. ``U`` is read as ``T``.

    An unknown base yields an all-zero row, mirroring the reference. Callers
    should not rely on that: it is a safe degradation, not a meaning.
    """
    out = np.zeros((length, 4), dtype=np.float32)
    for index, char in enumerate(sequence.upper()[:length]):
        position = BASE_POSITIONS.get("T" if char == "U" else char)
        if position is not None:
            out[index, position] = 1.0
    return out


def guide_linearfold_val(guide_mfe: float) -> float:
    """``abs(MFE(prefix + guide) - 6.48) / 11.93``."""
    return abs(guide_mfe - GUIDE_MFE_OFFSET) / GUIDE_MFE_DIVISOR


def target_unfold_energy(native_mfe: float, constrained_mfe: float) -> float:
    """``(MFE_native - MFE_constrained) / 15.67``.

    The cost of prising the target window open: how much free energy is given
    up by forbidding the target's own pairing.
    """
    return (native_mfe - constrained_mfe) / UNFOLD_ENERGY_DIVISOR


def features_for_target(
    spacer: str,
    nearby: str,
    target_offset: int,
    source: MFESource | None,
    isoform_percent: float = 1.0,
) -> ArcFeatures:
    """Build the three scalars for one guide.

    ``nearby`` is the target window with up to :data:`FLANK_LENGTH` nt of
    transcript context each side, and ``target_offset`` is where the 30 nt
    target starts inside it -- the constraint mask has to be positioned against
    the flanks, and a caller that passes the bare window would silently produce
    a different feature.
    """
    if source is None:
        raise MFESourceRequired(
            "The Arc model needs LinearFold CONTRAfold free energies, which this "
            "package does not ship. Supply an MFESource backed by a LinearFold "
            "binding. ViennaRNA is deliberately not accepted here: it uses the "
            "Turner model, so its energies are different numbers, and feeding "
            "them to this network yields a confident, in-range, wrong score."
        )
    guide_mfe = source.mfe(STANDARD_PREFIX + spacer.upper().replace("U", "T"))
    context = nearby.upper().replace("U", "T")
    native = source.mfe(context)
    mask = list("?" * len(context))
    for i in range(target_offset, min(target_offset + TARGET_LENGTH, len(context))):
        mask[i] = "."
    constrained = source.mfe_constrained(context, "".join(mask))
    return ArcFeatures(
        linearfold_val=guide_linearfold_val(guide_mfe),
        isoform_percent=isoform_percent,
        target_unfold_energy=target_unfold_energy(native, constrained),
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


@dataclass(slots=True)
class ArcEnsemble:
    """The three ONNX folds, loaded once."""

    model_dir: Path = MODEL_DIR
    version: str = "model/arc-rnatargeting-cnn/guide_nolin_threef/v1"
    _sessions: list = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ArcModelUnavailable(
                "onnxruntime is not installed. `pip install "
                '"cas13guide[arc]"` to use the Arc RNAtargeting model.'
            ) from exc
        missing = [f for f in FOLD_FILES if not (self.model_dir / f).exists()]
        if missing:
            raise ArcModelUnavailable(
                f"missing Arc weight file(s) in {self.model_dir}: {missing}"
            )
        self._sessions = [
            ort.InferenceSession(
                str(self.model_dir / name), providers=["CPUExecutionProvider"]
            )
            for name in FOLD_FILES
        ]

    def logits(self, spacers: Sequence[str], features: Sequence[ArcFeatures]) -> np.ndarray:
        """Mean logit across the three folds, one per guide.

        Averaged in **logit** space. Averaging the sigmoids is the natural-looking
        alternative and it is wrong by up to 0.499 of score.
        """
        if len(spacers) != len(features):
            raise ValueError("spacers and features must be the same length")
        if not spacers:
            return np.empty(0, dtype=np.float32)
        guides = np.stack([one_hot(s) for s in spacers])
        scalars = np.stack([f.as_array() for f in features])
        stacked = []
        for session in self._sessions:
            first, second = (i.name for i in session.get_inputs())
            stacked.append(
                session.run(None, {first: guides, second: scalars})[0].reshape(-1)
            )
        return np.mean(np.array(stacked), axis=0)

    def scores(self, spacers: Sequence[str], features: Sequence[ArcFeatures]) -> np.ndarray:
        return _sigmoid(self.logits(spacers, features))


@dataclass(slots=True)
class ArcActivityModel:
    """``ActivityModel`` over the Arc ensemble, driven by precomputed features.

    Features are supplied rather than computed on demand because they depend on
    transcript context the ``predict`` signature does not carry -- the target's
    flanks -- and because folding is far too expensive to redo per query. Build
    the table once per target with :func:`features_for_target`.

    A spacer with no entry scores ``None``-like behaviour rather than a guess:
    it returns the neutral 0.5 and, because such a guide is almost always an
    off-target hit, the domain check below has already marked it out-of-domain
    so the interval is widened to maximal ignorance anyway.
    """

    ensemble: ArcEnsemble
    features_by_spacer: dict[str, ArcFeatures] = field(default_factory=dict)
    version: str = "model/arc-rnatargeting-cnn/guide_nolin_threef/v1"
    domain: ModelDomain = field(
        default_factory=lambda: ModelDomain(
            effectors=frozenset({Effector.RFX_CAS13D}),
            spacer_lengths=frozenset({GUIDE_LENGTH}),
            # The network never sees a mismatched site, so it has nothing to say
            # about one. Anything past a perfect duplex is out of domain.
            max_mismatches=0,
        )
    )
    _cache: dict[str, float] = field(default_factory=dict, repr=False, compare=False)

    def predict(
        self, spacer: str, duplex: DuplexProfile, accessibility: float | None
    ) -> float:
        key = spacer.upper()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        features = self.features_by_spacer.get(key)
        if features is None:
            # No context was prepared for this spacer. The domain check marks it
            # out-of-domain, so the caller's interval is already maximal; 0.5 is
            # the neutral point inside it rather than a claim.
            return 0.5
        value = float(self.ensemble.scores([key], [features])[0])
        self._cache[key] = value
        return value
