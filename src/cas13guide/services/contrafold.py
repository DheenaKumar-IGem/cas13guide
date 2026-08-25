"""LinearFold-CONTRAfold folding scores, without a compiler.

This is the seam :mod:`cas13guide.services.arc_model` refuses to fill with
ViennaRNA. The Arc CNN was trained on **LinearFold in CONTRAfold mode**, and
until now nothing on a Windows box could produce those numbers: LinearFold ships
no Windows wheel on PyPI, bioconda has no ``win-64`` build, its GitHub has no
binary releases at all, and there is no C++ toolchain here to build it from
source.

What makes this possible is that CONTRAfold's model is a CRF with published
parameters, and ``gpu-contrafold`` reimplements it in pure Python over numpy and
numba -- a ``py3-none-any`` wheel that installs anywhere. Its exact Viterbi DP
is the same recurrence LinearFold runs.

Two things stand between "same model" and "same numbers", and both are load
bearing.

Sharp turns
-----------
LinearFold defaults to ``sharpturn=false``, which forbids a hairpin closed on
fewer than three unpaired bases. CONTRAfold -- and so ``gpu-contrafold`` --
allows them. On the 266 sequences in the Arc fixture this never comes up, so a
parity test built only from that fixture passes green with the difference
present. It is not rare in general: on random 66-mers it changes roughly one
sequence in twenty, and on GC-rich ones closer to one in six.

It is imposed here through the parameters rather than by editing the
recurrence. The hairpin term is a table lookup ``hp_cum[loop_length]``, so
setting the entries for loop lengths 0, 1 and 2 to negative infinity makes those
hairpins unreachable -- exactly what ``no_sharp_turn`` means, with no risk of
mis-editing a dynamic program.

Two decimals
------------
LinearFold prints its score with ``printf("%.2f")``, and the Arc pipeline parsed
that text. So the quantisation is part of the number, not a display detail:
leaving the extra precision in shifts Arc logits by ~2.9e-03, which fails the
parity gate in ``tests/test_arc_model.py``. :meth:`ContrafoldMFESource.mfe`
rounds, and the rounding is tested.

Not kcal/mol
------------
The value is CONTRAfold's raw CRF score: dimensionless log-odds where **higher
means more stable**, which is the opposite sign convention to a free energy. See
:class:`~cas13guide.services.arc_model.MFESource`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Hairpin loop lengths LinearFold forbids by default (``sharpturn=false``): a
#: hairpin needs at least three unpaired bases to close.
FORBIDDEN_HAIRPIN_LOOPS = (0, 1, 2)

#: LinearFold prints ``%.2f``; the Arc pipeline parsed that text.
SCORE_DECIMALS = 2


class ContrafoldUnavailable(RuntimeError):
    """``gpu-contrafold`` is not installed."""


def _cpu():
    try:
        from gpu_contrafold import cpu
    except ImportError as exc:  # pragma: no cover - exercised by the message test
        raise ContrafoldUnavailable(
            "LinearFold-CONTRAfold folding scores need the `gpu-contrafold` "
            'package. Install with: pip install "cas13guide[contrafold]". '
            "ViennaRNA is NOT a substitute: it is the Turner model, in different "
            "units, with the opposite sign convention."
        ) from exc
    return cpu


def no_sharp_turn(params: dict) -> dict:
    """Copy of ``params`` with sub-3-nt hairpin loops made unreachable.

    LinearFold's default. Applied to the parameter table rather than to the
    recurrence, because a lookup table is a safe thing to edit and a dynamic
    program is not.
    """
    cpu = _cpu()
    out = dict(params)
    hairpin = np.array(out["hp_cum"], copy=True)
    for length in FORBIDDEN_HAIRPIN_LOOPS:
        if length < len(hairpin):
            hairpin[length] = cpu.NEG
    out["hp_cum"] = hairpin
    return out


@dataclass
class ContrafoldMFESource:
    """A :class:`~cas13guide.services.arc_model.MFESource` over CONTRAfold.

    ``sharpturn=False`` is the default because it is LinearFold's default and
    the Arc model was trained against it. Pass ``sharpturn=True`` only to
    reproduce a plain CONTRAfold run.
    """

    sharpturn: bool = False
    _params: dict | None = None

    version = "contrafold/gpu-contrafold/viterbi/no-sharp-turn/v1"

    def __post_init__(self):
        cpu = _cpu()
        params = cpu.load()
        self._params = params if self.sharpturn else no_sharp_turn(params)
        if self.sharpturn:
            self.version = "contrafold/gpu-contrafold/viterbi/sharp-turn-allowed/v1"

    # -- internals ------------------------------------------------------
    def _score(self, sequence: str, forced: list[int] | None = None) -> float:
        from ._contrafold_kernel import viterbi_scored

        cpu = _cpu()
        rna = sequence.upper().replace("T", "U")
        encoded, length = cpu.encode(rna)
        mask = np.zeros(length + 2, dtype=np.int64)
        if forced is not None:
            for index in range(length):
                mask[index + 1] = int(forced[index])
        p = self._params
        _pair, score = viterbi_scored(
            encoded, length, mask,
            p["bp"], p["stack"], p["tm"], p["hc"], p["dl"], p["dr"],
            p["hp_cum"], p["cs"], p["i11"], p["b01"],
            p["mb"], p["mu"], p["mp"], p["eu"], p["ep"], cpu._CANON,
        )
        return round(float(score), SCORE_DECIMALS)

    # -- MFESource ------------------------------------------------------
    def mfe(self, sequence: str) -> float:
        """Unconstrained CONTRAfold score. Higher = more stable, not kcal/mol."""
        return self._score(sequence)

    def mfe_constrained(self, sequence: str, constraint: str) -> float:
        """Score with every ``.`` in ``constraint`` forced unpaired.

        The dot-bracket convention here is the one ``arc_model`` builds: ``.``
        means "must not pair" and ``?`` means "unconstrained". That is narrower
        than full dot-bracket, and deliberately so -- it is the only form the
        Arc feature needs, and accepting more would imply support for pairing
        constraints this does not implement.
        """
        if len(constraint) != len(sequence):
            raise ValueError(
                f"constraint is {len(constraint)} long, sequence is "
                f"{len(sequence)}; they must match position for position"
            )
        forced = [1 if character == "." else 0 for character in constraint]
        return self._score(sequence, forced)
