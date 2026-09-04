"""Spacer:target duplex algebra.

Why this module exists
----------------------
Cas13 guides bind target RNA as an RNA:RNA duplex, so G-U wobble pairs are
tolerated where a DNA-targeting nuclease would read a mismatch. Getting that
wrong silently inflates apparent specificity, so the rule is derived here once
and used everywhere.

Derivation
~~~~~~~~~~
Let ``s`` be the spacer and ``t`` an observed target site, binding
antiparallel, so ``s[j]`` pairs with ``t[L-1-j]``. Let ``p = revcomp(s)`` be
the *perfect* target site; then ``p[L-1-j] == complement(s[j])``.

* Watson-Crick at duplex position ``j``  <=>  ``t[L-1-j] == p[L-1-j]``.
  So a plain positional comparison of observed-vs-expected target site already
  gives the WC matches.
* Wobble needs ``(s[j], t[L-1-j])`` in ``{(G,U), (U,G)}``:
    - ``s[j] == G``  =>  expected base ``p[L-1-j] == C``, observed ``U``.
    - ``s[j] == U``  =>  expected base ``p[L-1-j] == A``, observed ``G``.

So in *target space*, exactly two substitutions preserve pairing:
``C -> U`` and ``A -> G``. Every other substitution is a true mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .seq import revcomp

#: Positional weight of a mismatch at either spacer terminus.
TERMINAL_WEIGHT = 0.25

#: The seed region, as a fraction of spacer length. Inside it a mismatch
#: carries full weight. Declared once here because ``models/features.py``
#: derives its ``gc_seed`` window from the same span -- retuning the seed in
#: one place while the other kept its own literal would leave the model's
#: features describing a different seed than its penalty.
SEED_SPAN: tuple[float, float] = (0.35, 0.75)

#: expected_base -> observed_base substitutions that still pair (G-U wobble).
WOBBLE_SUBSTITUTIONS: frozenset[tuple[str, str]] = frozenset({("C", "U"), ("A", "G")})


class PairKind(str, Enum):
    WATSON_CRICK = "WC"
    WOBBLE = "GU"
    MISMATCH = "MM"


def classify_pair(expected_base: str, observed_base: str) -> PairKind:
    if expected_base == observed_base:
        return PairKind.WATSON_CRICK
    if (expected_base, observed_base) in WOBBLE_SUBSTITUTIONS:
        return PairKind.WOBBLE
    return PairKind.MISMATCH


@dataclass(frozen=True, slots=True)
class DuplexProfile:
    """Positional description of one spacer:target-site duplex.

    Positions are reported in *spacer* coordinates (1-based, 5'->3' along the
    spacer), because that is the frame published Cas13 seed effects are
    described in.
    """

    length: int
    mismatch_positions: tuple[int, ...]
    wobble_positions: tuple[int, ...]
    cigar: str

    @property
    def n_mismatch(self) -> int:
        return len(self.mismatch_positions)

    @property
    def n_wobble(self) -> int:
        return len(self.wobble_positions)

    @property
    def is_perfect(self) -> bool:
        return not self.mismatch_positions and not self.wobble_positions


def profile_duplex(spacer: str, target_site: str) -> DuplexProfile:
    """Compare an observed ``target_site`` to what ``spacer`` would pair with.

    Both arguments are 5'->3' and equal length.
    """
    if len(spacer) != len(target_site):
        raise ValueError("spacer and target_site must be the same length")

    expected = revcomp(spacer)  # the perfect target site
    length = len(spacer)

    if length == 0:
        # An empty duplex is vacuously perfect. Returning it beats the
        # ``IndexError: list index out of range`` the CIGAR builder used to
        # raise from ``ordered[0]``: nothing about that traceback says
        # "you passed an empty sequence".
        return DuplexProfile(
            length=0, mismatch_positions=(), wobble_positions=(), cigar=""
        )

    mismatches: list[int] = []
    wobbles: list[int] = []
    cigar_parts: list[str] = []

    # Walk target coordinates, convert to spacer coordinates for reporting.
    for k, (exp_b, obs_b) in enumerate(zip(expected, target_site)):
        kind = classify_pair(exp_b, obs_b)
        spacer_pos = length - k  # 1-based spacer position
        if kind is PairKind.MISMATCH:
            mismatches.append(spacer_pos)
        elif kind is PairKind.WOBBLE:
            wobbles.append(spacer_pos)
        cigar_parts.append(kind.value)

    # Compact run-length CIGAR in spacer 5'->3' order.
    ordered = list(reversed(cigar_parts))
    cigar: list[str] = []
    run_kind, run_len = ordered[0], 1
    for symbol in ordered[1:]:
        if symbol == run_kind:
            run_len += 1
        else:
            cigar.append(f"{run_len}{run_kind}")
            run_kind, run_len = symbol, 1
    cigar.append(f"{run_len}{run_kind}")

    return DuplexProfile(
        length=length,
        mismatch_positions=tuple(sorted(mismatches)),
        wobble_positions=tuple(sorted(wobbles)),
        cigar="".join(cigar),
    )


def seed_weight(spacer_pos: int, length: int, seed_span: tuple[float, float] = SEED_SPAN) -> float:
    """Positional disruption weight for a mismatch at ``spacer_pos``.

    Cas13 tolerates terminal mismatches far better than central ones: the
    central region of the spacer behaves as a seed. ``seed_span`` gives the
    seed as a fraction of spacer length; inside it a mismatch carries full
    weight, outside it the weight tapers toward the termini.
    """
    frac = (spacer_pos - 0.5) / length
    lo, hi = seed_span
    if lo <= frac <= hi:
        return 1.0
    # Linear taper to TERMINAL_WEIGHT at each terminus. Each flank is
    # normalized by *its own* width, so both bottom out at TERMINAL_WEIGHT at
    # their own end of the spacer; normalizing both by the wider flank would
    # leave the narrower one short of it for no stated reason.
    #
    # That is symmetry in the *endpoints*, not in the positions. This comment
    # used to claim a mismatch at the last spacer position is penalized
    # exactly as much as one at the first, which is false: SEED_SPAN is not
    # centred, so the flanks are 0.35 and 0.25 wide, and the first and last
    # positions sit the same distance in from each end but are divided by
    # different widths. On a 28-mer they weigh 0.2883 and 0.3036.
    if frac < lo:
        travelled = (lo - frac) / lo if lo > 0 else 1.0
    else:
        travelled = (frac - hi) / (1.0 - hi) if hi < 1.0 else 1.0
    travelled = min(1.0, max(0.0, travelled))
    return 1.0 - (1.0 - TERMINAL_WEIGHT) * travelled


def duplex_penalty(profile: DuplexProfile, wobble_cost: float = 0.35) -> float:
    """Total position-weighted disruption. 0.0 == perfect duplex.

    Used only as a *feature* for the activity model and as a cheap ordering
    key during search. It is never itself a specificity score.
    """
    total = 0.0
    for pos in profile.mismatch_positions:
        total += seed_weight(pos, profile.length)
    for pos in profile.wobble_positions:
        total += wobble_cost * seed_weight(pos, profile.length)
    return total
