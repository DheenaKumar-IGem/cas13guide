"""Per-ortholog constants: spacer geometry, direct repeat, PFS policy.

Anything here bounds what the models are allowed to be asked about; the
registry derives its supported spacer lengths from :data:`EFFECTORS` so the two
cannot drift.

Two distinctions this module is careful about, because collapsing either one
produces a tool that refuses designs it should merely discourage:

* **Canonical length vs supported length.** ``spacer_lengths`` are the lengths
  the published constructs use and the lengths any bundled model was validated
  on. ``spacer_length_range`` is the wider span the ortholog will actually
  tolerate. Asking for a length inside the range but outside the canonical set
  is allowed, and the activity model's domain check flags it out-of-domain, so
  the calibration widens and the gate raises an R3 REVIEW rather than either
  refusing outright or answering with unearned confidence.
* **Required PFS vs preferred PFS** (FR2.3). A *required* flanking base is a hard
  incompatibility -- the guide cannot work. A *preferred* one is a bias: it
  lowers expected activity and should push a candidate down the ranking, not
  out of the design. The reported LwaCas13a 3' H preference is the second kind,
  and treating it as the first silently deleted a third of every candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Effector

#: Multiplier applied to a candidate's PFS preference score when the observed
#: flanking base is outside the preferred set. Not zero: a disfavoured PFS is a
#: real but surmountable penalty, and zero would make it a hard exclusion by
#: another route.
DISFAVOURED_PFS_FACTOR = 0.55


@dataclass(frozen=True, slots=True)
class PFSPolicy:
    """Protospacer flanking site constraint, in target-RNA coordinates.

    ``*_allowed`` are **hard** requirements: a base outside the set makes the
    guide deterministically incompatible and it fails QC. ``*_preferred`` are
    **soft**: a base outside the set is scored down but stays in the design.
    Empty set == unconstrained on that axis.
    """

    five_prime_allowed: frozenset[str] = frozenset()
    three_prime_allowed: frozenset[str] = frozenset()
    five_prime_preferred: frozenset[str] = frozenset()
    three_prime_preferred: frozenset[str] = frozenset()

    def check(self, five_prime: str | None, three_prime: str | None) -> bool:
        """Hard compatibility only. Preferences never fail this."""
        if self.five_prime_allowed and five_prime and five_prime not in self.five_prime_allowed:
            return False
        if self.three_prime_allowed and three_prime and three_prime not in self.three_prime_allowed:
            return False
        return True

    def preference(self, five_prime: str | None, three_prime: str | None) -> float:
        """Soft PFS score in ``(0, 1]``. 1.0 means every preference is met.

        A flank of ``None`` -- the target site sits at a transcript end, so
        there is no flanking base to read -- is not counted against the
        candidate. Absence of evidence is not a disfavoured base.
        """
        score = 1.0
        if self.five_prime_preferred and five_prime and five_prime not in self.five_prime_preferred:
            score *= DISFAVOURED_PFS_FACTOR
        if self.three_prime_preferred and three_prime and three_prime not in self.three_prime_preferred:
            score *= DISFAVOURED_PFS_FACTOR
        return score

    def is_preferred(self, five_prime: str | None, three_prime: str | None) -> bool:
        return self.preference(five_prime, three_prime) == 1.0


@dataclass(frozen=True, slots=True)
class EffectorParams:
    effector: Effector
    direct_repeat: str
    #: Canonical, published spacer lengths. Models are validated on these.
    spacer_lengths: tuple[int, ...]
    preferred_spacer_length: int
    pfs: PFSPolicy
    #: Collateral trans-cleavage is an intrinsic property of activation for
    #: most Cas13s; in vitro detection assays exploit it, cell assays suffer it.
    collateral_active: bool
    #: Inclusive ``(min, max)`` spacer length the ortholog tolerates. Wider than
    #: ``spacer_lengths``; lengths in here but not there are usable and are
    #: reported out-of-domain rather than rejected.
    spacer_length_range: tuple[int, int] = (20, 30)

    def supports_length(self, length: int) -> bool:
        low, high = self.spacer_length_range
        return low <= length <= high

    def is_canonical_length(self, length: int) -> bool:
        return length in self.spacer_lengths


EFFECTORS: dict[Effector, EffectorParams] = {
    # Cas13d has no reported PFS requirement, hard or soft.
    Effector.RFX_CAS13D: EffectorParams(
        effector=Effector.RFX_CAS13D,
        direct_repeat="CAACUAAACCCUAUAAGAAACGAAAUGAGAACCCAAUUCAUCGCU",
        spacer_lengths=(22, 23, 30),
        preferred_spacer_length=22,
        pfs=PFSPolicy(),
        collateral_active=True,
        spacer_length_range=(20, 30),
    ),
    # LwaCas13a prefers a non-G 3' PFS ("H"). A *preference*, not a
    # requirement: the original hard constraint failed roughly a quarter of all
    # candidates outright, which is not what the reported bias supports.
    Effector.LWA_CAS13A: EffectorParams(
        effector=Effector.LWA_CAS13A,
        direct_repeat="GAUUUAGACUACCCCAAAAACGAAGGGGACUAAAAC",
        spacer_lengths=(28,),
        preferred_spacer_length=28,
        pfs=PFSPolicy(three_prime_preferred=frozenset("AUC")),
        collateral_active=True,
        spacer_length_range=(20, 28),
    ),
    # PspCas13b prefers a non-C 5' PFS. Left as a hard constraint: unlike the
    # LwaCas13a case there is no requirement here asking for it to be relaxed,
    # and softening it would be a scientific call, not a mechanical one.
    Effector.PSP_CAS13B: EffectorParams(
        effector=Effector.PSP_CAS13B,
        direct_repeat="GUUGUGGAAGGUCCAGUUUUGAGGGGCUAUUACAAC",
        spacer_lengths=(30,),
        preferred_spacer_length=30,
        pfs=PFSPolicy(five_prime_allowed=frozenset("AUG")),
        collateral_active=True,
        spacer_length_range=(22, 30),
    ),
}
