"""Per-ortholog constants: spacer geometry, direct repeat, PFS policy.

Anything here bounds what the models are allowed to be asked about; the
registry cross-checks its supported domain against these values.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Effector


@dataclass(frozen=True, slots=True)
class PFSPolicy:
    """Protospacer flanking site constraint, in target-RNA coordinates.

    ``five_prime_allowed`` / ``three_prime_allowed`` are sets of permitted
    bases immediately flanking the target site. Empty set == unconstrained.
    """

    five_prime_allowed: frozenset[str] = frozenset()
    three_prime_allowed: frozenset[str] = frozenset()

    def check(self, five_prime: str | None, three_prime: str | None) -> bool:
        if self.five_prime_allowed and five_prime and five_prime not in self.five_prime_allowed:
            return False
        if self.three_prime_allowed and three_prime and three_prime not in self.three_prime_allowed:
            return False
        return True


@dataclass(frozen=True, slots=True)
class EffectorParams:
    effector: Effector
    direct_repeat: str
    spacer_lengths: tuple[int, ...]
    preferred_spacer_length: int
    pfs: PFSPolicy
    #: Collateral trans-cleavage is an intrinsic property of activation for
    #: most Cas13s; in vitro detection assays exploit it, cell assays suffer it.
    collateral_active: bool


EFFECTORS: dict[Effector, EffectorParams] = {
    # Cas13d has no reported PFS requirement.
    Effector.RFX_CAS13D: EffectorParams(
        effector=Effector.RFX_CAS13D,
        direct_repeat="CAACUAAACCCUAUAAGAAACGAAAUGAGAACCCAAUUCAUCGCU",
        spacer_lengths=(22, 23, 30),
        preferred_spacer_length=22,
        pfs=PFSPolicy(),
        collateral_active=True,
    ),
    # LwaCas13a prefers a non-G 3' PFS.
    Effector.LWA_CAS13A: EffectorParams(
        effector=Effector.LWA_CAS13A,
        direct_repeat="GAUUUAGACUACCCCAAAAACGAAGGGGACUAAAAC",
        spacer_lengths=(28,),
        preferred_spacer_length=28,
        pfs=PFSPolicy(three_prime_allowed=frozenset("AUC")),
        collateral_active=True,
    ),
    # PspCas13b prefers a non-C 5' PFS.
    Effector.PSP_CAS13B: EffectorParams(
        effector=Effector.PSP_CAS13B,
        direct_repeat="GUUGUGGAAGGUCCAGUUUUGAGGGGCUAUUACAAC",
        spacer_lengths=(30,),
        preferred_spacer_length=30,
        pfs=PFSPolicy(five_prime_allowed=frozenset("AUG")),
        collateral_active=True,
    ),
}
