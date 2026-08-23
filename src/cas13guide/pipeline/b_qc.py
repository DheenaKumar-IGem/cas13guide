"""Stage B -- deterministic QC.

Cheap, hard filters that need no model and no search: effector compatibility,
composition, self-structure, direct-repeat interaction and synthesis
constraints. A QC failure is a *deterministic incompatibility*, which the gate
treats as FAIL -- no amount of user preference weighting buys it back.
"""

from __future__ import annotations

from ..contract.enums import QCFlag
from ..contract.spec import ResolvedDesignSpec
from ..models.evidence import Candidate, QCResult
from ..util.seq import gc_fraction, has_polyu_terminator, max_homopolymer, revcomp

GC_MIN, GC_MAX = 0.30, 0.70
MAX_HOMOPOLYMER = 4
MAX_SELF_STEM = 6
MAX_DR_STEM = 6


def longest_self_complement(seq: str, min_len: int = 3) -> int:
    """Longest stem the spacer can form with itself.

    Proxy for spacer self-folding: the longest subsequence whose reverse
    complement also occurs elsewhere in the spacer. A real implementation
    swaps in a folding engine (RNAfold/LinearFold) behind the same call.
    """
    best = 0
    n = len(seq)
    for size in range(min_len, n // 2 + 1):
        for i in range(n - size + 1):
            stem = seq[i : i + size]
            rc = revcomp(stem)
            j = seq.find(rc)
            while j != -1:
                if j >= i + size or j + size <= i:  # non-overlapping
                    best = max(best, size)
                    break
                j = seq.find(rc, j + 1)
    return best


def longest_cross_complement(a: str, b: str, min_len: int = 3) -> int:
    """Longest duplex the two sequences can form with each other."""
    best = 0
    for size in range(min_len, min(len(a), len(b)) + 1):
        found = False
        for i in range(len(a) - size + 1):
            if revcomp(a[i : i + size]) in b:
                best, found = size, True
                break
        if not found:
            break
    return best


def qc_candidate(candidate: Candidate, spec: ResolvedDesignSpec) -> QCResult:
    params = spec.profile.params
    flags: list[QCFlag] = []
    spacer = candidate.spacer

    warnings: list[QCFlag] = []

    # Length outside what the ortholog can physically accommodate is a hard
    # incompatibility. Length merely outside its *canonical* set is not: it is
    # unvalidated, which the activity model's domain check already reports as
    # out-of-domain so the bounds widen and the gate raises an R3 REVIEW.
    # Failing it here as well made the window length un-configurable and, for a
    # 22-24 nt mature miRNA against 28 nt LwaCas13a spacers, made the target
    # undesignable rather than merely unusual.
    if not params.supports_length(candidate.length):
        flags.append(QCFlag.SPACER_LENGTH_UNSUPPORTED)
    elif not params.is_canonical_length(candidate.length):
        warnings.append(QCFlag.SPACER_LENGTH_NONCANONICAL)

    # Hard PFS requirement (e.g. PspCas13b's 5' constraint) fails the guide.
    if not params.pfs.check(candidate.five_prime_flank, candidate.three_prime_flank):
        flags.append(QCFlag.PFS_INCOMPATIBLE)

    # Soft PFS preference (e.g. the LwaCas13a 3' H bias) only deprioritizes.
    # Treating it as a hard exclusion deleted every G-flanked candidate from
    # the design, which is not what a reported base preference supports.
    pfs_preference = params.pfs.preference(
        candidate.five_prime_flank, candidate.three_prime_flank
    )
    if pfs_preference < 1.0:
        warnings.append(QCFlag.PFS_DISFAVOURED)

    gc = gc_fraction(spacer)
    if not (GC_MIN <= gc <= GC_MAX):
        flags.append(QCFlag.GC_OUT_OF_RANGE)

    homo = max_homopolymer(spacer)
    if homo > MAX_HOMOPOLYMER:
        flags.append(QCFlag.HOMOPOLYMER_RUN)

    # U-runs terminate Pol III transcription of the guide cassette.
    if has_polyu_terminator(spacer):
        flags.append(QCFlag.POLY_U_TERMINATOR)

    self_stem = longest_self_complement(spacer)
    if self_stem > MAX_SELF_STEM:
        flags.append(QCFlag.SPACER_SELF_FOLDING)

    dr_stem = longest_cross_complement(spacer, params.direct_repeat)
    if dr_stem > MAX_DR_STEM:
        flags.append(QCFlag.DR_SPACER_INTERACTION)

    # Manufacturability is a soft score; hard failures are the flags above.
    penalty = 0.08 * max(0, homo - 2) + 0.06 * max(0, self_stem - 3)
    penalty += 0.5 * abs(gc - 0.5)
    manufacturability = max(0.0, 1.0 - penalty)

    return QCResult(
        guide_id=candidate.guide_id,
        passed=not flags,
        flags=tuple(flags),
        gc=gc,
        max_homopolymer=homo,
        self_fold_score=float(self_stem),
        manufacturability=manufacturability,
        warnings=tuple(warnings),
        pfs_preference=pfs_preference,
    )


def run_qc(
    candidates: tuple[Candidate, ...], spec: ResolvedDesignSpec
) -> dict[str, QCResult]:
    return {c.guide_id: qc_candidate(c, spec) for c in candidates}
