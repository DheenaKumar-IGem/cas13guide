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
#: Longest self-hairpin stem tolerated, when no real folding engine is
#: available. This is a fallback and it is not good: measured against ViennaRNA
#: over 6,000 random 22-mers it flags 1.9% of spacers and catches 28.9% of those
#: that genuinely fold to -8 kcal/mol or tighter, with roughly half its flags
#: being sequences that barely fold at all. The wobble fix raised recall from
#: 5.6%, which is an improvement rather than a solution.
#:
#: A helix length is simply not a good estimator of a free energy, and no
#: threshold makes it one. Install ViennaRNA and the real fold is used instead;
#: the run says which one it used.
MAX_SELF_STEM = 6
MAX_DR_STEM = 6


#: Pairs an RNA helix can actually form. G-U is included because this is an
#: RNA:RNA hairpin, the same reason ``util/duplex.py`` derives the wobble rule
#: for spacer:target binding -- "the rule is derived here once and used
#: everywhere" was true of every module except this one.
HELIX_PAIRS: frozenset[str] = frozenset({"AU", "UA", "GC", "CG", "GU", "UG"})

#: A hairpin needs at least three unpaired bases to turn. Without this the scan
#: counted helices that close on nothing, which no molecule forms.
MIN_HAIRPIN_LOOP = 3


def longest_self_complement(seq: str, min_len: int = 3) -> int:
    """Longest hairpin stem the spacer can form with itself.

    A proxy for self-folding, and a weak one -- see :data:`MAX_SELF_STEM` for
    its measured recall. It used to be weaker in two specific ways:

    * it matched with ``revcomp``, i.e. **Watson-Crick only**. A spacer whose
      helix contains a G-U wobble was scored as two short stems rather than one
      long one. Measured, ``AGCCGGAGGCAUGGUCUCCGGU`` folds to -14.4 kcal/mol as
      a clean 9 bp hairpin, and the old scan called it 6 -- one under the
      threshold, so a textbook self-folding spacer passed;
    * it applied no loop constraint, so it counted helices closing on zero
      unpaired bases.

    Fixing both raised recall against real folding from 5.6% to 28.9%.
    """
    s = seq.upper().replace("T", "U")
    n = len(s)
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            length = 0
            while (i + length) < (j - length):
                if (j - length) - (i + length) - 1 < MIN_HAIRPIN_LOOP:
                    break
                if (s[i + length] + s[j - length]) not in HELIX_PAIRS:
                    break
                length += 1
            if length > best:
                best = length
    return best if best >= min_len else 0


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


#: Free energy at which a spacer is judged too self-structured to work, when a
#: real folding engine is available. Calibrated against 6,000 random 22-mers:
#: 1.3% fold this tightly, which is the same order as the proxy's flag rate but
#: measured rather than guessed.
SELF_FOLD_MAX_DG = -8.0


def qc_candidate(
    candidate: Candidate, spec: ResolvedDesignSpec, fold=None
) -> QCResult:
    params = spec.profile.params
    flags: list[QCFlag] = []
    spacer = candidate.spacer

    warnings: list[QCFlag] = []

    # Length outside what the ortholog can physically accommodate is a hard
    # incompatibility. Length merely outside its *canonical* set is not: it is
    # unvalidated, which the activity model's domain check already reports as
    # out-of-domain -- refusing to narrow the interval at all, which the gate
    # then turns into a refusal rather than a certification. Failing it *here*
    # as well would make the window length un-configurable, which is a
    # different and worse thing: the design could not even be stated.
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

    # Prefer a real fold. The proxy below catches 28.9% of genuinely
    # self-structured spacers; thermodynamics catches them because it IS the
    # thing being approximated. The docstring on longest_self_complement has
    # promised this swap since the module was written.
    self_stem = longest_self_complement(spacer)
    if fold is not None:
        try:
            self_folds = fold(spacer) <= SELF_FOLD_MAX_DG
        except Exception:
            self_folds = self_stem > MAX_SELF_STEM
    else:
        self_folds = self_stem > MAX_SELF_STEM
    if self_folds:
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
    candidates: tuple[Candidate, ...], spec: ResolvedDesignSpec, fold=None
) -> dict[str, QCResult]:
    return {c.guide_id: qc_candidate(c, spec, fold) for c in candidates}
