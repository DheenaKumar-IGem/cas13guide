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
from ..util.seq import gc_fraction, has_polyu_terminator, max_homopolymer

GC_MIN, GC_MAX = 0.30, 0.70
MAX_HOMOPOLYMER = 4
#: Longest self-hairpin stem tolerated, when no real folding engine is
#: available. This is a fallback and it is not good: measured against ViennaRNA
#: over 60,000 random 22-mers (seed 12345) it flags 1.77% of spacers and catches
#: 34.9% +/- 3.7% of those that genuinely fold to -8 kcal/mol or tighter, while
#: only 21% of its flags are sequences that really do fold -- four in five are
#: false. The wobble fix raised recall from 5.6%, which is an improvement rather
#: than a solution.
#:
#: Those figures were 1.9%, 28.9% and "roughly half its flags" on a 6,000-mer
#: sample, which puts ~53 genuine folders behind the recall estimate and a 95%
#: interval about +/-11% wide. The recall number was not wrong so much as noisy;
#: the precision one was materially optimistic. Re-measured at ten times the
#: sample, and pinned by ``tests/test_measured_claims.py`` so it cannot drift
#: again unnoticed.
#:
#: A helix length is simply not a good estimator of a free energy, and no
#: threshold makes it one. Install ViennaRNA and the real fold is used instead;
#: the run says which one it used.
MAX_SELF_STEM = 6

#: Longest spacer:DR helix tolerated before :data:`QCFlag.DR_SPACER_INTERACTION`
#: fires. Nine, not six, and the two numbers describe the same strictness under
#: two different pairing rules.
#:
#: This threshold and :func:`longest_cross_complement` are one calibration, not
#: two constants. Six was derived when the scan was Watson-Crick only, where it
#: flagged 2.1-2.3% of random 22-mers against all three shipped repeats -- a
#: rate low enough that a flag meant something specific about the spacer.
#: Teaching the scan G-U wobbles (which it had to learn: a real 12 bp helix
#: containing one wobble was being measured as 6, exactly at the old bar, so it
#: passed) roughly doubles the helix a random spacer reaches by chance. Left at
#: six, the same threshold flagged 14.7% of random spacers against the
#: LwaCas13a repeat and 47.5% against PspCas13b.
#:
#: That would have been the worse bug of the two. DR_SPACER_INTERACTION is a
#: *hard* flag -- the candidate is dropped, not ranked down -- so a rate like
#: that deletes a third of every candidate set for no reason, and the number
#: deleted depends mostly on which repeat the ortholog happens to have. It is
#: the failure ``contract/effectors.py`` describes at length about hard-vs-soft
#: PFS, arrived at from the other direction.
#:
#: Nine restores the original false-positive budget: 2.4% / 0.5% / 3.1% for
#: RfxCas13d / LwaCas13a / PspCas13b over 20,000 random 22-mers, seed 12345,
#: against 2.3% / 2.1% / 2.2% for the old rule at six. Detection is strictly
#: better than before rather than merely preserved -- every constructed spacer
#: carrying a 12 nt wobble-containing partner of its own repeat is still
#: flagged (19/19, 21/21, 24/24 across the three), and those are exactly the
#: ones the Watson-Crick scan could not see at any threshold.
#:
#: The spread across effectors is wider than it was (0.5% to 3.1%, against
#: 2.1% to 2.3%) because a wobble scan is sensitive to the repeat's G/U content
#: in a way a Watson-Crick scan is not. A per-effector threshold derived from
#: each repeat's own null would close that; a single number is what is here,
#: and this is where it is written down that the number is a rate, not a length
#: anyone measured in a tube.
MAX_DR_STEM = 9


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

    Fixing both raised recall against real folding from 5.6% to ~35%.
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
    """Longest duplex the two sequences can form with each other.

    Pairs by :data:`HELIX_PAIRS`, for the same reason
    :func:`longest_self_complement` does. This scan used to test
    ``revcomp(a[i:i+size]) in b`` -- a literal Watson-Crick substring match --
    so the two functions in this file disagreed about what a base pair is, and
    the one feeding the *hard* :data:`QCFlag.DR_SPACER_INTERACTION` flag was
    the stricter of the two. A single G-U wobble inside a helix split it into
    two shorter ones, and a spacer that really does sequester the direct repeat
    was measured as not doing so.

    That is the wrong direction for this particular check: the DR:spacer
    interaction is a *hazard*, so under-detecting it means shipping a guide
    whose scaffold is base-paired to its own spacer. The self-folding scan was
    fixed for exactly this; the fix landed on one of the two callers.

    No loop constraint here, unlike the hairpin scan: these are two separate
    strands, so nothing has to turn.
    """
    x = a.upper().replace("T", "U")
    y = b.upper().replace("T", "U")
    best = 0
    # Antiparallel: x[i + k] pairs with y[j - k], so the duplex runs 5'->3'
    # along x and 3'->5' along y.
    for i in range(len(x)):
        for j in range(len(y) - 1, -1, -1):
            length = 0
            while (i + length) < len(x) and (j - length) >= 0:
                if (x[i + length] + y[j - length]) not in HELIX_PAIRS:
                    break
                length += 1
            if length > best:
                best = length
    return best if best >= min_len else 0


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

    # An ambiguity code is not a sequence anyone can order, and every number
    # downstream treats it as known. `classify_pair` calls N:N a Watson-Crick
    # pair, so an N-bearing spacer scores a *perfect* on-target duplex and
    # lands in the mm0 calibration stratum; RNAplfold cannot pair an N, so it
    # reports unpaired probability exactly 1.0 there and accessibility -- which
    # carries weight 1.5 in the objective -- is inflated across every window
    # overlapping the gap. The result was that an assembly gap *attracted* the
    # ranking: measured on a 150 nt target with four Ns, ranks 1, 2, 3, 4 and 6
    # of 101 were all N-bearing, exported as orderable oligos with no warning.
    #
    # `N` stays a legal *input* character -- the target may genuinely be
    # unknown there -- and this refuses only to hand back a guide made of it.
    # It is the rule SYNTHESIS_CONSTRAINT was reserved for. Hard, not soft:
    # the sequence cannot be synthesised, so deprioritising it would still
    # leave it orderable from the export.
    if any(base not in "ACGU" for base in spacer):
        flags.append(QCFlag.SYNTHESIS_CONSTRAINT)

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

    # Prefer a real fold. The proxy below catches ~35% of genuinely
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
