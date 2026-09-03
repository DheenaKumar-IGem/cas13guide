"""Pick the search index that fits the corpus and the workload.

Both indexes are provably exhaustive and return identical hit sets, so this is
a pure performance decision -- but it is a real one: the FM-index is markedly
*slower* than the reference scanner on anything but a very large corpus, and
its suffix array costs a fixed amount to build before any query runs.

The decision uses an explicit cost model rather than a bare size threshold,
because the right answer depends on query volume as much as corpus size.
Measured on this machine (random RNA in 2 kb loci, 22-mers, ``k=4``, SA-IS
build):

==========  ==========  ==========  ==========  ===================
corpus      build       FM/query    scan/query  queries to amortize
==========  ==========  ==========  ==========  ===================
2 kb        0.006 s     0.027 s     0.0002 s    never
20 kb       0.040 s     0.163 s     0.0008 s    never
200 kb      0.585 s     0.858 s     0.0065 s    never
600 kb      1.16 s      0.742 s     0.018 s     never
1.5 Mb      5.83 s      2.990 s     0.054 s     never
3 Mb        12.56 s     2.758 s     0.089 s     never
==========  ==========  ==========  ==========  ===================

Scan time is linear in corpus size (~0.032 s/Mb/query). Build is roughly
linear once the corpus is past a megabase. Those two facts are the model
below, and they hold everywhere.

The third is narrower than it used to be stated. FM query time saturates
toward a ceiling set by the mismatch neighbourhood rather than the corpus --
**above about 1.5 Mb**. Below that it tracks the corpus, by two orders of
magnitude across the first three rows: backtracking exhausts a suffix interval
early when there is little to match against, so the neighbourhood bound is
never reached. The first three rows were added when a reader noticed the model
printing "FM ~2800s" for the shipped demo's 2,391 bases, where the measured
cost is about 27 seconds.

The flat constant is still what the *decision* uses, and that is safe: the
crossover sits near 87 Mb, deep inside the regime where it was measured, and
below it the scanner wins under either figure -- at 600 kb it is 0.018 s/query
against a real FM 0.742. What the floor changed is what gets *reported*; see
:data:`FM_FLAT_ABOVE_BASES`.

Where the crossover went
------------------------
It moved, a lot. These numbers replace an earlier set measured when the
scanner compared windows one character at a time in Python; vectorizing it
(see :mod:`cas13guide.services.indexes`) made it **32x faster on the same
machine** for the same hit set -- 0.678 -> 0.021 s/Mb/query at 3 Mb -- and the
crossover moved with it, from ~2.5 Mb to ~87 Mb of searchable sequence.

The practical consequence is that the FM-index is now the right choice only at
whole-transcriptome scale, and the stale constants were actively harmful:
they made ``DesignContext.auto`` build a suffix array at 3 Mb, where the
scanner is roughly 30x faster per query and free to build. The seam still
earns its keep -- a full GENCODE transcript set is a few hundred megabases,
comfortably past the crossover -- but the shipped default is now the scanner
for anything smaller.

The constants are machine-specific. Re-measure them with
``examples/bench_index.py`` if this runs somewhere materially different; the
model shape holds, the numbers move.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contract.enums import Universe
from .fmindex import FMIndexSearchIndex
from .indexes import ExhaustiveScanIndex, SearchIndex
from .transcriptome import Transcriptome

#: Seconds to build the suffix array, per megabase. Taken from the 3 Mb point;
#: the ratio is lower on sub-megabase corpora, but those never choose the FM
#: index anyway, so the large-corpus asymptote is the number that matters.
BUILD_SECONDS_PER_MB = 4.2

#: Seconds for one scanner query, per megabase of corpus.
SCAN_SECONDS_PER_MB_PER_QUERY = 0.032

#: Seconds for one FM backtracking query, **in the flat regime**. Governed by
#: the mismatch neighbourhood (``C(L,k) * 3^k``) once the corpus is large
#: enough that backtracking rarely exhausts a suffix interval early.
FM_SECONDS_PER_QUERY = 2.8

#: Corpus size above which :data:`FM_SECONDS_PER_QUERY` is calibrated.
#:
#: "Effectively flat in corpus size" is true above roughly this point and false
#: below it -- the table in this module's docstring says so itself, with 0.742
#: s/query at 600 kb against 2.758 at 3 Mb. Measured further down on the same
#: machine: 0.027 s/query at 2 kb, 0.163 at 20 kb, 0.858 at 200 kb. So the
#: flat constant overstates the FM cost on a small corpus by up to ~100x.
#:
#: That does not move the decision, and the direction is the safe one: the
#: crossover sits near 87 Mb, deep inside the flat regime, and below it the
#: scanner wins under either figure -- at 600 kb it is 0.019 s/query against a
#: real FM 0.742. The model is left alone for that reason.
#:
#: What it did move was the *reported* number. `plan_index` printed the
#: extrapolated figure as though it were a measurement: at the shipped demo's
#: 2,391 bases it said "FM ~2800s" where the real cost is ~27s. That string is
#: provenance -- it reaches stdout, the HTML report and the ledger's
#: `index_choice` -- so below this floor the rationale now states why the
#: scanner wins without quoting a counterfactual it cannot support.
FM_FLAT_ABOVE_BASES = 1_500_000

#: Retained for callers that want the headline number: the corpus size at
#: which per-query costs cross over, ignoring build. Derived, not independent.
CROSSOVER_BASES = int(FM_SECONDS_PER_QUERY / SCAN_SECONDS_PER_MB_PER_QUERY * 1_000_000)

SCANNER = "scan"
FM = "fm"


@dataclass(frozen=True, slots=True)
class IndexPlan:
    """The decision, before anything expensive happens."""

    kind: str
    searchable_bases: int
    rationale: str


@dataclass(frozen=True, slots=True)
class IndexChoice:
    """The decision plus the built index."""

    index: SearchIndex
    plan: IndexPlan

    @property
    def kind(self) -> str:
        return self.plan.kind

    @property
    def searchable_bases(self) -> int:
        return self.plan.searchable_bases

    @property
    def rationale(self) -> str:
        return self.plan.rationale


def searchable_bases(transcriptome: Transcriptome, universes: frozenset[Universe]) -> int:
    return sum(len(locus.sequence) for locus in transcriptome.in_universes(universes))


def estimate_seconds(total_bases: int, expected_queries: int) -> tuple[float, float]:
    """``(scanner_total, fm_total)`` estimated wall-clock for this workload."""
    mb = total_bases / 1_000_000
    scanner = expected_queries * SCAN_SECONDS_PER_MB_PER_QUERY * mb
    fm = BUILD_SECONDS_PER_MB * mb + expected_queries * FM_SECONDS_PER_QUERY
    return scanner, fm


def plan_index(total_bases: int, expected_queries: int = 1000) -> IndexPlan:
    """Decide which index to build. Pure, cheap, and independently testable.

    Kept separate from construction so the policy can be exercised without
    paying for a suffix array -- building one over a multi-megabyte corpus
    takes seconds, which does not belong in a unit test.
    """
    if total_bases <= 0 or expected_queries <= 0:
        return IndexPlan(
            kind=SCANNER,
            searchable_bases=max(0, total_bases),
            rationale="nothing to search; the scanner costs nothing to build",
        )

    scanner_seconds, fm_seconds = estimate_seconds(total_bases, expected_queries)

    if fm_seconds < scanner_seconds:
        return IndexPlan(
            kind=FM,
            searchable_bases=total_bases,
            rationale=(
                f"{total_bases:,} bases x {expected_queries} queries: FM "
                f"~{fm_seconds:.0f}s beats scan ~{scanner_seconds:.0f}s "
                "(build amortizes)"
            ),
        )

    if total_bases < FM_FLAT_ABOVE_BASES:
        # Below the calibration floor the FM figure is an extrapolation, and a
        # wrong one -- see FM_FLAT_ABOVE_BASES. Say why the scanner wins
        # instead of quoting a number this model cannot stand behind.
        return IndexPlan(
            kind=SCANNER,
            searchable_bases=total_bases,
            rationale=(
                f"{total_bases:,} bases x {expected_queries} queries: scan "
                f"~{scanner_seconds:.0f}s, and the suffix-array build does not "
                f"pay back (FM cost is only modelled above "
                f"{FM_FLAT_ABOVE_BASES / 1_000_000:g} Mb)"
            ),
        )

    return IndexPlan(
        kind=SCANNER,
        searchable_bases=total_bases,
        rationale=(
            f"{total_bases:,} bases x {expected_queries} queries: scan "
            f"~{scanner_seconds:.0f}s beats FM ~{fm_seconds:.0f}s "
            "(suffix-array build does not pay back)"
        ),
    )


def choose_index(
    transcriptome: Transcriptome,
    universes: frozenset[Universe],
    expected_queries: int = 1000,
) -> IndexChoice:
    """Plan, then build. The rationale travels with the index.

    It is returned rather than logged so the orchestrator can put it in the
    provenance ledger: which index ran is part of reproducing a result, even
    though it cannot change one.
    """
    plan = plan_index(searchable_bases(transcriptome, universes), expected_queries)
    index: SearchIndex = (
        FMIndexSearchIndex(transcriptome, universes)
        if plan.kind == FM
        else ExhaustiveScanIndex(transcriptome, universes)
    )
    return IndexChoice(index=index, plan=plan)
