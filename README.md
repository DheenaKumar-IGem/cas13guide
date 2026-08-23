# cas13guide

Cas13 target-RNA guide locator and scoring, built on calibrated evidence.

This is the implementation of design revision v2. The organising rule is that
**efficacy, specificity, intent and uncertainty are never collapsed into one
score before the safety gate and the Pareto pass** — and, unlike v1, that rule
lives in enforced contracts rather than in principles written on a diagram.

```
contract/    ResolvedDesignSpec — the question, stated precisely
services/    provenance · registry · calibration · context · indexes
             fmindex · suffixarray · index_selection · transcriptome
             benchmark · training · ingest · acceptance
                                                  (versioned, cross-cutting)
models/      features · evidence                  (feature layout + typed objects)
simulate.py  synthetic tiling screens (demos and tests only)
pipeline/    A candidates → B qc → C search → D pair evidence
             → E aggregate → F decide → G output   (+ optimizer, orchestrator)
metrics/     the six named metrics and the LCB/UCB bound algebra
util/        sequence and duplex primitives
```

## Run it

```bash
pip install -e ".[structure]"       # [structure] adds ViennaRNA (DR2)

# design against a pasted sequence, a FASTA file, or a built-in target
cas13guide design --sequence AUGCAUGC... --out results/
cas13guide design --fasta target.fa --min-accessibility 0.35 --out results/
cas13guide design --preset hsa-miR-21-5p
cas13guide design --list-presets
```

```bash
cas13guide serve                    # then open the page it prints
```

`serve` is the browser front end (FR1.2 upload, NFR2 no command-line
knowledge): paste or upload a sequence, pick a preset, get the same interactive
report. **Standard library only** — `http.server` plus `email` for multipart.
That is deliberate: choosing Flask or FastAPI would push a dependency *and* a
deployment shape onto whoever runs this, and neither is this package's call. It
is a development server — single-threaded, unauthenticated, running the design
inside the request — so it binds to localhost and says so loudly if asked to
bind anywhere else.

Both entry points are thin wrappers over `services/workflow.py:run_request`, so
a form POST and a command line reach the pipeline through one code path and
cannot drift into behaving differently. A real deployment can delete
`services/webapp.py` and keep everything else.

`design` runs the whole product in one command — input (FR1.1–FR1.5), scoring
(FR3/5/6), ranking (FR8.1) and CSV/FASTA export (FR8.3) — and is what a UI would
call. Malformed input is refused with a message naming the record and the
offending position rather than a stack trace (FR1.4, NFR3), and nothing is
silently repaired: `T`→`U` and whitespace go, everything else is an error,
because a guide designed against a sequence the tool quietly altered is worse
than no guide.

Two things it prints on purpose: that the built-in calibration is **synthetic**,
and that predicted signal is **relative**. Neither is a number to put in a paper.

```bash
pip install -e .            # or: export PYTHONPATH=src
python examples/demo_knockdown.py   # full pipeline on a small world
python -m cas13guide.cli check      # benchmark + conformal-coverage CI gate
python examples/demo_trained.py     # train a model, then design with it
python -m cas13guide.cli train screen.csv   # ingest a real screen + acceptance gate
python -m unittest discover -s tests -t tests    # 478 tests
pytest                              # same suite under pytest
```

## A ranked list is not a panel

Activity varies smoothly along a transcript, so the top of a raw ranking is
usually the *same window shifted by a base or two*. Measured on a 400 nt target
with a 28 nt spacer, the top ten ranked guides covered **two** distinct sites —
ranks 1, 2 and 3 differed by one nucleotide each:

```
#  1 start= 207  CGAGAUAAAUGGGUCGAGAGUUCGUGAC
#  2 start= 208  ACGAGAUAAAUGGGUCGAGAGUUCGUGA
#  3 start= 209  AACGAGAUAAAUGGGUCGAGAGUUCGUG
```

Those three share 27 of 28 target bases, the same local structure and the same
accessibility. They succeed or fail together. Ordering that list buys one
experiment ten times while looking, on screen, like ten independent candidates —
and it makes "test several candidates" impossible to follow from the list the
tool hands you.

So each ranked guide is marked `distinct_site`: is it the best-ranked
representative of a site not already listed above it? The CLI and the report
show distinct sites by default; `--all-sites` and the report's checkbox restore
the full list.

**It marks rather than filters, and ranks are never renumbered.** You see #1,
#2, #15, #22, #120 — which tells you the list was condensed, where a renumbered
1–5 would hide it. Nothing is dropped from the data; the exports carry the flag.

```
showing one guide per distinct site; 125 shifted copies of a listed site are
hidden (--all-sites to show them)
```

## What a run did not check

`services/limitations.py` returns the limitations of **this** run as data —
severity-tagged, with a source where the claim is about biology rather than
about this code. They lead the HTML report above the guide table, and they are
written into the provenance header of every CSV and FASTA export, so a file that
outlives the session still says what it did not check.

They are **derived from the run**, not a fixed banner. That is the whole point:
a static list keeps warning about placeholder kinetic constants after you have
supplied real ones, and stays silent about a heuristic off-target index because
nobody remembered to add the line. Either failure trains the reader to skim,
which is worse than saying nothing. So the same code produces different
limitations for different runs:

| Run | Includes | Omits |
|---|---|---|
| detection, built-in models | `provisional-kinetics` | `collateral-not-predicted` (collateral is the readout here) |
| knockdown, built-in models | `collateral-not-predicted` | `provisional-kinetics` (the run does not rank on signal) |
| fitted model, real probing track | — | `heuristic-activity-model`, `accessibility-is-predicted` |

Each one is tested in both directions — it appears when the condition holds and
is absent when it does not — because a warning that fires unconditionally is
indistinguishable from one that is broken.

### Exports carry their own provenance

Every CSV and FASTA leaves with a comment-prefixed header naming the version,
the spec fingerprint, every model version, any shortfalls and the limitations
above. Without it an exported table is numbers with no record of how they were
made, and it looks exactly like a table that has one. The CSV uses `#` (which
pandas' `comment='#'` and R's `comment.char='#'` skip) and the FASTA uses `;`,
the FASTA comment character — `#` would be read as sequence by some parsers, and
a provenance line concatenated onto a spacer is the kind of corruption that
survives review. The parser skips both, so an exported FASTA still reads back.

## Requirement coverage

| | Requirement | Where |
|---|---|---|
| ✅ | FR1.1–1.5 input, presets, validation, ortholog | `services/fasta.py`, `contract/presets.py`, `contract/effectors.py` |
| ✅ | FR2.1–2.3 tiling, PFS tagging, PFS deprioritisation | `pipeline/a_candidates.py`, `pipeline/b_qc.py` |
| ✅ | FR3.1–3.4 accessibility | `services/accessibility.py` |
| ⚠️ | FR4.1–4.3 BLAST off-target | `services/blast.py` — **needs DR1 + DR3** |
| ✅ | FR5.1–5.2 duplex free energy | `services/thermodynamics.py` |
| ⚠️ | FR6.1–6.2 kinetics, predicted signal | `services/kinetics.py` — **constants are placeholders**; relative ranking works, `absolute_signal()` refuses |
| ❌ | FR6.3 Bayesian panel confidence | **not implemented** — the model and its parameters are PanCRISP's and are not in this repository |
| ✅ | FR7.1–7.2 conserved regions, variant robustness | `services/conservation.py` |
| ✅ | FR8.1–8.4 ranking, table, export, visual | `pipeline/f_decide.py`, `services/fasta.py`, `services/report_html.py` |
| ✅ | NFR1–NFR6 | measured at 0.2–3.4 s; `serve`; located errors; provenance ledger; 375 px verified; score glossary |
| ❌ | DR1 RefSeq database, DR3 BLAST+, DR5 compute | infrastructure to provision |
| ✅ | DR2 ViennaRNA | `pip install -e ".[structure]"` |
| ⚠️ | DR4 model constants | JSON storage and loader exist; kinetic values are placeholders, Bayesian parameters absent |

**The three things this cannot do without you.** Real kinetic constants from
Feng et al. 2024 (`KineticConstants.from_json`, then set `provisional=False`);
the Bayesian diagnostic model for FR6.3, which has no stub because guessing at
someone else's statistical model is worse than leaving a hole; and a BLAST
database plus toolkit. Everything else runs today.

Two things it deliberately refuses rather than approximates: an absolute
fluorescence figure from placeholder constants, and a certified bound from
synthetic calibration. Both are announced on every run.

## The load-bearing ideas

**Asymmetric bounds.** Benefit is read at its lower confidence bound, hazard at
its upper one. `metrics/bounds.py` is the only place intervals are constructed,
so the asymmetry cannot be accidentally dropped downstream. A low point
prediction is never evidence of safety.

**Evidence is normalised at the pair level.** `PairEvidence` is
`guide × locus × context`; `GuideAssessment` is what you get after aggregation.
Nothing scalar exists before the gate.

**Gate → Pareto → preferences, in that order.** `pipeline/f_decide.py` enforces
it, and `rank_by_preference` drops FAIL verdicts itself rather than trusting
callers to pre-filter — so no user weighting can rescue a failed guide. The
non-dominated frontier is reported separately from the scalar ranking
(`DesignReport.pareto_frontier`), because the trade-off set is what a human
should actually look at.

## The twelve fixes, and where they live

| # | Fix | Module |
|---|-----|--------|
| 1 | Calibration as a first-class service (split conformal, stratified, CI-validated) | `services/calibration.py` |
| 2 | Discrimination margin `LCB(on)/UCB(off) ≥ τ` for allele/SNV tasks | `metrics/aggregate.py` |
| 3 | Universe adequacy as a **gate condition**, not documentation | `contract/enums.py`, `pipeline/f_decide.py` |
| 4 | Provable exhaustiveness bound instead of a soft completeness status | `services/indexes.py` |
| 5 | Explicit backtrack edges with iteration budgets and shortfall codes | `pipeline/orchestrator.py` |
| 6 | `UCB(expression)` — the hazard product is bounded on **both** axes | `services/context.py` |
| 7 | Versioned, auditable biological hazard weights | `contract/spec.py` |
| 8 | Benchmark / regression harness gating every registry change | `services/benchmark.py` |
| 9 | Wet-lab feedback routed to **calibration**, not to the spec | `services/benchmark.py` |
| 10 | Composition-matched, search-verified control guides | `pipeline/g_output.py` |
| 11 | Optimizer named: submodular greedy, guarantee reported honestly | `pipeline/optimizer.py` |
| 12 | Graded REVIEW (R1/R2/R3) plus a measurable `review_rate` | `pipeline/f_decide.py` |

## Loading a real screen

Everything above is validated on **simulated** screens. `services/ingest.py` is
the shortest path out of that: point it at a published screen and it produces
training examples, or tells you exactly why it will not.

```bash
python -m cas13guide.cli train screen.csv \
    --spacer-column guide_seq --activity-column log2fc \
    --group-column target_transcript --lower-is-better
```

Real files break in predictable ways, each handled and reported: DNA alphabet or
lowercase, ambiguity codes, missing or non-numeric activity, mixed spacer
lengths, duplicate guides, and — the one that silently inverts a model —
**log2 fold change**, where more negative means better knockdown. Rank
normalization makes that direction flip exact.

One condition is fatal rather than a warning: **no transcript column**. Without
groups there is no way to split by transcript, and every score would be measured
on near-duplicates of the training rows. The loader refuses rather than
returning a number that would look fine and mean nothing.

`services/acceptance.py` then gates the dataset on named criteria — rows,
transcript groups, calibration transcripts, grouped CV skill, held-out skill,
and conformal coverage — reporting each measured value so a rejection says
*which* property failed.

## Built-in targets, and the miRNA geometry problem

`contract/presets.py` ships the five miRNA targets, each with its miRBase
accession, and records exactly where the sequences came from — release, URL and
checksum — because a preset whose provenance is "miRBase" is not reproducible:

| preset | accession | length |
|---|---|---|
| hsa-miR-21-5p | MIMAT0000076 | 24 nt |
| hsa-miR-23a-3p | MIMAT0000078 | 23 nt |
| hsa-miR-216a-5p | MIMAT0000273 | 22 nt |
| hsa-miR-23b-3p | MIMAT0000418 | 23 nt |
| hsa-miR-29c-3p | MIMAT0000681 | 22 nt |

**A mature miRNA is 22–24 nt and an LwaCas13a spacer is canonically 28 nt.** You
cannot tile a 28-mer across a 24-mer, so at the canonical geometry these presets
yield *zero* candidates — not a poor design, no design at all. No choice of
software default makes that untrue; it is a property of the molecule. So the
package makes the trade-off explicit instead of hiding it:

- spacer length is configurable within each ortholog's tolerated range
  (`EffectorParams.spacer_length_range`), so a 24 nt spacer is expressible;
- a length outside the ortholog's *canonical* set is reported out-of-domain, so
  the bounds widen and the gate refuses to certify rather than inventing
  confidence for a geometry nobody validated;
- `designability()` states up front how many windows a target can yield, so
  "1 candidate" is a prediction rather than a surprise.

```python
designability(get_preset("hsa-miR-21-5p"), Effector.LWA_CAS13A).note
# "24 nt target cannot hold a 28 nt spacer. LwaCas13a tolerates 20-28 nt;
#  try a shorter spacer, or design against an amplicon (see amplicon_target)."
```

For a real assay the answer is to detect an **amplicon** — an RT-RPA product or
the pre-miRNA — which restores a normal-length target, normal tiling and the
canonical spacer length. `amplicon_target()` builds it.

### Hard constraints vs. soft preferences

Two rules were modelled as hard exclusions when the evidence only supports a
bias, and both silently deleted candidates:

| | was | now |
|---|---|---|
| LwaCas13a 3′ H preference | `PFS_INCOMPATIBLE` → deterministic FAIL | `PFS_DISFAVOURED` warning + a `pfs_preference` ranking objective |
| Non-canonical spacer length | `SPACER_LENGTH_UNSUPPORTED` → FAIL | `SPACER_LENGTH_NONCANONICAL` warning; out-of-*range* still fails |

A hard PFS requirement (PspCas13b's 5′ constraint) still excludes — the
distinction is now `PFSPolicy.*_allowed` (requirement) vs `*_preferred` (bias).

And when coverage fails *because* the evidence was out-of-domain or
uncalibrated, the gate now says so: the verdict is still FAIL, but the reason
distinguishes "nothing has been measured" from "this guide misses its target",
which point at completely different fixes.

## Diagnostic scoring: accessibility, thermodynamics, signal

Three axes that a knockdown designer does not need and a diagnostic one cannot
work without. All three sit behind Protocols with Null implementations, so the
package still runs — reporting *no evidence* rather than guessing — when
ViennaRNA is absent.

```bash
pip install -e ".[structure]"      # adds ViennaRNA (DR2)
python examples/demo_diagnostic.py
```

**Structural accessibility (FR3.x)** — `RNAplfold` in sliding 70-nt windows,
not a global fold: base pairs spanning kilobases dominate a global partition
function while being far less relevant in a cell, and local folding is
`O(n·W²)` instead of `O(n³)`.

A fold is deliberately **not** fed to the activity model. The design credits
accessibility to the model only from a source independent of it — SHAPE,
DMS-seq, icSHAPE — and a ViennaRNA fold is a prediction from the very sequence
the model already sees. Letting it in would have one computation vouch for
another and narrow a bound nothing new has constrained. So a computed fold
carries `is_independent_evidence = False`: it *ranks* candidates and can raise a
review, but never tightens a calibrated interval. A real probing track plugs
into the same Protocol, sets the flag `True`, and does reach the model.

**Duplex free energy (FR5.x)** — `RNAduplex`, reported per nucleotide so specs
transfer across spacer lengths. It is the one quantity here that is **never a
Pareto objective**, because it has an *optimum rather than a direction*: too
weak and the guide will not stay bound (no signal), too strong and it tolerates
mismatches it should reject (no discrimination). Every objective in
`MetricVector.objectives()` is oriented larger-is-better, and forcing an
interior optimum into that frame would make the frontier claim the most tightly
bound guide dominates — exactly backwards. It is reported and range-flagged
instead. The window is assay-specific, so it defaults to *off*;
`thermodynamics.summarize()` derives one from your own candidates.

**Predicted signal (FR6.x)** — a two-state ODE (activation, then
Michaelis–Menten trans-cleavage of the reporter) integrated over the assay
window with RK4. Effective target concentration is the nominal concentration
scaled by the guide's calibrated activity **lower** bound and its site
accessibility: a bright signal predicted off the wide end of an uncertain
interval is a hope, not a prediction.

Two things this refuses to do:

- **It will not emit absolute units from placeholder constants.** The real rate
  constants (Feng et al. 2024) are not in this repository. `relative_signal()`
  works regardless — a within-run ranking is invariant to the constants that
  scale it — but `absolute_signal()` raises until constants are supplied with
  `provisional=False`. A fluorescence figure in nM derived from stand-ins is a
  fabricated measurement wearing a unit label.
- **It will not rank knockdown guides by collateral cleavage.** Collateral is
  the *readout* of a detection assay and a *hazard* in a cell assay;
  `EffectorAssayProfile.collateral_is_hazard` already records which, and the
  objective is emitted only where signal is a benefit.

The shipped placeholder constants are also tuned to keep the assay in its
linear regime. A saturating assay scores every candidate 1.0 — not a weak
ranking but no ranking at all — and `SignalPrediction.saturated` says when real
constants have that problem.

## BLAST off-target search, and why it is the *heuristic* index

`services/blast.py` searches a BLAST database through the same `SearchIndex`
seam as the built-in indexes (FR4.1–FR4.3) — and reports `exhaustive=False`,
which the gate turns into an R2 REVIEW. That grading is the point, not a
limitation to work around:

- **Seeded, so it can miss.** BLAST extends exact word matches. A site whose
  differences leave no exact word of `word_size` intact is not found.
- **Not wobble-aware.** BLAST works in DNA space and scores G:U as a mismatch,
  because for DNA it is one. For a Cas13 RNA:RNA duplex it is a base pair, so
  BLAST both under-seeds wobble-rich sites and over-counts their mismatches.
  Reported mismatch counts are therefore **re-derived** with `profile_duplex`
  from the aligned subject sequence, never taken from BLAST's own column.
- **Only as complete as its database.** `bound.universes` records what the
  caller *asserts* the database covers — an assertion this module cannot verify.

Two filters worth knowing about: minus-strand alignments are dropped (a
transcript database holds sense strands, so a minus hit is not a site the guide
can bind, and counting one would invent an off-target), and partial or gapped
alignments are dropped (padding one out to spacer length would be inventing
sequence).

Requires BLAST+ (DR3) and a database (DR1), neither of which ships here. The
subprocess path is tested against a stub executable emitting real-format output,
which covers the argv, parsing, filtering and re-scoring — everything except
whether NCBI's aligner behaves, which is not this package's contract to test.

## The report

`design --out DIR` writes four files: `candidates.csv`, `candidates.fasta`,
`candidates_controls.fasta`, and `candidates.html`.

The HTML is one self-contained page — inline CSS and JS, an SVG accessibility
track drawn from the data, **no CDN and no external requests**. That is not
fussiness: a report that fetches a script at open time stops working on an
air-gapped lab machine, behind a proxy, or in three years. It sorts and filters
client-side (FR8.2), marks the top guides on the target's accessibility track
(FR8.4), reflows to a phone (NFR5), and explains every score in plain English
(NFR6) — including the two things the numbers cannot say for themselves: that a
bundled calibration is synthetic, and that predicted signal is relative.

Sequence names come from user files, so everything is HTML-escaped on the way
in; there are tests for a `<script>` in a spacer and for a label trying to break
out of an SVG `<title>`.

What this is **not** is a web server. FR1.2's upload and NFR2's "no command-line
knowledge" need one, and the framework choice belongs to whoever deploys it.
Everything a server would render is a pure function of
`DesignReport.to_dict()`, so wrapping it is a handful of lines rather than a
rewrite.

## Variants and conserved regions

```bash
cas13guide design --fasta target.fa --variants isoforms.fa
# conservation (2 variant(s)): 58/193 windows reach every variant within <=0 mismatches
```

**Conservation is measured in duplex space, not sequence space.** The obvious
implementation compares variants for identity, and that is the wrong question:
what matters is not whether two variants carry the same letter but whether one
spacer still *pairs* with both — and Cas13 binds RNA:RNA, so a G–U wobble is a
real base pair at a position where the variants differ. An identity-based
comparison discards guides that work perfectly well against every variant, and
does it silently. A window's conservation is therefore the worst *duplex*
mismatch its spacer incurs against any variant, computed by the same
wobble-aware exhaustive index the off-target search uses.

FR7.1 (*where can a shared guide exist*) is `services/conservation.py`. FR7.2
(*flag guides that lose a variant*) is the pipeline's existing `robustness`
metric — the minimum required coverage across contexts — and
`variant_contexts()` is what feeds it, one context per variant, all naming the
same target. A window that reaches no site at all in some variant reports
`worst_mismatches = None` rather than a large number: "no site" is worse than
any finite mismatch count and must not be flattened into one.

## Choosing a search index

Two exhaustive indexes ship, and **neither is universally faster**. Backtracking
over the FM-index costs roughly the size of the mismatch neighbourhood
(`C(L,k)·3^k` ≈ 590k strings for a 22-mer at k=4) and is nearly independent of
corpus size; the linear scanner is `O(N·L)` with a tiny constant. Measured per
query on random RNA in 2 kb loci:

| corpus | build | FM/query | scan/query | queries to amortize |
|--------|-------|----------|------------|---------------------|
| 600 kb | 1.16 s | 0.742 s | 0.018 s | never |
| 1.5 Mb | 5.83 s | 2.990 s | 0.054 s | never |
| 3 Mb | 12.56 s | 2.758 s | 0.089 s | never |

The scanner is vectorized: windows are compared a position at a time across one
concatenated array, with the surviving set compacted once the budget starts to
bind. That is **32× faster than the per-character scan it replaced**, on the
same machine, for a byte-identical hit set (0.678 → 0.021 s/Mb/query at 3 Mb).
Locus boundaries survive concatenation by the same two independent defenses the
FM-index uses — a joiner that is not a base, and an exact span check — and each
is pinned by its own test.

Suffix arrays are built with **SA-IS** (`services/suffixarray.py`), linear time,
2.5–2.7× faster than the prefix doubling it replaced and byte-identical in
output. The public entry point needs no unique-sentinel guarantee from the
caller and orders suffixes exactly as Python's string comparison would, so it
can be diffed against a brute-force sort.

Selection uses an explicit **cost model**, not a size threshold, because the
right answer depends on query volume as much as corpus size. Speeding the
scanner up moved the crossover from ~2.5 Mb to **~87 Mb** of searchable
sequence, so the FM-index is now the right call only at whole-transcriptome
scale — and the stale constants were actively harmful, building a suffix array
at 3 Mb where the scanner is ~30× faster per query and free to build. Below the
crossover no query volume can justify the build, because the scanner is cheaper
*per query* too. `services/index_selection.py` makes the call, and the reason
lands in the provenance ledger:

```python
ctx = DesignContext.auto("reference", transcriptome, universes, locus_to_target)
# index/exhaustive-scan/v1
# "2,400 bases x 1000 queries: scan ~0s beats FM ~2800s
#  (suffix-array build does not pay back)"
```

The constants are machine-specific; `examples/bench_index.py` re-measures them.
The tests pin the *shape* of the policy (query volume flips the decision above
the crossover, nothing flips it below) and derive the break-even from the
constants, so re-measuring cannot silently invalidate them.

Both return **identical hit sets** — there is a randomized differential test
asserting exactly that across mismatch budgets, repetitive sequence, and locus
boundaries — so the choice can only ever change speed, never results.

**Why backtracking and not pigeonhole seeding.** The standard trick is to split
the pattern into `k+1` parts, since any occurrence with ≤k mismatches must match
one part exactly. That filter is complete for Hamming distance and *silently
breaks here*: G–U wobbles pair for free but are literal differences, so a part
containing only wobbles fails exact lookup. Wobble count is unbounded, so no
partition size restores the guarantee — the filter would drop real hits while
still reporting `exhaustive=True`. Backtracking charges 0 for a match, 0 for a
wobble, 1 otherwise, and has no such blind spot.

## Two details worth knowing

**G–U wobble is asymmetric.** Cas13 binds RNA:RNA, so wobble pairs tolerate
mismatches a DNA-targeting nuclease would not. Working it through in target
space (`util/duplex.py`), exactly two substitutions still pair: `C→U` and
`A→G`. The reverses are true mismatches. Treating wobble as symmetric silently
inflates apparent specificity.

**Collateral activity is contextual, not a homology score.** For most Cas13
orthologs trans-cleavage is a property of *activation*, so it scales with target
abundance rather than with off-target similarity. The report keeps it separate
from specificity and the validation plan assays it directly.

## Seams for real components

Everything expensive sits behind a `Protocol` with a working reference
implementation:

| Seam | Reference impl | Drop-in |
|------|----------------|---------|
| `SearchIndex` | `ExhaustiveScanIndex`, `FMIndexSearchIndex` (both provably complete), `SeedHeuristicIndex` and **`BlastOffTargetIndex`** (honest heuristics) | — already production-shaped |
| `ActivityModel` | `HeuristicCas13Model`, **`TrainedActivityModel`** (fit by `services/training.py`) | trained CNN, TIGER-style model |
| `AccessibilityModel` | `NullAccessibilityModel`, **`ViennaRNAAccessibilityModel`** (fold, not independent) | SHAPE / DMS-seq / icSHAPE via `StaticAccessibilityModel` |
| `DuplexEnergyModel` | `NullDuplexEnergyModel`, **`ViennaRNADuplexEnergyModel`** | any hybridization model |
| `KineticsModel` | `NullKineticsModel`, **`CollateralKineticsModel`** (provisional constants) | PanCRISP constants from Feng et al. 2024 |
| `ExpressionSource` | `StaticExpressionSource` | GTEx, user RNA-seq, scRNA panel |

Accessibility returning `None` is deliberate: the design credits it only when it
comes from a source independent of the activity model, never from the
thermodynamic fold the model already used as a feature.

## Verification

478 tests, run under `unittest` or `pytest`. The suite is mutation-checked: the
invariants the README claims are each pinned by a test that fails when the
behaviour is reverted.

Fifty-six of those tests were added after a review found bugs the original
suite did not cover, and each one fails against the code as it stood. The ones
worth naming, because they were live defects rather than hypotheticals:

| What was wrong | Why it mattered | Pinned by |
|---|---|---|
| The FM walk branched over a hard-coded `ACGU` | `N` is a legal base, so any window overlapping an assembly gap was unreachable — the index returned a **smaller hit set than the scanner while still reporting `exhaustive=True`**, and a 1-mismatch paralog off-target simply vanished | `test_search.py` |
| Universe adequacy compared the *declaration* against the assay, never the search | An index built over fewer universes than the spec declares produced a clean off-target report: same spec, same transcriptome, 42 FAIL → 42 PASS with zero off-target rows, purely by narrowing the index | `test_gate.py` |
| `discrimination_margin` took its `min` over targets that *had* evidence | A required allele missing from a context **raised** the margin (6.00 vs 5.67) instead of zeroing it, and the cross-context `min` then hid the gap | `test_metrics.py` |
| `validate_coverage` scored the `[0,1]` fallback as coverage | A stratum with no calibration data at all reported **1.000 coverage and passed** — losing calibration made the CI gate greener | `test_calibration.py` |
| `DEFAULT_WEIGHTS` omitted `discrimination_margin` | The one metric an allele task exists for had weight 0, so it could not affect the ranking — while the rationale line still printed it | `test_decision_order.py` |
| `HazardWeights` replaced its default maps instead of merging | A partial `by_biotype` halved rRNA/mito hazard; a partial `by_class` raised `KeyError` mid-run | `test_context_evidence.py` |
| Controls were checked against a hand-copied subset of QC | A "matched" control could ship carrying a GC or self-folding violation the gate would FAIL | `test_controls.py` |
| `already_normalized` ignored `higher_is_better` | A pre-scaled depletion screen loaded with **every label inverted**, silently — the failure `ingest.py` calls fatal | `test_ingest.py` |
| `record_wetlab_result` dropped the group label | Zero group labels switches the `MIN_STRATUM_GROUPS` check off, so wet-lab feedback from one transcript came back looking certified | `test_benchmark.py` |
| `run_search` kept the *last* bound, not the weakest | A per-query heuristic bound would be overwritten by a later exhaustive one | `test_search.py` |
| `profile_duplex("", "")` died in the CIGAR builder | An `IndexError` that named neither the caller nor the cause | `test_duplex.py` |

The vectorized scanner is held to the original per-character implementation as
an oracle: randomized differential tests over ambiguity codes, loci shorter than
the spacer, and spacers drawn across locus junctions, plus a test that every
compaction point yields the same hit set.

The earlier forty-four targeted mutants are still killed: asymmetric seed
taper, naive min-max normalization, an unfiltered ranker, shallow control
verification, an unflagged conformal level, a deleted retile loop, averaged
cross-context hazard, UCB-for-LCB in the optimizer, reference-only evidence at
the gate, unmeasured-expression-as-zero — and, on the index, a pigeonhole-style
wobble shortcut, an off-by-one in `rank`, a crossable separator, a dropped span
check, early pruning, a wrong BWT rotation, an off-by-one `C` table, truncated
suffix-array doubling, and a selector that ignores corpus size; and on the
model layer, an ungrouped `group_kfold`, a train/calibrate overlap, a domain
widened past the training data, a penalized intercept, target-coordinate
mismatch features, position-blind one-hot, unnormalized dinucleotides, a
single-step IRLS, intent-based stratification, and a collapsed mismatch
regime; and on the data layer, a skipped SA-IS recursion, a dropped induction
sweep, inverted type propagation, an unstripped sentinel, a downgraded
missing-group error, an ignored direction flag, accepted ambiguity codes,
unchecked normalized ranges, a disabled group-adequacy check, unrecorded group
labels, always-passing acceptance criteria, and a flattened metric summary.

The two locus-boundary defenses are deliberately redundant, so each is pinned by
its own test — removing either alone is masked by the other, and a refactor that
deleted one would leave the system a single edit away from inventing
off-targets.

Test modules map to the claims rather than to the file layout:

| Module | Pins |
|--------|------|
| `test_duplex.py` | wobble asymmetry, spacer-coordinate convention, CIGAR orientation, seed taper |
| `test_calibration.py` | conformal coverage, fallback chain, uncertifiable levels, `(n+1)` correction |
| `test_metrics.py` | min-not-mean coverage, cumulative burden, both-axis bounds, discrimination margin |
| `test_gate.py` | every FAIL condition, R1/R2/R3 grading, FAIL outranks REVIEW |
| `test_decision_order.py` | Pareto, infinite objectives, order-independence, gate-before-ranking |
| `test_optimizer.py` | complementary selection, diminishing returns, spacing/diversity, LCB |
| `test_search.py` | exhaustive vs heuristic, `covers()` semantics, wobble hits |
| `test_fmindex.py` | suffix array vs brute force, rank exactness, **differential equivalence with the scanner**, locus-boundary defenses |
| `test_index_selection.py` | crossover policy, build amortization, choice cannot affect results |
| `test_orchestrator.py` | backtrack loops fire and terminate, shortfalls recorded once |
| `test_controls.py` | composition matching, verification depth |
| `test_context_evidence.py` | "not detected" ≠ "absent" |
| `test_candidates_qc.py` | tiling, dedup, PFS policy, composition filters |
| `test_benchmark.py` | skill/coverage regression gates, wet-lab → calibration |
| `test_suffixarray.py` | SA-IS vs brute force, edge cases, deliberate recursion-path coverage |
| `test_presets.py` | miRBase accessions and provenance, mature-miRNA geometry, the amplicon path |
| `test_structure_and_signal.py` | fold-vs-probing independence, energy is not an objective, signal monotonicity, provisional-constant guard |
| `test_io.py` | FASTA/raw parsing, error messages that locate the fault, CSV/FASTA export, REVIEW guides reach the ranking |
| `test_conservation.py` | conserved windows in duplex space, wobble tolerance, losing a variant lowers robustness |
| `test_limitations.py` | each warning appears exactly when its condition holds, and not otherwise |
| `test_webapp.py` | upload and paste round trips, defaults shared with the CLI, errors as messages not tracebacks |
| `test_blast.py` | short-query flags, strand and length filtering, wobble re-scoring, never claims exhaustiveness |
| `test_report_html.py` | balanced markup, no external requests, sort keys on every cell, escaping of untrusted names, score glossary completeness |
| `test_ingest.py` | direction flip exactness, fatal vs survivable data problems |
| `test_acceptance.py` | each criterion measured, rejection names the failure |
| `test_training.py` | IRLS vs sklearn oracle, grouped splitting, the measured split-bias regimes, unpenalized intercept |
| `test_trained_end_to_end.py` | fitted model satisfies the seam, ranking becomes informative, thin calibration refuses to certify |

## Training a real model

`HeuristicCas13Model` saturates — a perfect duplex scores ≈0.93 regardless of
sequence — so every perfect-match guide lands on the same `LCB` and the ranking
degenerates to tie-breaks. `services/training.py` replaces it. Measured on the
same design task (`examples/demo_trained.py`):

Discrimination is measured across **every assessed guide**, not just ranked
ones — the ranked list holds gate survivors, so a model that fails everything
would show "zero distinct scores" for reasons unrelated to discrimination:

| model | assessed | distinct coverage values | coverage spread |
|---|---|---|---|
| heuristic | 111 | 15 | 0.146 |
| trained (ridge-logistic) | 111 | 106 | 0.678 |

Nothing downstream changed — the fitted model enters through the same
`ActivityModel` protocol. `DesignReport.metric_summary` carries these numbers,
because when a run returns zero PASS the first question is whether the model
discriminated at all, and the ranked list cannot answer it.

**Splitting is grouped by transcript**, and the reason is the estimand rather
than a folk warning: grouped CV answers *"how will this do on a transcript it
has never seen"*, which is the deployment condition. Ungrouped CV answers a
question nobody designing guides is asking.

The size and even the **sign** of the ungrouped bias turns out to be regime
dependent, so the package measures it rather than asserting it
(`compare_splitting`):

| rows | grouped | ungrouped | gap |
|---|---|---|---|
| 156 | 0.540 | 0.703 | **+0.163** |
| 276 | 0.812 | 0.871 | +0.059 |
| 594 | 0.727 | 0.734 | +0.006 |
| 1432 | 0.743 | 0.697 | −0.046 |
| 2628 | 0.828 | 0.782 | −0.046 |

Two effects compete. **Leakage** inflates ungrouped scores when the model has
capacity to memorize transcript-level offsets (`n ≲ 2p` here; far larger `n` for
the high-capacity models real screens are fit with). **Test-set composition**
inflates *grouped* scores when data is plentiful, because a grouped test fold
holds two or three whole transcripts and so carries less unpredictable
between-transcript spread. Remove the transcript offset entirely and the gap is
exactly `0.000`, which pins the mechanism — neither effect comes from sequence
overlap alone. So "grouped is always more conservative" is false, and this
package does not claim it.

The protocol is a **three-way grouped split**: train fits weights, calibrate
fits the conformal quantiles, test measures skill *and* coverage. All disjoint
by transcript.

### Calibration strata are error regimes, not intent

Stratifying by target class (REQUIRED / FORBIDDEN / …) conflates *intent* with
*error*. What actually changes the model's error distribution is how far a
duplex sits from the dense part of the training screen, so strata are
`effector | assay | mismatch-regime` (`mm0`, `mm1-2`, `mm3+`). A REQUIRED and a
FORBIDDEN hit with the same duplex share a stratum; a perfect and a
4-mismatch hit against the same target do not.

### Grouped calibration weakens the conformal guarantee

Split conformal assumes calibration and test points are exchangeable. Under
grouped splitting they are not: calibration rows come from one set of
transcripts, test rows from another. Measured on simulated screens, nominal
0.90, **mean over 9 seeds**:

| calibration transcripts | mean coverage | sd | worst seed |
|---|---|---|---|
| 2 | 0.881 | 0.124 | 0.585 |
| 4 | 0.852 | 0.113 | 0.667 |
| 8 | 0.840 | 0.096 | 0.609 |
| 21 | 0.889 | 0.057 | 0.808 |
| 40 | 0.876 | 0.048 | 0.773 |

The honest reading is *not* "few groups means low coverage" — the mean is
roughly flat and slightly under nominal throughout, which is the exchangeability
violation showing up as a small optimistic bias. What collapses with few groups
is **reliability**: with two calibration transcripts a single screen can land at
0.585 actual coverage; with forty the worst case seen was 0.773.

So the package does two things and claims nothing more. A stratum backed by
fewer than `MIN_STRATUM_GROUPS` independent groups still returns its bound, but
flagged `ood` — usable, not certified. And coverage is never assumed, only
measured on held-out groups and gated in CI. That measurement is the protection
that actually holds.

### Thin calibration refuses to certify

With a small screen (69 mm0 calibration rows) the conformal interval is ~0.50
wide, the lower bound falls under the coverage floor, and the gate FAILs
everything. That is the architecture working: it declines to certify guides
whose evidence base cannot support the claim. More calibration data narrows the
interval and guides start passing — both behaviours are pinned by tests.
