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
pip install -e .            # or: export PYTHONPATH=src
python examples/demo_knockdown.py   # full pipeline on a small world
python -m cas13guide.cli check      # benchmark + conformal-coverage CI gate
python examples/demo_trained.py     # train a model, then design with it
python -m cas13guide.cli train screen.csv   # ingest a real screen + acceptance gate
python -m unittest discover -s tests -t tests    # 221 tests
pytest                              # same suite under pytest
```

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

## Choosing a search index

Two exhaustive indexes ship, and **neither is universally faster**. Backtracking
over the FM-index costs roughly the size of the mismatch neighbourhood
(`C(L,k)·3^k` ≈ 590k strings for a 22-mer at k=4) and is nearly independent of
corpus size; the linear scanner is `O(N·L)` with a tiny constant. Measured per
query on random RNA:

| corpus | build | FM/query | scan/query | queries to amortize |
|--------|-------|----------|------------|---------------------|
| 600 kb | 1.19 s | 0.998 s | 0.399 s | never |
| 1.5 Mb | 4.06 s | 1.778 s | 1.018 s | never |
| 3 Mb | 9.64 s | 1.939 s | 2.062 s | **78** |

Suffix arrays are built with **SA-IS** (`services/suffixarray.py`), linear time,
2.5–2.7× faster than the prefix doubling it replaced and byte-identical in
output. The public entry point needs no unique-sentinel guarantee from the
caller and orders suffixes exactly as Python's string comparison would, so it
can be diffed against a brute-force sort.

Selection uses an explicit **cost model**, not a size threshold, because the
right answer depends on query volume as much as corpus size — at 3 Mb the
break-even is 78 queries. `services/index_selection.py` makes the call, and the
reason lands in the provenance ledger:

```python
ctx = DesignContext.auto("reference", transcriptome, universes, locus_to_target)
# index/exhaustive-scan/v1
# "2,391 searchable bases is below the 2,500,000 crossover; the linear
#  scanner is faster here"
```

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
| `SearchIndex` | `ExhaustiveScanIndex`, `FMIndexSearchIndex` (both provably complete), `SeedHeuristicIndex` (honest heuristic) | — already production-shaped |
| `ActivityModel` | `HeuristicCas13Model`, **`TrainedActivityModel`** (fit by `services/training.py`) | trained CNN, TIGER-style model |
| `AccessibilityModel` | `NullAccessibilityModel` (returns `None`) | SHAPE / DMS-seq / icSHAPE |
| `ExpressionSource` | `StaticExpressionSource` | GTEx, user RNA-seq, scRNA panel |

Accessibility returning `None` is deliberate: the design credits it only when it
comes from a source independent of the activity model, never from the
thermodynamic fold the model already used as a feature.

## Verification

221 tests, run under `unittest` or `pytest`. The suite is mutation-checked: the
invariants the README claims are each pinned by a test that fails when the
behaviour is reverted. Forty-four targeted mutants are all killed: asymmetric seed
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
