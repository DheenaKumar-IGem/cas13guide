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

If the console script is not on your PATH — the usual outcome of
`pip install --user` on Windows — `python -m cas13guide <args>` is the same CLI
and needs nothing but the interpreter that installed it.

`serve` is the browser front end (FR1.2 upload, NFR2 no command-line
knowledge): paste or upload a sequence, pick a preset, get the same interactive
report. **Standard library only** — `http.server` plus `email` for multipart.
That is deliberate: choosing Flask or FastAPI would push a dependency *and* a
deployment shape onto whoever runs this, and neither is this package's call. It
is a development server — unauthenticated, unthrottled, running the design
inside the request — so it binds to localhost and says so loudly if asked to
bind anywhere else. It *is* threaded, so one slow design does not refuse
everyone else.

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
python -m unittest discover -s tests -t tests    # 828 tests
pytest                              # same suite under pytest
```

## A real trained model on the ActivityModel seam

The built-in `HeuristicCas13Model` saturates — a perfect duplex scores ≈0.93
whatever the sequence — so the ranking among well-matched guides carries almost
no information. `services/arc_model.py` replaces it with the **Arc RNAtargeting
CNN** (Wei et al., *Cell Systems* 2023), running the authors' published
`guide_nolin_threef` weights as a three-fold ensemble. The weights ship in the
package under MIT; see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

```bash
pip install -e ".[arc]"      # adds onnxruntime
```

**Parity is the gate.** `tests/test_arc_model.py` drives 133 guides from the
reference implementation and reproduces its scores to `<1e-6`. Four things in an
adapter like this keep every tensor shape valid and every score in range while
being wrong, and only end-to-end parity sees all four:

| Trap | What it costs |
|---|---|
| One-hot channel order is **A, T, C, G**, not ACGT | every score plausible and wrong |
| Ensemble averages **logits**, then sigmoids | measured: up to 0.037 of score, mean 0.007 — enough to reorder guides |
| Scalars ordered `linearfold_val, isoform_percent, target_unfold_energy` | silently shifts every prediction |
| Normalizers `6.48 / 11.93 / 15.67` are training-time values | ditto |

**It now has real folding scores, and still no ViennaRNA fallback.**
`services/contrafold.py` supplies them — the CONTRAfold CRF in pure Python, so
no compiler is needed, which matters because LinearFold has no Windows wheel on
PyPI, no `win-64` build on bioconda and no binary releases at all. It reproduces
all 399 reference folds **exactly** and drives the CNN to within `5.96e-08` of
the reference scores. Two details are load-bearing: LinearFold defaults to
`sharpturn=false` (imposed here through the parameter table, not by editing a
dynamic program), and its `%.2f` output quantisation is part of the number.
**Zero** of the fixture's 266 sequences contain a sharp turn, so parity alone
would pass green with that convention set either way — `test_contrafold.py`
carries separate known-answer vectors for it.

```bash
pip install -e ".[arc,contrafold]"
cas13guide design --fasta target.fa --effector RfxCas13d --spacer-length 30 --arc
```

**Why there is still no ViennaRNA fallback.** Two of
the three scalars are folding scores in LinearFold's **CONTRAfold**
parameterisation — *not* free energies in kcal/mol, despite the `mfe` name. They
are dimensionless log-odds where **higher means more stable**, quantised to two
decimals by LinearFold's `printf("%.2f")`. All 133 `guideMfe` values in the
fixture are positive (+6.48 to +11.96), which no kcal/mol MFE ever is, so a real
free energy substituted here is not merely inaccurate — it is sign-inverted. ViennaRNA — which this package *does* ship — uses the Turner
model, and its energies are simply different numbers. Substituting them yields a
confident, in-range, wrong score, which is the one failure this codebase exists
to prevent. `MFESource` is the seam; without one, `features_for_target()`
refuses and says why.

**Its domain is one perfect duplex.** The network sees the spacer and its
intended target's folding; it never sees a mismatched site, so `max_mismatches`
is `0`. Ask it about an off-target and the pipeline marks the answer
out-of-domain and widens the interval rather than passing an on-target efficacy
prediction off as an off-target risk estimate.

> **It found a bug on arrival.** The Arc prefix *is* the RfxCas13d direct
> repeat, and it did not match the one `contract/effectors.py` had always
> carried — a 45 nt sequence that folds to −1.1 kcal/mol on a 3 bp stem. A
> Cas13 repeat has to present a stable hairpin for the protein to grip; the
> other two orthologs here fold to −6.7 and −11.3, and the corrected RfxCas13d
> repeat folds to −15.4. The bad value even began as a near-copy of the real
> one (`CAAC*U*AAACCC*U*` against `CAAG*U*AAACCC*C*`) before diverging. It was
> load-bearing: `b_qc.py` flags spacers that can base-pair with their own
> repeat, and across 20,000 random 22 nt spacers the old value flipped that
> verdict for **4.9%** of them. Fixed, with the structural invariant pinned in
> `test_effector_constants.py` so the next bad value fails loudly.

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

### And a table is not a recommendation

The panel is PASS-only, deliberately — the machine's recommendation stays
certified even where the table a human triages is wider. Under the built-in
synthetic calibration every guide lands REVIEW, so there *is* no panel, and the
terminal used to print the shortfall only when nothing ranked at all. The
dangerous case is the opposite one: 264 ranked rows, an empty panel, and not a
word about it. The report and every export header carried the shortfall the
whole time; the surface a person actually reads did not.

```
  ! no panel: guides ranked but none PASSed the gate, and the panel is
    PASS-only. The rows above are candidates to triage, not a recommendation.
  ! no feasible panel covers required target(s): demo
```

## What a panel is worth (FR6.3)

`Panel.coverage` is a noisy-OR of the members' calibrated lower bounds,
`1 - prod(1 - lcb)`. It reads like the probability the panel detects the target
and it is not one, for two reasons that are separate mistakes and are fixed
separately in `services/panel_confidence.py`.

**A calibrated activity bound is not a success probability.** `lcb` bounds a
*continuous* predicted activity; whether a guide works is a *binary* event —
observed activity clears the assay threshold, or it does not. Converting one
into the other takes data about how often predictions at that level came true,
and that data is already here: the calibration strata hold `(predicted,
observed)` pairs. So per guide the model takes the calibration rows whose
prediction sits near this guide's, counts how many cleared the threshold, and
updates a Jeffreys `Beta(0.5, 0.5)` prior with that binomial likelihood. The
credible interval that comes back is wide exactly when the calibration set is
thin, which is the honest failure mode.

That posterior is built from **counts**, deliberately, not from the conformal
interval. The earlier attempt at FR6.3 derived its scale as `q / z` — a
distribution-free conformal quantile over a Gaussian one — which imports the
assumption conformal exists to avoid. Nothing in this module reads a conformal
quantile.

**Panel members do not fail independently.** They share a transcript, a prep
and one activity model whose errors are correlated; the spacing constraint only
stops them sharing a *site*. So the noisy-OR is not a probability, it is a
**ceiling** — the value reached only if the members really were independent.
The matching floor is the comonotone case, where they fail together as far as
their marginals allow and the panel is worth its best member:

```
max_g p_g   <=   P(panel detects t)   <=   1 - prod_g (1 - p_g)
```

Both ends are Fréchet–Hoeffding bounds. They need no parameter nobody measured,
and the gap between them is the width of the claim rather than a defect in it.
This package reads benefit at its lower bound everywhere else, so the number to
quote is the floor: **extra guides have to be shown independent before they
count.**

`examples/demo_trained.py`, whose calibration is a fitted screen rather than a
stand-in, prints the correction plainly. Its top guide carries a coverage of
`0.677`; guides predicted at that level cleared threshold in **52%** of the
calibration rows, so that is what the panel is worth:

```
--- panel detection probability (FR6.3) ---
  t_tgt: [0.523, 0.523]  from 1 member(s): g_2a2018f21c=0.523
```

Floor equals ceiling there because the spacing constraint left one member, and
one guide raises no independence question. Add a second member of the same
quality and the two ends separate — `[0.523, 0.772]` — which is the whole
disagreement made visible: the old number would have printed `0.772` alone.

Collapsing that to a single number takes the common-cause fraction `beta` of
the standard beta-factor reliability model (IEC 61508-6): each member's failure
splits into a shared part `beta * q_g` and an independent part `(1 - beta) q_g`,
and `Q = beta * mean(q_g) + prod((1 - beta) q_g)`. At `beta = 0` that is the
noisy-OR back again; at `beta = 1` the panel is one guide wearing three hats.
`beta` is a property of replicate *panel* experiments and cannot be estimated
from the calibration set this package holds, so **no default ships** — not even
a plausible one, because a plausible number in the slot where a measured one
goes does not look like a hole. Supply yours and the point estimate unlocks:

```bash
cas13guide design --preset hsa-miR-21-5p --panel-dependence beta.json
```

```json
{"common_cause_beta": 0.18, "source": "12 replicate 3-guide panels, 2026-08",
 "provisional": false}
```

`provisional: true` is refused exactly as an absent value is, or the flag would
be decoration. Same bargain as `--kinetics-constants`, for the same reason.

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

**Every** means the controls FASTA too, which is where it matters most. A
non-targeting control is a negative result that looks exactly like a guide —
same length, same base composition, ordered into the same plate — and six
months on, nothing tells it apart from a real candidate except a record of the
run and panel it was matched to. It also states its own count, because an empty
control set used to be a **zero-byte file announced beside the CSV and the
report**, which reads as "no controls needed" when the fact is "the panel these
would have been matched to is empty". Those are opposite claims, and the file
now makes the second one out loud.

```
; non-targeting controls: 0
; none were generated, because the panel they are matched to is empty -- a
; control is a shuffle of a panel spacer. The shortfalls above say why.
```

## Requirement coverage

| | Requirement | Where |
|---|---|---|
| ✅ | FR1.1–1.5 input, presets, validation, ortholog | `services/fasta.py`, `contract/presets.py`, `contract/effectors.py` — all five presets verified against miRBase release 23 (sha256 pinned, re-checkable via `verify_against_mirbase()`) |
| ✅ | FR2.1–2.3 tiling, PFS tagging, PFS deprioritisation | `pipeline/a_candidates.py`, `pipeline/b_qc.py` |
| ✅ | FR3.1–3.4 accessibility | `services/accessibility.py` |
| ✅ | FR4.1–4.3 BLAST off-target | `services/blast.py` — `--blast-db`, refusing legibly without BLAST+. `--background` is the exhaustive, wobble-aware alternative that needs no database |
| ✅ | FR5.1–5.2 duplex free energy | `services/thermodynamics.py` |
| ⚠️ | FR6.1–6.2 kinetics, predicted signal | `services/kinetics.py` — **constants are placeholders**; relative ranking works, `absolute_signal()` refuses. Supply measured values with `--kinetics-constants FILE.json` and it unlocks |
| ✅ | FR6.3 Bayesian panel confidence | `services/panel_confidence.py` — Beta-Binomial success posterior per member from the calibration counts, combined under Fréchet bounds. Reported as an interval; a point estimate needs a measured common-cause fraction (`--panel-dependence`) and is refused without one |
| ✅ | FR7.1–7.2 conserved regions, variant robustness | `services/conservation.py` |
| ✅ | Trained activity model | `services/arc_model.py` + `services/contrafold.py` — Arc CNN on real CONTRAfold energies, 399/399 reference folds exact. `--arc` |
| ✅ | FR8.1–8.4 ranking, table, export, visual | `pipeline/f_decide.py`, `services/fasta.py`, `services/report_html.py` |
| ✅ | NFR1–NFR6 | measured at 0.2–3.4 s; `serve`; located errors; provenance ledger; 375 px verified; score glossary |
| ⚠️ | DR1 RefSeq database, DR3 BLAST+, DR5 compute | your infrastructure to provision — no compiler or admin needed; `--background` needs none of it |
| ✅ | DR2 ViennaRNA | `pip install -e ".[structure]"` |
| ⚠️ | DR4 model constants | JSON storage and loaders exist for both (`--kinetics-constants`, `--panel-dependence`); kinetic values are placeholders and no common-cause fraction ships |

**The two measurements this cannot make for you.** Real kinetic constants —
`k_cat` and `K_M` for your reporter, which Feng et al. 2024 does not report, so
they have to come from a plate reader (`--kinetics-constants FILE.json`, with
`provisional=false`); and the common-cause fraction of panel failure, which
comes from replicate panel runs (`--panel-dependence FILE.json`). Both have
working machinery and no shipped value, and in both cases the missing number
buys a *narrower* claim rather than no claim: signal still ranks, and panel
confidence still reports the interval it can prove. What neither will do is
manufacture the input — an earlier FR6.3 attempt derived its scale as `q / z`,
dividing a distribution-free conformal quantile by a Gaussian one, which is the
exact assumption conformal exists to avoid.

BLAST is optional rather than blocking: `--background <FASTA>` already runs an
exhaustive, wobble-aware scan over a transcriptome you supply, and reports
`exhaustive=True`, which a seeded heuristic cannot.

Four things it deliberately refuses rather than approximates: an absolute
fluorescence figure from placeholder constants, a single panel detection
probability without a measured dependence parameter, a certified bound from
synthetic calibration, and an Arc score computed from Turner energies. The
first three are announced on every run; the last raises.

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
  confidence for a geometry nobody validated. The domain pairs length **with
  ortholog**: it used to be the union across the two, which admitted RfxCas13d's
  22/23/30 for LwaCas13a and LwaCas13a's 28 for RfxCas13d — so four of the five
  miRNA presets were being certified at a length no model here was fit on, on
  the package's own headline path;
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
| `ActivityModel` | `HeuristicCas13Model`, `TrainedActivityModel` (fit by `services/training.py`), **`ArcActivityModel`** (Wei et al. 2023 weights) | TIGER-style model |
| `AccessibilityModel` | `NullAccessibilityModel`, **`ViennaRNAAccessibilityModel`** (fold, not independent) | SHAPE / DMS-seq / icSHAPE via `StaticAccessibilityModel` |
| `DuplexEnergyModel` | `NullDuplexEnergyModel`, **`ViennaRNADuplexEnergyModel`** | any hybridization model |
| `KineticsModel` | `NullKineticsModel`, **`CollateralKineticsModel`** (provisional constants) | PanCRISP constants from Feng et al. 2024 |
| `ExpressionSource` | `StaticExpressionSource` | GTEx, user RNA-seq, scRNA panel |

Accessibility returning `None` is deliberate: the design credits it only when it
comes from a source independent of the activity model, never from the
thermodynamic fold the model already used as a feature.

## Verification

828 tests, run under `unittest` or `pytest`. The suite is mutation-checked: the
invariants the README claims are each pinned by a test that fails when the
behaviour is reverted — and that sentence is now *measured* rather than
asserted, by the same rule that gave `verify_against_mirbase()` its reason to
exist. A claim is worth what the procedure that reproduces it is worth.

```bash
python examples/verify_mutation_coverage.py     # 32 mutations, 32 caught, 0 holes
```

It copies the tree to a temporary directory, applies one small **semantic**
change at a time — code that still runs and still looks plausible — and reports
any that the suite fails to notice. Two things keep the result honest rather
than flattering. The mutations are subtle: disabling a whole check is caught by
any competent suite and proves nothing, so half of these instead move a boundary
one step, swap a bound for its opposite, drop the finite-sample `(n+1)`
correction from the conformal quantile, or take the right operation on the wrong
variable. And a survivor is confirmed against the **whole** suite before being
called a hole — **10 of the 32 are caught only by the full run**, so a screening
subset would have reported ten holes that are not there.

What it mutates: the LCB/UCB asymmetry, the six named metrics, every FAIL branch
in the gate, the QC checks, the panel's spacing and diversity constraints, the
candidate-budget split, the out-of-domain substitution, the conformal quantile
and its group-adequacy floor, the search-bound algebra, the FR6.3 posterior, the
kinetics stability guard, the conservation worst-case, and the wobble rule.

Two hundred and nine of those tests were added after eight review passes found
bugs the original suite did not cover, and each one fails against the code as it
stood. Each pass re-read the previous pass's own diff, and three of them
caught a regression the previous pass had introduced — which is the argument for
doing that, and also its limit.

Confirmed defects per pass were **25, 4, 2, 6, 9, 7, 3, 4**, and the shape of
that curve is the useful part. Passes one to three *read* the code — module by module,
then diff by diff — and converged at 2. Passes four to six *ran* it: flag
combinations rather than one flag at a time, input fuzzing, determinism across
process boundaries, the three output surfaces compared field by field, real HTTP
POSTs, the `train` and `demo` subcommands nobody had exercised, the
`--out`/`--stem` paths, and the README audited claim by claim. The count went
straight back up. Reading a module tells you whether it is self-consistent; only
running the thing tells you whether two modules agree — and nearly everything
left was two modules disagreeing.

The eighth pass turned the instruments on the project's own quality machinery
rather than on the product — the docstrings audited claim by claim, a sweep for
tests that cannot fail, and the four service modules nothing had pointed at.

Three instruments found nothing that survived verification, which is worth as
much as a finding: differential testing across all four `SearchIndex`
implementations, property testing of the gate → Pareto → preference order, and
the threaded server under genuine concurrent load. The ones worth naming,
because they were live defects rather than hypotheticals:

| What was wrong | Why it mattered | Pinned by |
|---|---|---|
| The FM walk branched over a hard-coded `ACGU` | `N` is a legal base, so any window overlapping an assembly gap was unreachable — the index returned a **smaller hit set than the scanner while still reporting `exhaustive=True`**, and a 1-mismatch paralog off-target simply vanished | `test_search.py` |
| Universe adequacy compared the *declaration* against the assay, never the search | An index built over fewer universes than the spec declares produced a clean off-target report: same spec, same transcriptome, 42 FAIL → 42 PASS with zero off-target rows, purely by narrowing the index | `test_gate.py` |
| `discrimination_margin` took its `min` over targets that *had* evidence | A required allele missing from a context **raised** the margin (6.00 vs 5.67) instead of zeroing it, and the cross-context `min` then hid the gap | `test_metrics.py` |
| `validate_coverage` scored the `[0,1]` fallback as coverage | A stratum with no calibration data at all reported **1.000 coverage and passed** — losing calibration made the CI gate greener | `test_calibration.py` |
| The acceptance gate printed `PASS` beside a value below the threshold on the same line | The verdict applied `empirical >= nominal - 0.05` and the line printed `nominal`, so `[PASS] conformal coverage 0.889 (need 0.9)` accepted a dataset — in the one criterion annotated "bounds mean what they claim" | `test_bugfix_regressions_round8.py` |
| `BenchmarkReport.summary()` printed `1.000` for strata with **no calibration data** | The `[0,1]` fallback covers by construction, so that figure is vacuous rather than perfect. `acceptance.py` already refused to print it and said why — and the summary it *embeds* printed it two lines away | `test_bugfix_regressions_round8.py` |
| `-` and `*` were stripped from input sequences as if they were whitespace | Deleting a gap **splices**: the base before the run joins the base after it, and the tiler walks a window across a junction in no molecule. On a 400 nt target with a 30-base gap, 21 of 297 exported guides had a site occurring nowhere in the real sequence — while the same alignment's ruler *digits* were refused with a located message | `test_bugfix_regressions_round7.py` |
| The web upload decoded with `errors="replace"` | So a cp1252 FASTA the CLI refuses as unreadable ran to completion in the browser with its record name silently swapped to U+FFFD — on the one door whose users cannot read a traceback | `test_bugfix_regressions_round7.py` |
| A target shorter than the spacer vanished without a word | The run said only "no feasible panel covers t1", which reads as "t1 has no usable guides" rather than "nobody looked" — and on a multi-record FASTA shipped 271 scored candidates from the *other* record while saying it | `test_bugfix_regressions_round7.py` |
| `reference` is a reserved context key **and** a name a user may give a record | A variant record called `reference` shared a key with the reference context, so the variant's evidence overwrote the target's and every number for the REQUIRED target was computed against the variant: renaming one record took a run from PASS=96 to PASS=0 with a shortfall about a target that was never evaluated | `test_bugfix_regressions_round6.py` |
| The web form folded an upload into the paste field | So the one-source guard could not see it, and a filled textarea silently discarded an uploaded FASTA — the exact behaviour the guard was added to stop, on the door it was added for | `test_bugfix_regressions_round6.py` |
| Measured kinetic constants killed the run | `StiffKineticsError` is raised from inside stage E, past every input check, so the flag advertised as the way to unlock absolute signal answered real constants with a traceback | `test_bugfix_regressions_round6.py` |
| `--stem ../escaped` wrote outside `--out` | A stem names a file, not a path; and every other filesystem failure arrived as a pathlib traceback and exit 1 — the same code `design` returns for "no guide cleared the gate" | `test_bugfix_regressions_round6.py` |
| An `N` in the target attracted the ranking | `classify_pair` calls `N:N` a Watson-Crick pair and RNAplfold reports unpaired probability exactly 1.0 there, so an assembly gap scored a **perfect duplex with inflated accessibility** — ranks 1, 2, 3, 4 and 6 of 101 were N-bearing spacers, exported as orderable oligos | `test_bugfix_regressions_round5.py` |
| The paste door rejected FASTA the file door read | `parse_sequence_input` sniffed FASTA from the first non-blank line while `parse_fasta` skips `;` comments, so a comment-first file — including **this tool's own exported FASTA** — was refused at character one as a malformed raw sequence | `test_bugfix_regressions_round5.py` |
| Calibration seed and noise were in no recorded field | They set every conformal interval and therefore every verdict, so two runs disagreeing on fifty verdicts carried byte-identical provenance — the one thing provenance exists to make impossible | `test_bugfix_regressions_round5.py` |
| The 413 reply was destroyed by the reset it caused | Answering an oversized upload without reading the body closes the socket on a pending body, so the OS sends a TCP RST and the response dies before any browser reads it. The server logged 413; the user saw "Not found." | `test_bugfix_regressions_round5.py` |
| The validation plan cited an off-target explorer no surface drew | The report told the reader to "sequence the top off-target loci from the explorer table" while rendering no such table, though the 50 hazard-sorted rows were in the report dict all along | `test_bugfix_regressions_round5.py` |
| A `--background` record sharing a target's name **replaced** it | `Transcriptome.add` overwrites by id, so the run searched the target against a sequence that was not it — PASS 97 → 0, best coverage 0.7647 → 0.0, with nothing said. `variant_contexts` already guarded this, so the two paths disagreed | `test_bugfix_regressions_round4.py` |
| `distinct_site_ids` grouped by the record a guide was *tiled from* | Not the same question as where it binds: with two target records, guides one nucleotide apart on the REQUIRED target were both marked distinct, under a line that says shifted copies are hidden | `test_bugfix_regressions_round4.py` |
| The FR8.4 track marked guides from *other* FASTA records | `start` is an offset on the guide's own source target, and records 1+ are DESIRED targets that get tiled too — so marks were drawn at foreign coordinates on the primary's profile, or past its end and silently clipped out of the picture | `test_bugfix_regressions_round3.py` |
| `--variants` dropped every target record after the first | Records 1+ stayed in the contract and existed in no context, so `desired_utility` — weight 1.0 in the ranking — read 0.0 for every guide, and those loci stopped being searched | `test_bugfix_regressions_round3.py` |
| `--blast-db` replaced the search instead of adding to it | One index answers both "where does this bind off-target" and "does it bind its target at all", and a BLAST database can only answer the first — it never sees the run's own transcriptome. Every guide came back with no on-target evidence: **39/39 FAIL** against 28 REVIEW for the identical design without the flag | `test_bugfix_regressions_round2.py` |
| The activity model's domain was a **union** across orthologs | Length and effector were tested as independent axes, so each ortholog's published lengths were admitted for the other. Four of the five miRNA presets returned a certified PASS at a geometry nobody validated, while the note above the table promised the opposite | `test_bugfix_regressions.py` |
| `--variants` replaced the whole context list | `--background` was parsed and discarded and `--blast-db` was validated and never queried, so adding a variant to an otherwise identical request turned a report with off-target hits into a **clean** one | `test_bugfix_regressions.py` |
| `fingerprint()` serialised a `frozenset` | Enum hashing is per-process randomised, so the digest that identifies a design — carried in the ledger, the report and every export header — **changed between runs of the same design** | `test_bugfix_regressions.py` |
| The candidate budget was spent in contract order | The third of three ordinary 3 kb REQUIRED transcripts was never tiled at the shipped default, and the run reported "no feasible panel covers t3" — which reads as "no usable guides" rather than "nobody looked" | `test_bugfix_regressions.py` |
| A fixed 400 RK4 steps, with clamps | Unstable for an ordinary measured `k_on`; the per-step clamps turned the divergence into a plausible **wrong number** (+19.5%) instead of a NaN, in the one path that unlocks when real constants are supplied | `test_bugfix_regressions.py` |
| Two variants sharing a FASTA name | `Transcriptome.add` keys by name, so one silently replaced the other and conservation reported "reaches every variant" for a variant it never searched | `test_bugfix_regressions.py` |
| `RNA.pfl_fold_up` at 3 and 4 nt | A native abort, not an exception: the interpreter died with no traceback, which under `serve` would take every in-flight design with it | `test_bugfix_regressions.py` |
| `discrimination_margin` returned `inf` before reading the on-target term | The same defect the row above records fixing, surviving in the other branch: a context that had lost a required allele scored the best value the metric can take | `test_bugfix_regressions.py` |
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
| `test_webapp_concurrency.py` | a slow design must not block every other connection |
| `test_kinetics_mass_balance.py` | the ODE cannot create target it never had; constants can actually be supplied |
| `test_contrafold.py` | 399 reference folds exact, and the sharp-turn convention the fixture is blind to |
| `test_self_folding.py` | the hard-FAIL self-folding check, wobble-aware and preferring a real fold |
| `test_signal_is_not_double_counted.py` | an axis that is a rescaled copy of two others is reported, not ranked |
| `test_panel_guarantee.py` | any approximation bound the report prints is brute-force checked against the true optimum |
| `test_panel_site_consistency.py` | the panel and the report must agree on what one site is; examples still run |
| `test_preset_provenance.py` | preset invariants, the cross-reactivity claim, and miRBase verification |
| `test_effector_constants.py` | ortholog data that nothing else exercises: DR length, stem, fold, distinctness |
| `test_module_entry_point.py` | `python -m cas13guide` runs, out of process |
| `test_arc_model.py` | 133-guide parity with the reference, channel order, ensemble space, the ViennaRNA refusal |
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
