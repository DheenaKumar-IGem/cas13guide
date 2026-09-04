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
python -m unittest discover -s tests -t tests    # 930 tests
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
> verdict for **2.8%** of them. Fixed, with the structural invariant pinned in
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
    PASS-only. The rows below are candidates to triage, not a recommendation.
  ! no feasible panel covers required target(s): t_tgt
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
| ⚠️ | FR6.1–6.2 kinetics, predicted signal | `services/kinetics.py` — **constants are placeholders**; relative ranking works and the absolute figure is withheld. Supply measured values with `--kinetics-constants FILE.json` and `predicted_signal_nm` appears on every ranked row and in the CSV, blank until then |
| ✅ | FR6.3 Bayesian panel confidence | `services/panel_confidence.py` — Beta-Binomial success posterior per member from the calibration counts, combined under Fréchet bounds. Reported as an interval; a point estimate needs a measured common-cause fraction (`--panel-dependence`) and is refused without one |
| ✅ | FR7.1–7.2 conserved regions, variant robustness | `services/conservation.py` |
| ✅ | Trained activity model | `services/arc_model.py` + `services/contrafold.py` — Arc CNN on real CONTRAfold energies, 399/399 reference folds exact. `--arc` |
| ✅ | FR8.1–8.4 ranking, table, export, visual | `pipeline/f_decide.py`, `services/fasta.py`, `services/report_html.py` |
| ✅ | NFR1–NFR6 | measured at 0.2–10 s, bounded by the candidate budget rather than by target length — see [Runtime](#runtime); `serve`; located errors; provenance ledger; 375 px verified; score glossary |
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
its upper one. A low point prediction is never evidence of safety.

What holds that in place is not where intervals are *constructed* — they are
built in three modules, and construction is not what could drop the asymmetry
anyway. It is that **every benefit read goes through `benefit_lower`**: no
module anywhere reads `.lower` off an activity or expression interval directly,
so the one dangerous mistake — reading a benefit at the *upper* end of a wide
interval and calling it evidence — has no site to happen at. Hazards do read
`.upper` directly in five places, which is the safe direction and visibly so.
`tests/test_bound_asymmetry.py` checks both halves; this section used to claim
the construction mechanism instead, which was false and protected nothing.

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

## Runtime

Measured on one desktop, warm, `RfxCas13d` / knockdown, one target and no
background corpus. Re-measure on your own machine before quoting these.

| Target | Default | `--no-structure` | Candidates |
|---|---|---|---|
| 300 nt | 0.20 s | 0.12 s | 216 |
| 1 kb | 0.61 s | 0.31 s | 761 |
| 3 kb | 3.51 s | 1.88 s | 2,509 |
| 6 kb | 7.12 s | 3.91 s | 4,156 |
| 12 kb | 10.07 s | 4.88 s | 4,271 |

The shape matters more than the numbers. Runtime tracks the **candidate count**,
not the target length, and the candidate count is capped by
`SearchBudget.max_candidates` (5,000) — so it climbs to about ten seconds and
then flattens, and a 12 kb transcript costs barely more than a 6 kb one. What
you lose past the cap is tiling *density*, not coverage: each target is spanned
end to end at a widened step, and the run says so.

Where the time goes, profiled at 10 kb: ViennaRNA's `pfl_fold_up` for the
accessibility profile (3.2 s, one call over the whole transcript), the
wobble-aware scan (2.0 s), `duplexfold` for the per-guide hybridisation energy
(1.3 s), and the QC self-fold (0.6 s). Three of those four are ViennaRNA, which
is why `--no-structure` roughly halves the total and why the README quoted a
much smaller range before anyone measured a full-length mRNA: the earlier figure
was taken on targets short enough that folding was cheap.

Profiling also found work being done for nothing. Accessibility was computed for
*every* hit, and every reader of it is guarded by `is_intended` — so each
off-target hit paid a full fold over its locus's whole sequence and the result
was discarded. That is the dominant cost of a run with a real background corpus,
because a background corpus is precisely where the off-target hits are: on a
1.5 kb target against 25 background loci, **7.85 s → 3.01 s**, for a report that
is byte-identical either way.

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

1241 tests, run under `unittest` or `pytest` — both discover the same set —
and 91% line coverage. Three of them need a miRBase `mature.fa` and skip
without it, so *collected* is the figure quoted here: the passing count moves
with an environment variable, which makes it a number nobody else can
reproduce from this page.

Nine claims this section makes are **measured rather than asserted**, each by a
procedure anyone can re-run — the rule that gave `verify_against_mirbase()` its
reason to exist. A claim is worth what the procedure that reproduces it is
worth.

```bash
pytest                                                   # 1241 tests
python examples/verify_mutation_coverage.py              # 74 mutations, 74 caught
python examples/verify_vocabulary_coverage.py            # 52 of 52 things it can say
python examples/measure_coverage.py                      # 91% lines, subprocess-traced
python examples/verify_nothing_is_computed_for_nobody.py # no value computed for nobody
python examples/verify_output_against_input.py           # the guides are right
python examples/verify_the_maths_against_a_referent.py   # and so is the arithmetic
python examples/verify_the_documented_numbers.py         # and so is this page
python -m mypy                                           # py.typed, checked
```

**The prose is checked too.** Every other instrument here checks the package
against itself or against an external referent. `verify_the_documented_numbers.py`
checks the *documents* — this README and the published landing page — because
that is where this repository has drifted most often, and the drift is invisible
from inside the code. It caught all three kinds on its first run.

The test count had been stated in three places reading 1119, 1122 and 1135 at
the same time; nothing failed, because no test compares two documents. The
DR-interaction flip rate was stated in five places as 4.9%, and all five went
stale together the moment the spacer:DR scan learned G-U wobbles. And the
landing page quoted the four CRITICAL limitations as the tool's own words when
all four had been abridged — what was dropped, each time, was the remedy and
the sharpest consequence: *"Fit a real screen with `cas13guide train` before
treating a bound as a guarantee"*, *"so no specificity score on this page bounds
it"*, and the nine sequence universes actually searched. On a page whose whole
premise is that the tool tells you what it does not know, the quotes had been
softened in exactly the places that say how bad it is and what to do about it.

That last one is why this is a script rather than a lint rule about digits. A
published claim about what a tool says is checkable by running the tool, and
until something ran it, the page and the package had disagreed since the day it
was written. It also re-derives the page's worked example from the sequences the
page itself prints — the spacer against its site, the hand-drawn duplex against
both, and the GC against the spacer.

The count it pins is *collected*, not passing: three tests need a miRBase
`mature.fa` and skip without it, so the passing count moves with an environment
variable and is not a number a reader could reproduce from this page.

**The suite is mutation-checked.** `verify_mutation_coverage.py` copies the tree
to a temporary directory, applies one small *semantic* change at a time — code
that still runs and still looks plausible — and reports any the suite fails to
notice. Two things keep that honest rather than flattering. The mutations are
subtle: disabling a whole check is caught by any competent suite and proves
nothing, so half of these instead move a boundary one step, swap a bound for its
opposite, drop the finite-sample `(n+1)` correction from the conformal quantile,
or take the right operation on the wrong variable. And a survivor is confirmed
against the **whole** suite before being called a hole. None of the 74 are
caught only by the full run today, because each hint names the module *measured*
to catch it rather than the one a reader would guess — which is not always the
same module, and where they differ the difference is worth seeing. That also
took the sweep from about thirty minutes to five, and an instrument nobody runs
documents drift instead of preventing it. The second stage stays: it is what
makes "hole" a claim about the whole suite rather than about which modules
somebody chose.

It mutates the LCB/UCB asymmetry, the six named metrics, every FAIL branch in
the gate, the QC checks, the panel constraints, the candidate budget, the
out-of-domain substitution, the conformal quantile, the search-bound algebra,
the FR6.3 posterior and its Fréchet floor, the kinetics stability guard, the
wobble rule, the expression bound, the PFS exemption for an unread base, the
spacer:DR pairing rule with its threshold, the shared binding-site rule, the
world digest, the exit codes, the off-target table's ordering, a Jeffreys prior
and a reported precision.

### It could not report a hole, and twenty were hiding

This is the part worth reading before believing the paragraph above, because
for most of this repository's life the sweep was **structurally incapable of
finding anything**.

It mutates a *copy* of the tree, and the copy held `src`, `tests` and
`pyproject.toml` — while eight tests read `README.md` or `examples/`. So the
copied tree failed six tests with two collection errors before a single mutation
was applied. Its logic is: screen a mutation, and if it survives, confirm
against the whole suite — green means hole, red means caught. With the baseline
permanently red the whole-suite run was never green, so `holes` could never be
non-empty. **"0 holes" was true by construction rather than by measurement.**

Eight distinct ways it could report success or failure without measuring turned
up once that one was found. Five were live and three were latent:

| The mechanism | What it did |
|---|---|
| A red baseline | Every mutation "caught", because the suite was already failing |
| A missing anchor | `distinct-sites-best-becomes-worst` stopped running when its code moved, and the sweep still printed success |
| An ambiguous anchor | `wobble-rule-dropped` renamed a constant appearing twice — a *consistent* rename, so a no-op — and passed only because two other modules import that name and the suite died of an `ImportError` |
| A missing screening module | `pytest tests/nope.py` exits 4, which reads as "the suite went red", so a stale hint is a permanent silent false pass |
| A subprocess importing unmutated code | An editable install points at the real `src/`, so every test that shells out was passing under mutations it never ran |
| A no-op mutation | Identical to the code it replaces, so a guaranteed false *hole* |
| Stale bytecode | `__pycache__` carried into a tree whose whole purpose is to have its source edited |
| An unhandled hang | `TimeoutExpired` ended the sweep and lost every verdict after it |

All eight now fail loudly. Two of them were found only because fixing an
earlier one exposed the next, which is the argument for treating an instrument's
own failure modes as the first thing to audit.

### What was behind them

Twenty behaviours that could each be reverted without a single test noticing.
Grouped by what went wrong rather than by the order they turned up, because the
kinds repeat and the order does not mean anything.

**A fixture too small to see the difference.** The discrimination margin's `max`
over forbidden hits could become a `min` — the fixtures carry one forbidden
target, and over one element they are the same function. Desired-target
weighting could be dropped entirely, because every fixture used weight 1.0. The
panel's coverage could read the upper bound instead of the lower, which needs
two hits on one target to be visible at all: with one, the comparison is against
0.0 and both bounds clear it.

**A boundary tested away from the boundary.** The gate's coverage floor could
move by 0.05, because every gate test sat clear of it. The conformal
group-adequacy minimum could go off by one, so "one fewer than too few" would
certify. The coverage slack could double. The candidate budget's remainder could
go unspent, turning ten slots over three targets into nine.

**A claim about nothing, accepted as a claim.** `all([])` is `True`, so the
acceptance gate's own guard against an empty report — the clause standing
between "we checked six things and they passed" and "we checked nothing" — was
unpinned. Asking where else that idiom ran over a possibly-empty collection
found the same defect in the package rather than the tests:
`BenchmarkReport.coverage_ok` had **no** guard, so a benchmark that evaluated no
cases answered "yes, the bounds mean what they claim" — and two gates read it,
including `cas13guide check`, whose entire job is conformal coverage.

**An identity that was not one.** The Fréchet floor could become a mean,
reporting a panel as weaker than a guide it contains — not a bound at all. The
provenance ledger digest could depend on the order a caller wrote keyword
arguments, so two runs recording the same facts disagreed.

**A caveat that kept its words and lost its mark.** The synthetic-calibration
CRITICAL could be demoted to a warning at the source, and — separately, one
surface further out — the HTML report could stop marking CRITICAL notes at all.
Same text, same position, no mark; every test looked for the words. Severity is
not decoration here: it also sets the order, so a demotion moves that sentence
below the notes a reader is meant to skim.

**A page that is internally consistent and points the wrong way.** The
off-target explorer keeps one row per locus carrying its worst hit, then sorts
by hazard descending and truncates. Neither rule was pinned. Sorting ascending
puts the mildest first and the truncation keeps the safest, discarding exactly
the rows the table exists to show — while every individual hazard on the page
stays correct. The README tells a reader to "sequence the top off-target loci
from the explorer table".

**A status nothing asserted.** Both CLI exit codes could lie: `train` reporting
success while the gate rejected the dataset, `design` while nothing ranked. The
tests asserted 0 for success and 2 for a usage error; nothing asserted **1**,
the status meaning "completed, and the answer is no". Both were unreachable by
the sweep until the subprocess blind spot was closed, which is the clearest
argument that closing it was worth it.

**The wrong world, and the wrong reading of an empty one.** With `--variants`, a
variant context could be chosen as *the* reference — which decides the
exon-junction map handed to tiling. And a whitespace-only paste could be treated
as a supplied sequence, so a user pasting spaces got a parse failure about a
sequence they did not knowingly give, instead of "No sequence supplied. Paste a
sequence, upload a FASTA file, or pick a built-in target."

All twenty are pinned in `tests/test_mutation_holes.py`, each verified to fail
against the mutation it closes. Where the fix was mine, that verification caught
me twice: a test asserting the wrong offset passed under the mutation it was
written for, and a first attempt at the reference-context test asserted against
evidence the choice does not actually affect.

### Two things it found that were not holes

A mutation the suite cannot catch is not always a gap. The FM index documents
two independent defences against a match spanning two loci — the separator is
excluded from the walk alphabet, and `_locate` rejects a window running off the
end of its locus — and says the redundancy is deliberate. Neither is
individually mutable while the other stands, which is what defence in depth
means. The answer was to test `_locate` directly, at the offset that separates a
correct guard from one loosened by an off-by-one, so the redundancy is verified
rather than assumed. A probe into `features.py` was the same shape with no such
payoff — its guard only matters at a spacer length of 1 and the contract admits
22 to 30 — so it was removed and the reasoning left in its place.

**Everything the tool can say to a user is said by a test.**
This package has four vocabularies for talking to a user: `ReasonCode`, what the
gate says and what a caller switches on; `Limitation`, what the run admits it
could not establish; `QCFlag`, what is wrong with a spacer; and `ReviewTier`,
how much of a human a REVIEW needs. `verify_vocabulary_coverage.py` records
which values of each the suite actually produces — at the gate, in the
orchestrator's budget shortfalls, in `limitations_for_run` and in
`qc_candidate` — and reports any that never come out. A value nothing produces
is a sentence a user can receive that nobody has watched it say: either dead
code, or a live path with no coverage.

It found five of the second kind. Three were verdicts: two reachable only once
the spec sets a floor or a window — and `--min-accessibility` is a shipped flag,
so turning it on ran code no test had executed — and one needing no flag at all,
sitting in the narrow band just under the hazard budget. Two were limitations.
The FR6.3 thin-calibration warning, which tells a reader an interval is wide
from missing data rather than from a doubtful guide, needed a confidence model
no fixture supplied; the unverified-preset warning cannot fire at all while
`PRESETS_VERIFIED` is `True`, and is exactly what a contributor meets the moment
they add a preset they have not checked. All five were correct — which is the
usual outcome, and not a reason to skip the check, because nobody knew. The QC
flags and the review tiers came back clean, all thirteen, which is a result
worth having rather than a wasted question.

It measures whether a value is *produced*, not whether anything asserts it: a
floor, in the way line coverage is a floor. The mutation check above is what
asks whether behaviour is pinned.

**The arithmetic is right too, against something that is not this package.**
Round eleven's critic pointed out that eleven rounds of *consistency* work
cannot find a formula that is internally coherent, documented as exactly what it
does, tested against its own output, and wrong. Part of that gap needs a wet-lab
screen and no file here can stand in for one. But the mathematics is not in that
category: a regularized incomplete beta has a definition independent of any
implementation, split conformal has a finite-sample coverage *theorem*, and
Fréchet bounds are checkable by brute force.

`verify_the_maths_against_a_referent.py` re-derives all four by a different
method — 77 exact cases for the incomplete beta (closed forms plus the
binomial-tail identity `I_x(k, n-k+1) = P(Bin(n,x) ≥ k)`), the symmetry relation
over 35 more, 64 closed-form inverse quantiles, the conformal coverage
guarantee by simulation at three `(alpha, n)` settings, and the Fréchet bounds
against enumeration of the joint. Everything agrees to ~1e-15.

Writing it was instructive in a way worth recording. The referent was wrong
twice before the package was wrong once. A first draft integrated the Beta
density directly and reported the package off by 0.148 at `Beta(0.5, 0.5, 0.9)`
— Simpson's rule mishandling the integrable singularity at `t → 0`. A second
substituted the singularity away and then failed for large `a` instead. What
settled it needed no integrator at all: `Beta(0.5, 0.5)` is the arcsine
distribution with closed form `(2/π)·arcsin(√x)`, and the package matched it to
1e-16 both times. The file is now built only on identities that are exact, and
says so. Verified to detect a wrong number by scaling the beta by 0.1% and by
stubbing the quantile; a third injection, moving the continued fraction's
branch point, turned out to change no value at all — a convergence choice, not
a correctness one — and it was right not to be flagged.

**The guides that come out are right.** Every other instrument here checks the
*process* — are the invariants pinned, is every verdict reached, is anything
computed for nobody, what fraction of lines ran. None asks the question a user
actually has. `verify_output_against_input.py` runs a design and then
recomputes each claim from the input in plain Python, without asking the
package how it got there: `revcomp` is written out again in that file, because
a check that calls `util.seq.revcomp` proves the exporter and the sequence
layer agree, not that either is correct.

The load-bearing check is the first. A designed guide is a claim — *this 22-mer
binds that transcript at position 154* — and the claim is checkable by slicing
the input at the stated position and reverse-complementing it. Everything else
is downstream of that. The rest close the same loop: GC and the homopolymer run
must describe the spacer printed beside them, every survivor must satisfy the
gate rules that let it through, the CSV and the FASTA must agree guide for
guide, an off-target planted in the background must be found, and every
off-target reported must really occur in the background supplied. Currently 237
guides, every claim reproduced. Verified to detect a wrong answer by shifting
the reported position by one, by exporting a constant GC, and by skipping the
reverse-complement — all three caught.

**Nothing is computed for nobody.** `verify_nothing_is_computed_for_nobody.py`
sweeps every dataclass field and reports any the package constructs and then
reads nowhere. `GateResult.detail` was the first: eight places in the gate
compose a numeric explanation for every verdict and nothing read one. Five more
came out of the sweep, including the three things QC measures about every
spacer, the limitations that never reached the machine-readable report, and the
absolute signal figure `--kinetics-constants` was advertised as unlocking.

The rule is stricter than it first looks. An earlier cut excluded each field's
home module and returned 138 candidates, nearly all read a few lines below
their own declaration; the rule that describes the defect is that the value is
read by *nothing, anywhere*. It errs in both directions and the file says how —
a field consumed through `asdict` reads as unused, and a field exported under
its own name keeps a matching string constant after its last reader goes. It is
worth running because five real defects came out of it, not because a clean run
proves anything.

**`py.typed` means what it says.** The marker has shipped in the wheel since the
first commit, telling every consumer's type checker to trust these annotations,
and no checker had ever been run: 21 errors in ten files. Most were imprecision
— but imprecision is what the marker promises there is none of, and one was a
real inconsistency in the CIGAR builder that worked only because a `str` Enum
formats as its value.

**The suite itself can fail.** Nine tests that could not have failed turned up
across the nine passes — and two of the nine were written *by* the pass hunting
them. They recur because a green test is indistinguishable from a green test
until you break the thing it names. `tests/test_the_suite_can_fail.py` now
checks the three shapes that have actually occurred here: a `__main__` guard
above a class, so `python tests/x.py` silently runs a subset (`test_blast.py`
ran 21 of 27 that way); a test with no assertion at all; and `assertTrue` on a
class attribute, which for a `slots` dataclass is a `member_descriptor` that is
unconditionally truthy. Nothing static catches every vacuous test — this catches
the ones that have happened, which is the difference between a lesson and a
guard.

**The measured figures are recomputed.** Nine stale numbers have been corrected
across the passes — a value measured once, quoted in a comment, and never
checked again while the code moved. `tests/test_measured_claims.py` recomputes
the load-bearing one: the self-folding proxy's recall, which justifies a hard
FAIL that cannot be bought back and is quoted in four places including a
limitation a user reads. Recomputing it found the quoted figures rested on a
sample too small to support them — at 6,000 random 22-mers only ~53 sequences
genuinely fold, so 28.9% was really 34.9% ± 3.7% at ten times the sample, and
"roughly half its flags" was 21%. Asserted as intervals, because a test
demanding three decimals of a Monte Carlo quantity is a flaky test dressed as a
precise one. Two further figures recomputed clean and are now pinned rather
than merely believed: the signal-degeneracy ratio that justifies reporting
`predicted_signal` without ranking on it (0.161% against a stated 0.159%), and
the DR-interaction verdicts the superseded RfxCas13d direct repeat flipped —
recovered from the first commit and re-measured at 4.64% against a stated 4.9%,
well inside sampling noise at that sample size. (That pair later moved together
to 2.8% when the spacer:DR scan learned G-U wobbles and its threshold was
re-derived with it. The repeat did not change; the ruler did, and the pinned
figure moving with it is the guard working rather than failing.)

**The invariants hold on generated input.** `tests/test_properties.py` states 40
of them for `hypothesis`, which was a declared dev dependency that nothing
imported and that was not even installed. `revcomp` is its own inverse, a spacer
against its own target is a perfect duplex, hazard is monotone in every factor,
a smaller alpha never buys a narrower interval, the FR6.3 point estimate never
leaves the interval it is printed beside, the scanner and the FM index return
the same hit set on any generated world, and — the rule seven passes kept
finding new ways to break — an accepted sequence comes back unchanged.

The last seven are the claims the decision layer rests on, which need a
generator because each has to hold for *every* weighting a user might supply
rather than for the handful somebody wrote down: no weighting resurrects a
failed guide or changes a verdict, hazard is the worst context and never the
mean, robustness is the lowest coverage and not the reference one, and
evaluating one more context can only make a guide look worse. Each was checked
by injecting its own defect into the source and watching the property go red —
which found two of the properties wanting rather than any defect in the
package, both since restated.

**Coverage is 91%, and the instrument needs checking before the number does.**
`measure_coverage.py` runs it, because the recipe is not the obvious one and
each way of getting it wrong understates precisely the modules the tracing was
turned on for. Several tests drive the CLI through `subprocess`, which
`coverage` does not follow unless the child is told to trace itself, so a plain
run reports `cli.py` at 6%; traced, it is 90%. In parallel mode each process
writes its data file beside the *working directory*, and several of those tests
run the CLI inside a temporary directory that is deleted straight after — their
coverage goes with it, and the total reads 88%. And `source` must be a package
name: as a relative path it means nothing in a child running elsewhere, and as
an absolute one it matches no module and collects nothing at all.

`_contrafold_kernel.py` reports 2% because it is `@njit` and numba-compiled
code is not traced at all. That one is a real artefact, and acting on it would
mean writing tests for code that is already tested. `demo.py` sat in the same
list at 20% and was **not** an artefact: nothing ran the demo, only
`demo --help`. It is 98% now, and the run is checked against the transcript
this README quotes.

### What nine review passes found

A hundred and fifty-five of them are regression tests from nine review passes
over the finished code, each written against a defect the original suite did not
cover, and each failing against the code as it stood. A further sixty came from
the instruments in the section above — properties, packaging, and the two
subcommands nothing had run. **61 defects** were fixed, in batches of
**23, 4, 2, 6, 9, 8, 3, 4, 2** — and the shape of that curve is the useful part.

Passes one to three *read* the code, module by module and then diff by diff, and
converged at 2. Passes four to six *ran* it — flag combinations rather than one
flag at a time, input fuzzing, determinism across process boundaries, the three
output surfaces compared field by field, real HTTP POSTs, the `train` and `demo`
subcommands nobody had exercised, the `--out`/`--stem` paths — and the count went
straight back up. Reading a module tells you whether it is self-consistent; only
running the thing tells you whether two modules agree, and nearly everything left
was two modules disagreeing. Passes seven to nine turned the instruments on the
project's own machinery: the docstrings audited claim by claim, a sweep for tests
that cannot fail, and the biological constants checked against each other.

Each pass re-read the previous pass's diff, and **three caught a regression the
previous pass had introduced** — which is the argument for doing that, and also
its limit.

Two later instruments found one more defect each, outside the passes: installing
the wheel into a clean interpreter (`cas13guide demo` had never worked for anyone
who did not clone the repo), and the type check above.

**Six instruments found nothing that survived verification**, which is worth as
much as a finding: differential testing across all four `SearchIndex`
implementations, property testing of the gate → Pareto → preference order, the
threaded server under genuine concurrent load, mutation testing, an audit of the
units and directions every quantity travels in — the error class that produces
confident, in-range, wrong numbers — and a review of the packaging change itself.

The defects worth naming, because they were live rather than hypothetical:

| What was wrong | Why it mattered | Pinned by |
|---|---|---|
| The FM walk branched over a hard-coded `ACGU` | `N` is a legal base, so any window overlapping an assembly gap was unreachable — the index returned a **smaller hit set than the scanner while still reporting `exhaustive=True`**, and a 1-mismatch paralog off-target simply vanished | `test_search.py` |
| Universe adequacy compared the *declaration* against the assay, never the search | An index built over fewer universes than the spec declares produced a clean off-target report: same spec, same transcriptome, 42 FAIL → 42 PASS with zero off-target rows, purely by narrowing the index | `test_gate.py` |
| `discrimination_margin` took its `min` over targets that *had* evidence | A required allele missing from a context **raised** the margin (6.00 vs 5.67) instead of zeroing it, and the cross-context `min` then hid the gap | `test_metrics.py` |
| `validate_coverage` scored the `[0,1]` fallback as coverage | A stratum with no calibration data at all reported **1.000 coverage and passed** — losing calibration made the CI gate greener | `test_calibration.py` |
| `cas13guide demo` **tracebacked for every installed user** | `_demo` put `<cli.py>/../../examples` on `sys.path` and imported `demo_knockdown`; from site-packages that resolves to `Lib/examples`, which does not exist, because `examples/` is not in the wheel. A subcommand named in `--help` died with `ModuleNotFoundError` and exit 1 for anyone who did not clone the repo | `test_installed_package.py` |
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

### A twelfth, over the code eleven rounds barely touched

Round eleven's critic named what its dimensions never reached: the learning half
behind `train` and `check` — ingest, training, acceptance, benchmark,
calibration, features, about 1,900 lines — and stages B and F. Eleven verified,
**three confirmed**, all three high.

Two of them are the class the critic said this method is *structurally unable*
to find: a defect where every surface agrees and all of them are wrong.

| What was wrong | Why it mattered | Pinned by |
|---|---|---|
| A duplicate CSV header silently inverted every training label | `csv.DictReader` keeps the **last** column of a repeated name, and the required-column check tested membership only. A screen with two `activity` columns — an ordinary artefact of a join or a merged replicate export — was read from whichever came second. Measured on a 672-row screen whose stray column held the negated value: **every fitted weight flipped sign** (`gc` +0.278 → −0.278) while **all six acceptance numbers stayed identical**, because rank normalisation is invariant under exact inversion. Both runs printed `dataset ACCEPTED`. This is the failure `ingest.py`'s own docstring calls fatal, and no printed surface distinguished the inverted run from the correct one. The same hole on the *grouping* column is worse than an inversion: grouping by a duplicate holding a plate id puts the same transcript on both sides of the split — the leak that column exists to prevent | `test_bugfix_hunt_round12.py` |
| A stray double-quote crashed `train` with a traceback and exit 1 | `csv` consumes to EOF looking for the closing quote, so any file past the 128 KB field limit raises `_csv.Error` — converted nowhere, while `cli._train` catches `DatasetError` alone. Exit 1 is what `train` uses for "loaded, and rejected by the acceptance gate", so a malformed file and a rejected dataset were indistinguishable to anything reading the status | `test_bugfix_hunt_round12.py` |
| A WARNING restated a measured fact as a different, false one | `b_qc`'s docstring is right — "only 21% of its flags really do fold, four in five are false", where *fold* means reaching −8 kcal/mol. The user-facing limitation paraphrased that as "four in five of the guides it does reject fold no more tightly than **average**". Not the same claim, and false: re-measured at the sample and seed `b_qc` itself cites, flagged spacers average **−5.564 kcal/mol against a population −1.861**, and only **9.7%** fold no more tightly than that mean. The proxy is over-strict, not arbitrary. This reaches stdout and the HTML report on every run without ViennaRNA, which is the default install | `test_bugfix_hunt_round12.py` |

### An eleventh, aimed at the tenth's own diff

Three of round ten's findings were regressions from round nine, so this round
pointed a finder straight at round ten's changes and spent the rest on places no
round had covered: BLAST coordinate handling, the preset tiling path, the
browser door, the panel posterior, conservation. Fourteen candidates, ten
refuted, **four confirmed**. The count is falling — 8, then 11 with the critic,
now 4 — but one of the four is the worst single defect in the project's history,
and it was in the feature this branch exists to add.

| What was wrong | Why it mattered | Pinned by |
|---|---|---|
| **FR6.3's headline number was conditioned on the wrong point of the scale** | `for_panel` passed the conformal *lower bound* to `guide_success`, which selects calibration rows by `abs(predicted - x) <= bandwidth` — but `add_observation` stores the model's raw point prediction, so the window landed a full conformal quantile below the rows it was meant to sit among. The module docstring states both halves of why that is wrong in its first paragraph: the rows are the ones "whose prediction sits near this guide's", and "nothing here reads a conformal quantile". On a perfectly calibrated screen with q = 0.30, a guide predicted at 0.60 — a level where **every** calibration row cleared the threshold — came back with posterior mean **0.0012 instead of 0.9988**. The answer inverted, on the number the panel floor and every printed `success_mean` rest on | `test_bugfix_hunt_round11.py` |
| A single IUPAC code in a BLAST subject aborted the run with a bare `ValueError` | Refusing is right — skipping the row would hide a real off-target behind an ambiguity code. The defect was the exception *type*: blastn exiting non-zero already raises `BlastUnavailable`, which becomes exit 2, while this escaped as a traceback with exit 1. RefSeq, nt and TSA routinely carry R/Y/M/K/S/W/B/D/H/V, and the user cannot edit a third-party database | `test_bugfix_hunt_round11.py` |
| `serve --port 0` printed and opened an address that cannot work | The URL was built from the *requested* port. `--port 0` is the conventional "pick a free one"; the server bound an ephemeral port and announced `http://127.0.0.1:0/`, with the working address appearing nowhere. This is the door aimed at people who will not use a command line | `test_bugfix_hunt_round11.py` |
| `--variants` rewrote the recorded genome build and annotation | Every variant world was built as `Transcriptome("variants", record.name)`, and that version is what the ledger files for each context — so a flag about isoforms discarded the user's `--genome-build` and `--annotation`, including for the context named `reference`, and filed a FASTA header in the annotation slot | `test_bugfix_hunt_round11.py` |

Its critic then found a fifth, and said something about the method worth
recording. Asked whether eleven rounds are converging, it answered yes — falling
yield and falling severity, with confirmed findings clustering in code written
in the previous two or three rounds — and then named the limit:

> This is a **consistency** method, not a **ground-truth** method. It is
> structurally unable to find a defect where every surface agrees and all of
> them are wrong: a formula that is internally coherent, documented as exactly
> what it does, tested against its own output, and scientifically incorrect.
> Nothing in this repo compares a number to an external referent.

That is accurate and worth stating plainly. `verify_output_against_input.py`
closes part of it — it checks the guides against the input rather than against
the package's own account of them — but nothing here validates the *science*
against a measured screen or a published dataset, and mutation coverage
measures whether the tests pin the code, not whether the code is right.

The critic also named a blind spot the method *creates*: because the standard
fix for two surfaces disagreeing is to add a caveat, a flag or a probe, the repo
accumulates **guards that nothing reads**. Then it found one.

| What was wrong | Why it mattered | Pinned by |
|---|---|---|
| A saturated assay switched the signal axis **on** | `SignalPrediction.saturated` was computed on every prediction and read by nothing in `src/` — while `PROVISIONAL_CONSTANTS`' own comment names it as the check that decides this. The gate that did run measures a ratio spread, which cannot tell an assay bending because its *activation* curve bends (informative) from one bending because the *reporter is exhausted* (no ordering at all). With measured constants that exhaust the reporter: all 323 guides at 499.9999 nM of a 500 nM reporter, a `relative_signal` spread of **2.5e-08** that `_normalize` stretches to fill [0, 1] — so an axis weighted 2.5 contributed up to **22% of an 11-point score out of float noise**, and **309 of 323 guides changed rank** | `test_round11_critic_findings.py` |

The sweep that should have caught it could not: it walked dataclass *fields*,
and `saturated` is a property — which is exactly the shape a computed-but-unread
guard takes. It now walks properties too, and found one more the same hour
(`SearchOutcome.universe_adequate`, a dead alias for a check the gate makes
directly, now removed).

### A tenth pass, run as eight parallel hunts

Eight finders over eight dimensions, each finding put to two skeptics told to
refute it by default. Twenty-three candidates, fifteen refuted, **eight
confirmed** — and three of the eight were regressions introduced by the round
immediately before it, which is both the argument for adversarial verification
and its limit: a pass that only re-reads its own diff finds what it already
expects.

The eight share a shape. Every one is a **disagreement between two descriptions
of the same run** — a note saying a record was ignored beside a metric computed
from it, a table omitting the hazard the verdict rested on, a glossary denying a
figure the CSV carries, a count that moves with a display flag. Not one is a
wrong formula. The arithmetic here has been checked to death; what had not been
checked was whether the package's own accounts of itself agree.

| What was wrong | Why it mattered | Pinned by |
|---|---|---|
| A `--background` record the run said it ignored was searched anyway — but only with `--variants` | The collision guard filtered the reference world and the **unfiltered** list went to the variant contexts, where the name no longer collides. Adding `--variants` to an otherwise identical request took `worst_off_target` from 0.000 to **0.571** while the note kept saying the copy was ignored. A background corpus pulled from the same annotation routinely contains the target's own id | `test_bugfix_hunt_round10.py` |
| A guide could FAIL for a hazard the report never displayed | `worst_off_target` is a max across contexts; the off-target explorer read the reference alone. A hit living only in a haplotype drove the metric and the verdict and appeared in no row — under a validation plan that says "sequence the top off-target loci from the explorer table". Measured: 18 guides judged on a hazard, **0 rows** | `test_bugfix_hunt_round10.py` |
| A trained model was asked a question its domain had already refused | `evidence_for_hit` computed `in_domain`, called `predict()` anyway, and discarded the result three lines later. Harmless for the built-in heuristic; a trained model's FeatureSpec is fixed at the screen's spacer length, so fitting at 22 nt and designing at 23 nt — **both canonical for RfxCas13d** — crashed the run on a prediction nothing would read | `test_bugfix_hunt_round10.py` |
| An unreadable input path escaped as a traceback and exit 1 | `read_fasta` caught only `UnicodeDecodeError`. A tab-completed `--fasta data/` gave a stack trace and the status `design --help` documents as "the run completed and nothing survived the gate" — so a wrapper read a typo as a legitimate empty result | `test_bugfix_hunt_round10.py` |
| `cas13guide demo` dropped the CRITICAL that qualifies every number it prints | `limitations_for_run` reads `calibration.synthetic`; the design path sets it and the demo's own stand-in never did. The demo printed three of its four CRITICALs, omitting the one saying the calibration is synthetic and every PASS therefore exercisable rather than earned — from the terminal **and** from the `report.json` it writes. The commit that added that section had named this exact caveat as the thing that was missing | `test_bugfix_hunt_round10.py` |
| `--top 0` reported a count from the display list | It said "N guide(s) ranked … the exports carry them all" while pointing at a CSV holding a different number, and N moved with `--all-sites` — a display flag deciding a stated fact about the run, days after a sibling fix stopped one deciding the exit status | `test_bugfix_hunt_round10.py` |
| The HTML denied an absolute figure the same run's CSV carried | The Signal glossary was a module constant, so it told every reader "the constants shipped by default are placeholders, so no absolute figure is offered" — including on a run that supplied measured constants, whose CSV carries `predicted_signal_nm` and whose limitations correctly drop the placeholder warning | `test_bugfix_hunt_round10.py` |

Then a ninth agent was asked a different question — not "is there a bug here"
but "what did those eight dimensions *not* look at". It named six places, went
and checked the most promising, and three of its leads were real. The first is
the worst thing this round found.

| What was wrong | Why it mattered | Pinned by |
|---|---|---|
| A gene name the FASTA reader is built to accept killed the CLI | `read_fasta` decodes `utf-8-sig`, so a UTF-8 header is supported input; the record name becomes a `target_id` the CLI prints to a stream left at the platform default. On Windows that is cp1252 and the print raises — on **NF-κB1, IFN-γ1, TGF-β1, HIF1-α**, which is how those genes are written. And it landed after the design succeeded and before the `--out` block, so a run with **159 PASS and a full panel wrote no files and returned 1**, the status documented as "nothing survived the gate". The browser door rendered the same bytes fine, so the two doors that "cannot drift" did | `test_round10_critic_findings.py` |
| Non-targeting controls were QC'd by a proxy the run itself did not use | The run QCs real guides with the ViennaRNA fold and records `qc/vienna-mfe/v1` in the ledger; the control probe fell back to the stem-length proxy, which catches about a third as much. Measured over 4,000 shuffles: **226 (5.7%)** pass the control check and are FAILed by the QC that actually ran — the exact divergence the comment above that call says it removed | `test_round10_critic_findings.py` |
| A duplex fold was run for every off-target hit and discarded | `duplex_dg_per_nt` has one reader and it skips non-intended pairs; the off-target explorer has no ΔG column. The identical waste was removed for `accessibility` in the field beside it and left here. The report is byte-identical without it, and the discarded share grows with the background corpus while the intended folds stay fixed | `test_round10_critic_findings.py` |

### What came after the passes

The nine passes read the code and ran it. A later round asked a different
question — *what does this package compute, or promise, that nothing ever
reads?* — and that turned up five more live defects. Each is the same shape:
the work was done correctly and the result went nowhere.

| What was wrong | Why it mattered | Pinned by |
|---|---|---|
| The gate composed a numeric explanation for every verdict and nothing read it | `GateResult.detail` is built in eight places — "forbidden hit UCB(activity) 0.412 > 0.200" — and a grep for a *read* of it across `src/` returned nothing at all. A run reported **`FAIL: 41` and offered no way to learn what was wrong with any of the 41**: `ranked` holds no FAIL rows by design, so that was the only place a reason could have surfaced | `test_rejection_reporting.py` |
| A threshold the user set could not be applied, and the run did not say so | `--min-accessibility 0.9` without ViennaRNA asks for a gate that cannot fire — the metric is `None`, so the check is skipped, correctly. The run then emitted the **same generic warning it gives a user who never mentioned accessibility**, so the one reader who needed to know their floor went unused got the sentence written for the reader who did not | `test_inert_thresholds_are_reported.py` |
| No export said where a guide binds | `source_target_id` was added to the report dict when the accessibility track was found drawing guides from one record at another's coordinates — and it reached the track and nothing else. On a multi-record design the CSV, the FASTA and the HTML table each named the guide, its score and its verdict, and **never which target it was for**; two guides at the same coordinate can be on different records | `test_exports_place_the_guide.py` |
| A display flag decided the exit status | The status came from the list the terminal table prints, after the distinct-site filter and `--top`. So `--top 0` **exited 1 and printed "no guide cleared the gate" beside a CSV holding 322 of them** — and 1 is the code a wrapper script reads as "the run refused everything" | `test_exit_codes_are_a_contract.py` |
| The advertised kinetics unlock reached no surface a user reads | `absolute_signal()` is a library method and **nothing in `src/` called it**: the pipeline asked `predict()` for a figure, kept the relative score and discarded the nM from the same object. So supplying measured constants — the work the README names as unlocking an absolute figure — changed a warning and produced no number anywhere | `test_kinetics_mass_balance.py` |
| The caveats reached the reader and not the machine — and not the demo at all | Four CRITICAL limitations apply to an ordinary run, including that the calibration is synthetic so **every PASS is exercisable rather than earned**. `report.json` had no limitations key at all, so a pipeline saw PASS verdicts and calibrated-looking intervals with nothing to qualify either — the one surface where a caveat cannot be inferred from tone. And `cas13guide demo`, the first command anybody runs, printed none of them anywhere: a grep of its whole output for "critical", "warning" or "limitation" returned zero | `test_caveats_reach_every_surface.py` |
| QC measured three things about every spacer and reported none of them | Of `QCResult`'s fields only the flags, `manufacturability` and `pfs_preference` were read anywhere. **`gc`, `max_homopolymer` and the self-fold proxy were computed per candidate and dropped** — and GC fraction is the first number anybody ordering an oligo writes down, with the gate FAILing outside [0.30, 0.70], so where a surviving guide sits inside that band was exactly what a reader wanted and could not get | `test_qc_measurements_reach_the_user.py` |
| The demo was never run, only `demo --help` | `demo.py` sat at 20% *with* subprocess tracing on, so the first command a new user types could have raised on line one with the suite green. The README quotes its transcript, and two of the quoted lines had drifted from what it prints | `test_demo_runs_and_matches_the_readme.py` |

Five more were coverage gaps rather than defects: three gate verdicts and two
limitations that no test had ever produced, found by the vocabulary check
described above. All five were correct, which is the usual outcome and the
reason to run the check anyway.

Four further instruments found nothing, which is worth recording alongside what
they did not find: the three input doors (`--fasta`, a paste, a browser upload)
produce byte-identical reports down to the provenance ledger; every design flag
that changes no exported artefact turned out to have a documented reason;
sixteen full runs across four `PYTHONHASHSEED` values produced identical files;
and every report key the HTML and CSV read is one the report produces.


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
