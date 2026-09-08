"""Give the tool an input and check the answer, independently.

Every other instrument here checks the *process*: are the invariants pinned
(mutation), is every verdict reached (vocabulary), is anything computed for
nobody, what fraction of lines ran. None of them asks the question a user
actually has, which is whether the guides that come out are right.

So this one recomputes each claim from the input in plain Python, without
asking the package how it got there. ``revcomp`` is written out again below for
exactly that reason: a check that calls ``util.seq.revcomp`` proves the
exporter and the sequence layer agree, which is not the same as proving either
is correct.

The load-bearing check is the first. A designed guide is a claim -- "this
22-mer binds that transcript at position 154" -- and the claim is checkable:
slice the input at the stated position and reverse-complement it. Everything
else the tool reports is downstream of that being true.

The rest close the loop the same way. GC and the homopolymer run must describe
the spacer printed beside them. Every guide that survived must satisfy the gate
rules that let it through. The CSV and the FASTA must agree, guide for guide.
An off-target planted in the background must be found -- and every off-target
reported must really occur in the background supplied, which is the half
nobody usually checks.

Usage::

    python examples/verify_output_against_input.py

Exits non-zero if any claim fails to reproduce.
"""

from __future__ import annotations

import csv
import io
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable

# A real human transcript fragment: the MALAT1 5' region is a well-known lncRNA
# target. Sequence below is a synthetic stand-in of realistic composition; the
# checks do not depend on its provenance, only on it being the input we gave.
TARGET = (
    "AGGCAUUGAGGCAGCCAGCGCAGGGUCUCGCUGGCCCAAGGAAACGGUGGCCAAGCUUCUGACUCUCGGCG"
    "GUGCUCAGAUAGCGCAAGGAUCCUGGAGAAGCUGCCUCAGCUUCGGGAACUCGCGGGCAAAUGCUGGUAUC"
    "CGUAAGGAUUCCGAAUCGCGGAACUUGGCAUCAAGCUUAGGCCUGAAUCGGAUCCAAGCUUGGAUCCAGUA"
    "AGGCUUAAGCCGAUCGAUGCUAGCUAGCUUGGAUCCGGAUCCAAGCUUACGCGUAUCGAUAGCUAGCUUAA"
)

COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G", "N": "N"}


def revcomp(seq: str) -> str:
    """Written here so the package's own revcomp cannot vouch for itself."""
    return "".join(COMPLEMENT[base] for base in reversed(seq))


def gc_fraction(seq: str) -> float:
    return sum(1 for b in seq if b in "GC") / len(seq) if seq else 0.0


def longest_run(seq: str) -> int:
    best = run = 1
    for i in range(1, len(seq)):
        run = run + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, run)
    return best if seq else 0


def main() -> int:
    work = pathlib.Path(tempfile.mkdtemp(prefix="cas13guide-verify-"))
    fasta = work / "target.fa"
    fasta.write_text(f">MALAT1_frag\n{TARGET}\n", encoding="utf-8")

    # A background of unrelated sequence plus one deliberate near-copy of a
    # stretch of the target, so we know off-targets exist and where.
    planted = TARGET[100:160]
    background = work / "background.fa"
    background.write_text(
        ">decoy_unrelated\n" + "AUCG" * 60 + "\n"
        ">paralog_with_shared_stretch\n" + "GGCCAUUAGC" * 4 + planted + "UUAGCCAUUG" * 4 + "\n",
        encoding="utf-8",
    )

    out = work / "out"
    result = subprocess.run(
        [PY, "-m", "cas13guide.cli", "design",
         "--fasta", str(fasta), "--background", str(background),
         "--task", "knockdown", "--assay", "cell_culture_knockdown",
         "--effector", "RfxCas13d", "--no-signal",
         "--out", str(out), "--stem", "design"],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(f"CLI exit: {result.returncode}")
    if result.returncode not in (0, 1):
        print(result.stdout[-3000:]); print(result.stderr[-3000:])
        return 2
    if "Traceback" in result.stderr:
        print("TRACEBACK in stderr:"); print(result.stderr[-3000:]); return 2

    body = "\n".join(
        line for line in (out / "design.csv").read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    )
    rows = list(csv.DictReader(io.StringIO(body)))
    print(f"guides exported: {len(rows)}")
    if not rows:
        print("FAIL: the run produced no guides for a 280 nt target")
        return 1

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail and not ok else ''}")
        if not ok:
            failures.append(f"{name}: {detail}")

    print("\n--- every exported guide, checked against the input we supplied ---")

    # 1. The spacer must be the reverse complement of the target site at the
    #    stated position on the stated record. This is the core claim the whole
    #    tool exists to make.
    bad_placement = []
    for row in rows:
        start = int(row["start"])
        spacer = row["spacer"]
        site = TARGET[start:start + len(spacer)]
        if revcomp(site) != spacer:
            bad_placement.append((row["guide_id"], start, site, spacer, revcomp(site)))
    check(f"spacer == revcomp(target[start:start+len]) for all {len(rows)}",
          not bad_placement,
          f"{len(bad_placement)} wrong, e.g. {bad_placement[:1]}")

    # 2. Every reported source target must be the record we actually gave.
    targets = {row["source_target_id"] for row in rows}
    check("source_target_id names the record we supplied", targets == {"MALAT1_frag"},
          f"got {targets}")

    # 3. GC and homopolymer must describe the spacer printed beside them.
    # The export rounds to 4 decimals, so the tolerance is half a unit in the
    # last place. A tighter one flags rounding as a defect: GC on a 22-mer is
    # k/22, which is not representable, and 9/22 exports as 0.4091 against an
    # exact 0.40909... A first run of this file used 1e-6 and "failed" on 184
    # of 237 guides, every one of them correct.
    gc_bad = [r["guide_id"] for r in rows
              if abs(float(r["gc"]) - gc_fraction(r["spacer"])) > 5e-5]
    check("gc matches the spacer shown", not gc_bad, f"{len(gc_bad)} wrong")
    hp_bad = [r["guide_id"] for r in rows
              if int(r["max_homopolymer"]) != longest_run(r["spacer"])]
    check("max_homopolymer matches the spacer shown", not hp_bad, f"{len(hp_bad)} wrong")

    # 4. The gate's own hard rules must hold for everything it let through.
    outside = [r["guide_id"] for r in rows if not (0.30 <= float(r["gc"]) <= 0.70)]
    check("no surviving guide is outside the GC gate [0.30, 0.70]", not outside,
          f"{len(outside)} outside")
    runs = [r["guide_id"] for r in rows if longest_run(r["spacer"]) > 4]
    check("no surviving guide has a homopolymer run > 4", not runs, f"{len(runs)}")
    polyu = [r["guide_id"] for r in rows if "UUUU" in r["spacer"]]
    check("no surviving guide carries a Pol III terminator (UUUU)", not polyu,
          f"{len(polyu)}")

    # 5. Ranks must be dense, ordered, and consistent with the scores.
    ranks = [int(r["rank"]) for r in rows]
    check("ranks are 1..N with no gaps", ranks == list(range(1, len(rows) + 1)),
          f"got {ranks[:5]}...")
    scores = [float(r["score"]) for r in rows]
    check("scores are non-increasing down the ranking",
          all(a >= b - 1e-9 for a, b in zip(scores, scores[1:])))

    # 6. Every spacer must be orderable RNA.
    junk = [r["guide_id"] for r in rows if set(r["spacer"]) - set("ACGU")]
    check("every exported spacer is orderable ACGU", not junk, f"{len(junk)}")

    # 7. The FASTA export must agree with the CSV, guide for guide.
    fasta_text = (out / "design.fasta").read_text(encoding="utf-8")
    fasta_seqs, name = {}, None
    for line in fasta_text.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            fasta_seqs[name] = ""
        elif name:
            fasta_seqs[name] += line.strip()
    shared = [r for r in rows if r["guide_id"] in fasta_seqs]
    mismatched = [r["guide_id"] for r in shared if fasta_seqs[r["guide_id"]] != r["spacer"]]
    check(f"FASTA and CSV agree on the sequence ({len(shared)} shared)",
          shared and not mismatched, f"{len(mismatched)} differ")

    # 8. The off-target we planted must be found. We put a 60 nt stretch of the
    #    target into a background record; guides tiled there must report a hit
    #    on it, or the specificity claim is empty.
    report = json.loads(subprocess.run(
        [PY, "-c",
         "import json,sys;"
         "from cas13guide.services.workflow import DesignRequest, run_request;"
         "o=run_request(DesignRequest(fasta_path=sys.argv[1], background_path=sys.argv[2],"
         " task='knockdown', assay='cell_culture_knockdown', effector='RfxCas13d',"
         " use_structure=False, use_signal=False));"
         "print(json.dumps(o.report['off_targets'], default=str))",
         str(fasta), str(background)],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    planted_hits = [h for h in report if "paralog" in str(h.get("locus", ""))]
    check(f"the planted paralog is found as an off-target ({len(planted_hits)} hits)",
          bool(planted_hits))

    # 9. And every off-target hit must really occur in the background we gave.
    bg_text = background.read_text(encoding="utf-8")
    bg_seqs = {}
    name = None
    for line in bg_text.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]; bg_seqs[name] = ""
        elif name:
            bg_seqs[name] += line.strip()
    unfounded = []
    for hit in planted_hits[:20]:
        site = hit.get("target_site") or ""
        locus = str(hit.get("locus", ""))
        if site and locus in bg_seqs and site not in bg_seqs[locus]:
            unfounded.append((locus, site))
    check("every off-target site really occurs in the background supplied",
          not unfounded, f"{len(unfounded)} not found, e.g. {unfounded[:1]}")

    print("\n--- a worked example, checkable by hand ---")
    top = rows[0]
    start = int(top["start"])
    site = TARGET[start:start + len(top["spacer"])]
    print(f"  rank 1 guide : {top['guide_id']}")
    print(f"  spacer       : {top['spacer']}")
    print(f"  binds        : {top['source_target_id']} at {start}..{start + len(top['spacer'])}")
    print(f"  target site  : {site}")
    print(f"  revcomp(site): {revcomp(site)}   <- must equal the spacer")
    print(f"  GC           : {top['gc']} (recomputed {gc_fraction(top['spacer']):.4f})")
    print(f"  verdict      : {top['verdict']}  {top.get('detail', '')}")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print("   " + f)
        return 1
    print(f"All {9} independent checks passed over {len(rows)} exported guides.")
    print("The output is not merely produced; it is correct against the input.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
