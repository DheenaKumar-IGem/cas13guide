"""Check what the documents claim against what the package does.

Every other instrument here checks the package against itself or against a
referent. This one checks the *prose* -- the README and the published landing
page -- because that is where this repository has drifted most often, and the
drift is invisible from inside the code.

Three separate sessions of it, all the same shape:

* the test count was stated in three places and they read 1119, 1122 and 1135
  simultaneously. Nothing failed, because no test compares two documents;
* the DR-interaction flip rate was stated in five places as 4.9%. When the
  spacer:DR scan learned G-U wobbles the true figure became 2.8%, and all five
  went stale together;
* the landing page quotes the four CRITICAL limitations as what the tool says
  "in its own words". All four had been abridged, and what was dropped was the
  remedy and the sharpest consequence each time -- "Fit a real screen with
  `cas13guide train` before treating a bound as a guarantee", "so no
  specificity score on this page bounds it", the nine universes actually
  searched. On a page whose whole premise is that the tool tells you what it
  does not know, the quotes had been softened in exactly the places that say
  how bad it is and what to do about it.

The last one is why this file exists rather than a lint rule about numbers.
A published claim about what a tool says is checkable by running the tool, and
until something ran it, the page and the package had been disagreeing since the
day it was written.

Usage::

    python examples/verify_the_documented_numbers.py

Exits non-zero if any documented claim fails to reproduce.
"""

from __future__ import annotations

import html
import pathlib
import re
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

README = ROOT / "README.md"
PAGE = ROOT / "docs" / "index.html"

#: The target the page's worked example was designed from.
TARGET = (
    "GGACAUCGAAUCGCGGAACUUGGCAUCAAGCUAGCUAGGCAUCGAUCGAUCAGCUAGCAUCGAUCGGAUCC"
    "AUGGAUUCCAGGCACUUCUCGAUCAUAGGACUCGGUAAUGCCAUGAGUCGAUGGACUUCAGUGCCAUCGAU"
    "CAGCUUAAGGCAUCGAUCGGAUCCAUGGAUUCCAGGCACUUCUCGAUCAUAGGACUCGGUAAUGCCAUGAG"
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    return ok


def normalise(text: str) -> str:
    """One spelling for a sentence, whatever surface it was written on.

    The terminal prints ``--`` and ``*emphasis*`` and backticked code; the page
    writes an em dash, ``<em>`` and ``<code>``. Those are the same sentence, and
    a check that called them different would be a check nobody could keep
    green. Everything else -- a dropped clause, a changed number, a softened
    consequence -- survives this normalisation and is caught.
    """
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    text = text.replace("—", "--").replace("–", "--")
    text = text.replace("*", "").replace("`", "")
    return " ".join(text.split())


def documented_test_counts() -> dict[str, int]:
    """Every place that states how many tests there are."""
    readme = README.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    found: dict[str, int] = {}
    prose = re.search(r"^(\d{3,5}) tests, run under", readme, re.M)
    if prose:
        found["README prose"] = int(prose.group(1))
    block = re.search(r"^pytest\s+#\s*(\d{3,5}) tests", readme, re.M)
    if block:
        found["README command block"] = int(block.group(1))
    cell = re.search(r'<td class="est">([\d,]+) tests', page)
    if cell:
        found["landing page"] = int(cell.group(1).replace(",", ""))
    return found


def collected_test_count() -> int:
    """What the suite actually holds, by the runner that needs no plugin.

    Collected rather than passing: three tests need a miRBase ``mature.fa`` and
    skip without it, so the passing count moves with an environment variable
    and is not a number a reader could reproduce from the page.
    """
    sys.path.insert(0, str(ROOT / "tests"))
    suite = unittest.TestLoader().discover(str(ROOT / "tests"),
                                           top_level_dir=str(ROOT / "tests"))
    return suite.countTestCases()


#: Files that state the DR-interaction flip rate. The figure lived in five
#: places and all five went stale together when the spacer:DR scan learned G-U
#: wobbles: the number describes the *ruler*, not the direct repeat, so nothing
#: about the repeat changing was what moved it.
FLIP_RATE_SOURCES = (
    "src/cas13guide/contract/effectors.py",
    "tests/test_effector_constants.py",
    "tests/test_measured_claims.py",
    "README.md",
)

FLIP_RATE_PATTERN = re.compile(
    r"(?:changed|flipped)\s+(?:that|the DR-interaction)\s+verdict\s+for\s+[*]*"
    r"(\d+\.?\d*)%"
)

#: The corrupted RfxCas13d repeat as it shipped in 8cbe60f, recovered from the
#: first commit. The flip rate is measured against this.
SUPERSEDED_REPEAT = "CAACUAAACCCUAUAAGAAACGAAAUGAGAACCCAAUUCAUCGCU"


#: How many places state the size of the mutation set. Pinned so a document
#: cannot drop out of the comparison by being reworded past its regex.
EXPECTED_MUTATION_SOURCES = 8


def mutation_set_size() -> int:
    """How many mutations the sweep actually carries, from the sweep itself."""
    sys.path.insert(0, str(ROOT / "examples"))
    import verify_mutation_coverage

    return len(verify_mutation_coverage.MUTATIONS)


def stated_mutation_counts() -> dict[str, int]:
    """Every figure in the documents that is a count of the mutation set.

    Three of them, and one is a denominator rather than a headline -- "10 of
    the 32 are caught only by the full run". Adding a mutation moves all three,
    and the denominator is the one a writer forgets, because it does not look
    like the number they came to change.
    """
    readme = README.read_text(encoding="utf-8")
    page = PAGE.read_text(encoding="utf-8")
    found: dict[str, int] = {}
    block = re.search(r"verify_mutation_coverage\.py\s+#\s*(\d+) mutations,\s*(\d+) caught",
                      readme)
    if block:
        found["README command block (total)"] = int(block.group(1))
        found["README command block (caught)"] = int(block.group(2))
    denominator = re.search(r"of the (\d+) are caught\s*\n?\s*only by the full run",
                            " ".join(readme.split()))
    if denominator is None:
        denominator = re.search(r"of the (\d+) are caught only by the full run",
                                " ".join(readme.split()))
    if denominator:
        found["README full-run denominator"] = int(denominator.group(1))
    cell = re.search(r"(\d+) semantic mutations,\s*(\d+) caught", page)
    if cell:
        found["landing page (total)"] = int(cell.group(1))
        found["landing page (caught)"] = int(cell.group(2))

    # The sweep's own docstring, which is the source a reader reaches when they
    # want to know whether to believe the sweep. It carried a fourth number:
    # 33, while the README said 32 twice and the page said 32, and the list
    # itself held 32. Nothing compared any of them.
    sweep = (ROOT / "examples" / "verify_mutation_coverage.py").read_text(encoding="utf-8")
    header = sweep[: sweep.index('"""', 3) + 3]
    measured = re.search(r"Last measured:\s*[*]*(\d+) mutations,\s*(\d+) caught", header)
    if measured:
        found["sweep docstring (total)"] = int(measured.group(1))
        found["sweep docstring (caught)"] = int(measured.group(2))
    denominator = re.search(r"of the (\d+) are caught", " ".join(header.split()))
    if denominator:
        found["sweep docstring denominator"] = int(denominator.group(1))
    return found


def stated_flip_rates() -> dict[str, float]:
    """Every place that states how often the wrong repeat moved a verdict.

    One value per file, and a file that states it twice differently is itself
    a disagreement -- so those are surfaced rather than overwritten. Keeping
    only the last match would have hidden exactly the drift this checks for,
    inside a single document.
    """
    found: dict[str, float] = {}
    for rel in FLIP_RATE_SOURCES:
        raw = (ROOT / rel).read_text(encoding="utf-8")
        raw = re.sub(r"(?m)^\s*>\s?", " ", raw)          # markdown blockquotes
        values = {float(v) for v in FLIP_RATE_PATTERN.findall(" ".join(raw.split()))}
        if len(values) == 1:
            found[rel] = values.pop()
        elif values:
            # Deliberately not a single number: the caller compares the values
            # across sources, and a file at odds with itself must not be able
            # to look like agreement.
            found[f"{rel} (states {len(values)} different figures)"] = min(values)
            found[f"{rel} (and)"] = max(values)
    return found


def measured_flip_rate(seed: int = 12345, n: int = 20_000) -> float:
    """Re-measure it, rather than trusting four files that agree with each other.

    Four documents agreeing is what the stale 4.9% looked like too. The
    superseded repeat is the one recovered from the first commit; the rule and
    the threshold are whatever the package currently uses, which is the point --
    this figure is a property of the ruler as much as of the repeat.
    """
    import random

    from cas13guide.contract.effectors import EFFECTORS
    from cas13guide.contract.enums import Effector
    from cas13guide.pipeline.b_qc import MAX_DR_STEM, longest_cross_complement

    current = EFFECTORS[Effector.RFX_CAS13D].direct_repeat
    rng = random.Random(seed)
    flips = 0
    for _ in range(n):
        spacer = "".join(rng.choice("AUGC") for _ in range(22))
        before = longest_cross_complement(spacer, SUPERSEDED_REPEAT) > MAX_DR_STEM
        after = longest_cross_complement(spacer, current) > MAX_DR_STEM
        flips += before != after
    return 100.0 * flips / n


#: Messages the README quotes as the package's own output, and how to ask the
#: package for them. The index rationale in this list had gone stale -- the
#: README still showed "scan ~0s beats FM ~2800s" months after that
#: counterfactual was removed for overstating the FM cost a hundredfold. The
#: designability note beside it was still exact, so the risk is real and
#: unevenly distributed, which is the argument for checking rather than
#: reading.
def vocabulary_size() -> int:
    """How many distinct things this package can say to a user.

    The four vocabularies `verify_vocabulary_coverage.py` reports on, counted
    without running the suite: three enums and the limitation ids scraped from
    the source. The README quotes "52 of 52 things it can say", and the
    denominator is the half that drifts -- adding a reason code or a limitation
    moves it, and nothing about writing one prompts you to update a figure two
    files away. The numerator needs a whole suite run to establish and stays
    the instrument's job.
    """
    import ast

    from cas13guide.contract.enums import QCFlag, ReasonCode, ReviewTier

    tree = ast.parse(
        (ROOT / "src/cas13guide/services/limitations.py").read_text(encoding="utf-8"))
    limitations = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "Limitation"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    return len(ReasonCode) + len(limitations) + len(QCFlag) + len(ReviewTier)


def quoted_tool_messages() -> list[tuple[str, str, str]]:
    """``(label, what the README says, what the package says)``."""
    from cas13guide.contract.enums import Effector
    from cas13guide.contract.presets import designability, get_preset
    from cas13guide.services.index_selection import plan_index

    readme = README.read_text(encoding="utf-8")
    out: list[tuple[str, str, str]] = []

    # Each is a `# "..."` block whose continuation lines start `#  `.
    for label, marker, actual in (
        ("designability note", '# "24 nt target cannot hold',
         designability(get_preset("hsa-miR-21-5p"), Effector.LWA_CAS13A).note),
        ("index rationale", '# "2,391 bases x 1000 queries',
         plan_index(2391, 1000).rationale),
    ):
        start = readme.find(marker)
        if start < 0:
            out.append((label, "", normalise(actual)))
            continue
        block: list[str] = []
        for line in readme[start:].splitlines():
            if not line.startswith("#"):
                break
            block.append(line.lstrip("#").strip())
        # The README writes the message inside quotation marks; the
        # package returns it without them.
        quoted = normalise(" ".join(block)).strip().strip(chr(34))
        out.append((label, quoted, normalise(actual)))
    return out


def page_critical_quotes() -> list[str]:
    page = PAGE.read_text(encoding="utf-8")
    blocks = re.findall(r'<li class="lim">.*?</li>', page, re.S)
    quotes = []
    for block in blocks:
        if "sev-crit" not in block:
            continue
        body = " ".join(re.findall(r"<p>(.*?)</p>", block, re.S))
        quotes.append(normalise(body))
    return quotes


def tool_critical_limitations() -> list[str]:
    """What a default run actually prints, at the page's own configuration."""
    from cas13guide.services.workflow import DesignRequest, run_request

    with tempfile.TemporaryDirectory() as directory:
        fasta = pathlib.Path(directory) / "target.fa"
        fasta.write_text(f">MALAT1_frag\n{TARGET}\n", encoding="utf-8")
        outcome = run_request(DesignRequest(
            fasta_path=str(fasta), task="knockdown",
            assay="cell_culture_knockdown", effector="LwaCas13a",
        ))
    return [normalise(item["text"]) for item in outcome.report["limitations"]
            if item["severity"] == "critical"]


def page_worked_example() -> dict[str, str]:
    """The rank-1 guide the page prints, parsed back out of the page."""
    page = PAGE.read_text(encoding="utf-8")
    block = re.search(r'<dl class="kv">(.*?)</dl>', page, re.S)
    if not block:
        return {}
    pairs = re.findall(r"<dt>(.*?)</dt>\s*<dd[^>]*>(.*?)</dd>", block.group(1), re.S)
    return {normalise(k).lower(): normalise(v) for k, v in pairs}


#: Enough to count a verification block. Written out because the README writes
#: the number as a word, and a word is what a reader checks against.
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
}


def claimed_and_actual_procedures() -> tuple[int | None, int]:
    """"N claims ... each by a procedure anyone can re-run" against the block.

    The sentence said "Nine" from the commit that introduced it, when the block
    below it held eight commands. Nothing compared the two, so it stayed wrong
    until a ninth command was added for an unrelated reason and made it
    accidentally true. A number that becomes correct by coincidence was never
    a claim; this makes it one.
    """
    readme = README.read_text(encoding="utf-8")
    match = re.search(r"^(\w+) claims this section makes", readme, re.M)
    claimed = NUMBER_WORDS.get(match.group(1).lower()) if match else None

    start = readme.index("claims this section makes")
    block = readme[start:]
    block = block[block.index("```bash") + 7:]
    block = block[: block.index("```")]
    actual = len(re.findall(r"^(?:pytest|python) ", block, re.M))
    return claimed, actual


def regression_class_count() -> int:
    """Defects with a regression test class named for them.

    The page used to say "about eighty defects", which is not a number anyone
    could check -- a defect is not a countable object in a repository, and an
    approximation nobody can reproduce is the kind of claim this file exists to
    stop. One class per defect *is* countable, so that is what the page now
    says and this is what counts it.
    """
    total = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        name = path.name
        if not (name.startswith("test_bugfix") or name.startswith("test_round")):
            continue
        total += len(re.findall(r"^class Test", path.read_text(encoding="utf-8"), re.M))
    return total


def page_duplex() -> tuple[str, str]:
    """The base-paired alignment the page draws, as (top strand, bottom strand).

    The most prominent claim on the page and the easiest to let rot: it is
    hand-drawn ASCII, so nothing but this connects it to the sequences in the
    row above it.
    """
    page = PAGE.read_text(encoding="utf-8")
    block = re.search(r'<pre class="duplex">(.*?)</pre>', page, re.S)
    if not block:
        return ("", "")
    strands = re.findall(r"[53]' ([ACGU]+) [35]'", html.unescape(block.group(1)))
    return (strands[0], strands[1]) if len(strands) == 2 else ("", "")


def main() -> int:
    failures: list[str] = []

    # ---- 1. one test count, stated the same everywhere -------------------
    print()
    print("the test count, wherever it is written down")
    counts = documented_test_counts()
    for where, value in sorted(counts.items()):
        print(f"          {where}: {value}")
    if not check("every document states a count", len(counts) == 3,
                 f"found {sorted(counts)}"):
        failures.append("a document stopped stating the count")
    elif not check("they agree with each other", len(set(counts.values())) == 1,
                   "; ".join(f"{k}={v}" for k, v in sorted(counts.items()))):
        failures.append("documented test counts disagree")
    actual = collected_test_count()
    if not check(f"and they match the suite ({actual} collected)",
                 set(counts.values()) == {actual},
                 f"documents say {sorted(set(counts.values()))}, suite holds {actual}"):
        failures.append("documented test count is stale")

    # ---- 2. the page quotes the tool, not a paraphrase of it -------------
    print()
    print("the four CRITICAL limitations the page prints as the tool's own words")
    quoted = page_critical_quotes()
    printed = tool_critical_limitations()
    if not check(f"the page shows {len(printed)} of them", len(quoted) == len(printed),
                 f"page has {len(quoted)}, a run prints {len(printed)}"):
        failures.append("the page shows a different number of CRITICAL limitations")
    for quote in quoted:
        matched = quote in printed
        label = quote[:58] + ("..." if len(quote) > 58 else "")
        if not matched:
            nearest = max(printed, key=lambda t: len(set(t.split()) & set(quote.split())))
            if nearest.startswith(quote):
                detail = f"page stops early; dropped: ...{nearest[len(quote):][:120]}"
            else:
                detail = f"tool says: {nearest[:120]}"
        else:
            detail = ""
        if not check(f'"{label}"', matched, detail):
            failures.append("a quoted limitation is not what the tool prints")

    # ---- 3. the verification block is as long as it says -----------------
    print()
    print("the number of procedures the README says it offers")
    claimed, actual = claimed_and_actual_procedures()
    if not check(f"the prose count matches the block ({actual} commands)",
                 claimed == actual,
                 f"prose says {claimed}, the block lists {actual}"):
        failures.append("the README miscounts its own verification commands")

    # ---- 4. how many things the package can say --------------------------
    print()
    print("the size of the vocabulary the README quotes")
    stated = re.search(r"verify_vocabulary_coverage\.py\s+#\s*(\d+) of (\d+) things",
                       README.read_text(encoding="utf-8"))
    size = vocabulary_size()
    if not check(f"the denominator is the vocabulary's size ({size})",
                 bool(stated) and int(stated.group(2)) == size,
                 f"README says {stated.group(2) if stated else '(not found)'}, "
                 f"the package can say {size} things"):
        failures.append("the documented vocabulary size is stale")
    if not check("and it does not claim more produced than exist",
                 bool(stated) and int(stated.group(1)) <= size,
                 f"claims {stated.group(1) if stated else '?'} of {size}"):
        failures.append("the documented vocabulary coverage exceeds the vocabulary")

    # ---- 5. messages the README quotes as the package's own output -------
    print()
    print("messages the README quotes as output")
    for label, quoted, actual in quoted_tool_messages():
        if not check(f"{label} is what the package prints",
                     bool(quoted) and quoted == actual,
                     f"README: {quoted[:90] or '(not found)'}"
                     f"  ||  package: {actual[:90]}"):
            failures.append(f"the README misquotes the {label}")

    # ---- 6. the size of the mutation set -------------------------------
    print()
    print("the mutation sweep's size, wherever the documents state it")
    size = mutation_set_size()
    stated_mutations = stated_mutation_counts()
    for where, value in sorted(stated_mutations.items()):
        print(f"          {where}: {value}")
    # Pinned, like the test-count and flip-rate checks: a reworded document
    # whose regex stops matching would otherwise drop out silently and the
    # check would pass on whatever remained. Vacuous agreement is the failure
    # mode this whole file exists to stop, and it applies to the file itself.
    if not check(f"all {EXPECTED_MUTATION_SOURCES} sources still state it",
                 len(stated_mutations) == EXPECTED_MUTATION_SOURCES,
                 f"found {len(stated_mutations)}: {sorted(stated_mutations)}"):
        failures.append("a document stopped stating the mutation count")
    totals = {k: v for k, v in stated_mutations.items() if "caught" not in k}
    if not check(f"every stated total is the set's size ({size})",
                 bool(totals) and set(totals.values()) == {size},
                 f"documents say {sorted(set(totals.values()))}, the sweep carries {size}"):
        failures.append("the documented mutation count is stale")
    caught = {k: v for k, v in stated_mutations.items() if "caught" in k}
    # Whether every mutation is *caught* is the sweep's own result, not
    # something this file can assert -- it only checks the claim is not larger
    # than the set it is about.
    if not check("and no more are claimed caught than exist",
                 all(v <= size for v in caught.values()),
                 f"claims {sorted(set(caught.values()))} caught of {size}"):
        failures.append("more mutations are claimed caught than the sweep carries")

    # ---- 7. the most-repeated figure in the repository -------------------
    print()
    print("the DR-interaction flip rate, stated in four files")
    stated = stated_flip_rates()
    for where, value in sorted(stated.items()):
        print(f"          {where}: {value}%")
    if not check(f"every source states it ({len(FLIP_RATE_SOURCES)} expected)",
                 len(stated) == len(FLIP_RATE_SOURCES),
                 f"found {sorted(stated)}"):
        failures.append("a source stopped stating the flip rate")
    elif not check("they agree with each other",
                   len(set(stated.values())) == 1,
                   "; ".join(f"{k}={v}" for k, v in sorted(stated.items()))):
        failures.append("the stated flip rates disagree")
    if stated:
        measured = measured_flip_rate()
        claimed = next(iter(stated.values()))
        # Monte Carlo at n=20,000: 2.77% +/- 0.18% over five seeds, so a tenth
        # of a point is noise and half a point is not.
        if not check(f"and they match a fresh measurement ({measured:.2f}%)",
                     abs(measured - claimed) < 0.5,
                     f"documents say {claimed}%, re-measured {measured:.2f}%"):
            failures.append("the stated flip rate is stale")

    # ---- 8. the defect count is a count of something -------------------
    print()
    print("the defect count the page closes on")
    stated = re.search(r"have fixed (\d+) defects", PAGE.read_text(encoding="utf-8"))
    counted = regression_class_count()
    if not check(f"the page states a countable figure ({counted} regression classes)",
                 bool(stated) and int(stated.group(1)) == counted,
                 f"page says {stated.group(1) if stated else 'nothing countable'}, "
                 f"the tree holds {counted}"):
        failures.append("the stated defect count is not what the tree holds")

    # ---- 9. the worked example re-derives --------------------------------
    print()
    print("the worked example the page invites a reader to check by hand")
    from cas13guide.util.seq import revcomp

    shown = page_worked_example()
    if not check("the page still carries one", bool(shown), "no <dl class=\"kv\"> found"):
        failures.append("the worked example vanished from the page")
    else:
        spacer = shown.get("spacer", "")
        site = shown.get("site", "")
        if not check("the spacer is the reverse complement of the site it shows",
                     bool(spacer) and revcomp(site) == spacer,
                     f"revcomp({site}) = {revcomp(site) if site else '?'} != {spacer}"):
            failures.append("the page's spacer does not pair with the page's site")
        top, bottom = page_duplex()
        if not check("the duplex diagram draws those same two sequences",
                     top == spacer and bottom == site[::-1],
                     f"diagram top={top!r} (spacer={spacer!r}), "
                     f"bottom={bottom!r} (site reversed={site[::-1]!r})"):
            failures.append("the drawn duplex is not the pair the row states")

        gc_shown = shown.get("gc", "")
        if spacer:
            gc_real = (spacer.count("G") + spacer.count("C")) / len(spacer)
            if not check(f"the GC it prints is the spacer's GC ({gc_real:.4f})",
                         bool(gc_shown) and abs(float(gc_shown) - gc_real) < 5e-5,
                         f"page says {gc_shown!r}, spacer gives {gc_real:.4f}"):
                failures.append("the page's GC does not match the page's spacer")

    print()
    if failures:
        print(f"{len(failures)} documented claim(s) do not reproduce:")
        for item in failures:
            print("   " + item)
        return 1
    print("Every number and quotation in the README and on the landing page")
    print("reproduces against the package. A claim is worth what the procedure")
    print("that reproduces it is worth, and the prose is now inside that rule")
    print("rather than beside it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
