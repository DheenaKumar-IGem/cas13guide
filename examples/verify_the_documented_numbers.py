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


TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
        "seventy": 70, "eighty": 80, "ninety": 90}


def words_to_int(phrase: str) -> int | None:
    """"A hundred and fifty-five" -> 155.

    The README spells its counts out, for the reason NUMBER_WORDS already
    gives: a word is what a reader checks against. Checking them means
    reading them back.
    """
    total = 0
    current = 0
    seen = False
    for token in phrase.lower().replace("-", " ").replace(",", " ").split():
        if token in ("and", "a", "an"):
            continue
        if token == "hundred":
            current = (current or 1) * 100
            seen = True
        elif token == "thousand":
            total += (current or 1) * 1000
            current = 0
            seen = True
        elif token in TENS:
            current += TENS[token]
            seen = True
        elif token in NUMBER_WORDS:
            current += NUMBER_WORDS[token]
            seen = True
        else:
            return None
    return total + current if seen else None


def test_methods_in(pattern: str) -> int:
    """Test methods under ``tests/`` whose module name matches ``pattern``."""
    import ast

    total = 0
    for path in sorted((ROOT / "tests").glob(pattern)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        total += sum(
            1
            for cls in ast.walk(tree)
            if isinstance(cls, ast.ClassDef)
            for fn in cls.body
            if isinstance(fn, ast.FunctionDef) and fn.name.startswith("test")
        )
    return total


def tests_needing_mirbase() -> int:
    """Test methods that call the helper whose failure path is ``skipTest``."""
    import ast

    tree = ast.parse(
        (ROOT / "tests" / "test_preset_provenance.py").read_text(encoding="utf-8"))
    return sum(
        1
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef)
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef)
        and fn.name.startswith("test")
        and any(isinstance(call, ast.Call)
                and getattr(call.func, "attr", "") == "_mature_fa"
                for call in ast.walk(fn))
    )


def unresolvable_file_citations() -> list[tuple[str, int]]:
    """``(path, how many times it is cited)`` for citations naming no file.

    The two documents point at 115 distinct ``.py`` paths across 179
    citations -- every defect table row names the test module holding its
    regression, and the prose names the module each behaviour lives in. A
    rename is ordinary work and orphans every one of them silently, which is
    the same failure the mutation sweep had when its screening hints named
    modules that no longer existed. That guard caught two stale hints; this is
    the same guard pointed at the prose.

    Paths are written relative to the package (``services/arc_model.py``) or
    as bare filenames (``b_qc.py``), so both are accepted.
    """
    import collections

    bases = [ROOT / "src" / "cas13guide", ROOT / "src", ROOT / "examples",
             ROOT / "tests", ROOT]
    known: set[str] = set()
    for base in bases:
        for path in base.rglob("*.py"):
            known.add(path.relative_to(base).as_posix())
            known.add(path.name)

    cited: collections.Counter[str] = collections.Counter()
    for document in (README, PAGE):
        for name in re.findall(r"`([a-z0-9_]+(?:/[a-z0-9_]+)*[.]py)`",
                               document.read_text(encoding="utf-8")):
            cited[name] += 1
    return [(name, count) for name, count in sorted(cited.items())
            if name not in known]


def documented_word_counts() -> list[tuple[str, int | None, int]]:
    """``(label, what the README spells out, what the tree holds)``.

    These are the counts that move when somebody does ordinary work -- adds a
    regression test, runs another review pass, writes a test that needs the
    miRBase file. Each is stated once in words and nowhere derived, which is
    why they are worth deriving.

    The review-pass section is a scoped historical account: passes ten to
    thirteen have their own sections below it, so its numbers describe the
    eight ``test_bugfix_regressions*.py`` modules those nine passes produced,
    not every regression module in the tree.
    """
    readme = README.read_text(encoding="utf-8")
    spelled = re.search(r"([a-zA-Z -]+?) of them are regression tests "
                        r"from ([a-z]+) review passes", readme)
    batches = re.search(r"[*][*]([0-9]+) defects[*][*] were fixed, in batches of"
                        r"[^*]*[*][*]([0-9, ]+)[*][*]", readme)
    heading = re.search(r"### What ([a-z]+) review passes found", readme)
    needing = re.search(r"([A-Za-z]+) of them need a miRBase", readme)

    parts = ([int(n) for n in batches.group(2).split(",") if n.strip()]
             if batches else [])
    return [
        ("regression tests from the nine passes",
         words_to_int(spelled.group(1)) if spelled else None,
         test_methods_in("test_bugfix_regressions*.py")),
        ("defects fixed, against the batches listed",
         int(batches.group(1)) if batches else None, sum(parts)),
        ("review passes, in the heading",
         words_to_int(heading.group(1)) if heading else None, len(parts)),
        ("review passes, in the sentence",
         words_to_int(spelled.group(2)) if spelled else None, len(parts)),
        ("tests needing a miRBase file",
         words_to_int(needing.group(1)) if needing else None,
         tests_needing_mirbase()),
    ]


def documented_vocabulary() -> tuple[int | None, int | None, int]:
    """``(produced, of how many, how many the package can say)``.

    The README quotes "52 of 52 things it can say". The denominator is the
    half that drifts: adding a reason code or a caveat is a local edit, and
    nothing about writing one prompts you to update a figure two files away.
    Establishing the numerator needs a whole suite run and stays
    ``verify_vocabulary_coverage.py``'s job.
    """
    stated = re.search(r"verify_vocabulary_coverage[.]py[^#]*#[ ]*"
                       r"([0-9]+) of ([0-9]+) things",
                       README.read_text(encoding="utf-8"))
    return (int(stated.group(1)) if stated else None,
            int(stated.group(2)) if stated else None,
            vocabulary_size())


def top_ranked_guide() -> dict:
    """Rank 1 of a real run over the same target the page's row describes.

    The page's spacer is 22 nt, which is RfxCas13d's geometry, not the
    LwaCas13a configuration the limitations block above it comes from.
    """
    from cas13guide.services.workflow import DesignRequest, run_request

    with tempfile.TemporaryDirectory() as directory:
        fasta = pathlib.Path(directory) / "target.fa"
        fasta.write_text(
            ">MALAT1_frag" + chr(10) + TARGET + chr(10), encoding="utf-8")
        outcome = run_request(DesignRequest(
            fasta_path=str(fasta), task="knockdown",
            assay="cell_culture_knockdown", effector="RfxCas13d",
        ))
    ranked = outcome.report["ranked"]
    return ranked[0] if ranked else {}


def longest_shared_run(sequences: list[str]) -> int:
    """The longest stretch of target every one of ``sequences`` covers."""
    first, rest = sequences[0], sequences[1:]
    best = 0
    for start in range(len(first)):
        for stop in range(start + 1, len(first) + 1):
            run = first[start:stop]
            if len(run) > best and all(run in other for other in rest):
                best = len(run)
    return best


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
    produced, of_how_many, size = documented_vocabulary()
    if not check(f"the denominator is the vocabulary's size ({size})",
                 of_how_many == size,
                 f"README says {of_how_many}, the package can say {size} things"):
        failures.append("the documented vocabulary size is stale")
    if not check("and it does not claim more produced than exist",
                 produced is not None and produced <= size,
                 f"claims {produced} of {size}"):
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

    # ---- 10. the near-duplicate windows overlap by what it says ----------
    print()
    print("the three near-duplicate windows the ranking section prints")
    text = README.read_text(encoding="utf-8")
    windows = re.findall(r"^#\s+\d+\s+start=\s*\d+\s+([ACGU]+)\s*$", text, re.M)
    claim = re.search(r"share (\d+) of (\d+) target bases "
                      r"\((\d+) for a neighbouring pair\)", text)
    if not check("the block still prints three of them", len(windows) == 3,
                 f"found {len(windows)}"):
        failures.append("the near-duplicate window block changed shape")
    elif not check("and the sentence still states the overlap", bool(claim),
                   "no 'share N of M target bases (K for a neighbouring pair)'"):
        failures.append("the overlap sentence changed shape")
    else:
        assert claim is not None
        shared, width, pairwise = (int(g) for g in claim.groups())
        if not check(f"each window is {width} nt",
                     all(len(w) == width for w in windows),
                     f"lengths {[len(w) for w in windows]}"):
            failures.append("the stated spacer width is not the width shown")
        common = longest_shared_run(windows)
        if not check(f"all three share {shared} of {width}", common == shared,
                     f"they actually share {common}"):
            failures.append("the three-way window overlap is misstated")
        neighbours = min(longest_shared_run(windows[i:i + 2]) for i in range(2))
        if not check(f"a neighbouring pair shares {pairwise}",
                     neighbours == pairwise,
                     f"a neighbouring pair actually shares {neighbours}"):
            failures.append("the pairwise window overlap is misstated")

    # ---- 11. the canonical spacer lengths it quotes ----------------------
    print()
    print("the canonical spacer lengths the domain section quotes")
    from cas13guide.contract.effectors import EFFECTORS
    from cas13guide.contract.enums import Effector

    quoted = re.search(r"RfxCas13d's\s+([\d/]+) for LwaCas13a and "
                       r"LwaCas13a's\s+(\d+) for RfxCas13d", text)
    if not check("the sentence still names both sets", bool(quoted),
                 "the out-of-domain sentence changed shape"):
        failures.append("the quoted canonical lengths changed shape")
    else:
        assert quoted is not None
        for label, effector, stated in (
            ("RfxCas13d", Effector.RFX_CAS13D,
             sorted(int(n) for n in quoted.group(1).split("/"))),
            ("LwaCas13a", Effector.LWA_CAS13A, [int(quoted.group(2))]),
        ):
            real = sorted(EFFECTORS[effector].spacer_lengths)
            if not check(f"{label}'s canonical set is {real}", stated == real,
                         f"README says {stated}"):
                failures.append(f"the quoted canonical lengths for {label} are stale")

    # ---- 12. the heuristic saturates where it says it does ---------------
    print()
    print("the saturation figure that justifies a CRITICAL limitation")
    import random

    from cas13guide.services.registry import HeuristicCas13Model
    from cas13guide.util.duplex import DuplexProfile

    pattern = r"a perfect duplex scores\s*(?:≈|about\s+)(\d\.\d+)"
    stated_readme = re.findall(pattern, text)
    stated_page = re.findall(pattern, PAGE.read_text(encoding="utf-8"))
    everywhere = set(stated_readme) | set(stated_page)
    if not check("the README and the page agree on one figure",
                 len(everywhere) == 1 and len(stated_readme) >= 1 and len(stated_page) >= 1,
                 f"README says {stated_readme}, the page says {stated_page}"):
        failures.append("the saturation figure differs between documents")
    else:
        saturation = float(everywhere.pop())
        model = HeuristicCas13Model()
        rng = random.Random(20260904)
        scores = []
        for _ in range(4000):
            length = rng.choice([22, 23, 28, 30])
            spacer = "".join(rng.choice("ACGU") for _ in range(length))
            scores.append(model.predict(
                spacer,
                DuplexProfile(length=length, mismatch_positions=(),
                              wobble_positions=(), cigar="M" * length),
                None))
        ceiling = max(scores)
        if not check(f"a perfect duplex tops out at {saturation} (measured {ceiling:.4f})",
                     round(ceiling, 2) == saturation,
                     f"the documents say {saturation}, the model gives {ceiling:.4f}"):
            failures.append("the stated saturation is not what the model does")
        # "regardless of sequence" is the load-bearing half of the sentence:
        # it is the reason the ranking among well-matched guides is said to
        # carry almost no information. A wide spread would falsify it while
        # leaving the headline number correct.
        spread = ceiling - min(scores)
        if not check(f"and barely moves with sequence (spread {spread:.4f})",
                     spread < 0.05,
                     f"perfect duplexes range over {spread:.4f}, which is not "
                     f'"regardless of sequence"'):
            failures.append("the saturation is not sequence-independent as claimed")

    # ---- 13. the worked example is a run, not a tableau -------------------
    print()
    print("the rank-1 row, against a run of the tool over the same target")
    top = top_ranked_guide()
    if not check("the run still ranks something", bool(top), "nothing ranked"):
        failures.append("the worked example's run produces no ranking")
    elif shown:
        spacer = shown.get("spacer", "")
        site = shown.get("site", "")
        start = top.get("start")
        binds = shown.get("binds", "")
        for label, page_says, run_says in (
            ("the guide id", shown.get("guide", ""), str(top.get("guide_id", ""))),
            ("the spacer", spacer, str(top.get("spacer", ""))),
            ("the verdict", shown.get("verdict", ""), str(top.get("verdict", ""))),
        ):
            if not check(f"{label} is the run's ({run_says})",
                         page_says.startswith(run_says) and bool(run_says),
                         f"the page says {page_says!r}"):
                failures.append(f"the page's {label} is not the run's")
        detail = " ".join(top.get("detail", []))
        if not check("the verdict's reason is the run's own words",
                     bool(detail) and detail in shown.get("verdict", ""),
                     f"run says {detail!r}, page says {shown.get('verdict', '')!r}"):
            failures.append("the page's verdict reason is not the run's")
        # The coordinate is the field a reader checks by hand, and it was the
        # one field nothing checked: every other value on the row agreed with
        # every other, and the location pointed 147 bases away from the site
        # printed beside it. Tie all three together -- page coordinate, page
        # site, real target -- so agreement among the row's own cells can no
        # longer stand in for agreement with the target.
        expected = f"MALAT1_frag at {start}..{start + len(spacer)}" if spacer else ""
        if not check(f"the coordinate is the run's ({expected})",
                     binds == expected, f"the page says {binds!r}"):
            failures.append("the page's coordinate is not the run's")
        if not check(f"and the target really carries that site at {start}",
                     bool(site) and TARGET[start : start + len(site)] == site,
                     f"target[{start}:{start + len(site) if site else '?'}] = "
                     f"{TARGET[start : start + len(site)] if site else '?'!r}, "
                     f"page shows {site!r}"):
            failures.append("the page's site is not at the page's coordinate")

    # ---- 14. the counts the README spells out in words --------------------
    print()
    print("the counts the review-pass section states as words")
    for label, stated, actual in documented_word_counts():
        if not check(f"{label}: {actual}", stated == actual,
                     f"the README says {stated}"):
            failures.append(f"the stated count of {label} is stale")

    # ---- 15. every file the documents point at ---------------------------
    print()
    print("the files the two documents name")
    dangling = unresolvable_file_citations()
    if not check("every cited .py path resolves",
                 not dangling,
                 "; ".join(f"{name} (cited {count}x)" for name, count in dangling)):
        failures.append("the documents cite files that do not exist")

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
