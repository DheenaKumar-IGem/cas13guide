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

    # ---- 3. the defect count is a count of something -------------------
    print()
    print("the defect count the page closes on")
    stated = re.search(r"have fixed (\d+) defects", PAGE.read_text(encoding="utf-8"))
    counted = regression_class_count()
    if not check(f"the page states a countable figure ({counted} regression classes)",
                 bool(stated) and int(stated.group(1)) == counted,
                 f"page says {stated.group(1) if stated else 'nothing countable'}, "
                 f"the tree holds {counted}"):
        failures.append("the stated defect count is not what the tree holds")

    # ---- 4. the worked example re-derives --------------------------------
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
