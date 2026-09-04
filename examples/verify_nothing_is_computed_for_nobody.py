"""Find values this package computes and then reads nowhere.

``GateResult.detail`` was the first of these to turn up: eight places in the
gate compose a numeric explanation for every verdict -- "forbidden hit
UCB(activity) 0.412 > 0.200" -- and a grep for a *read* of it across ``src/``
returned nothing at all. A run reported ``FAIL: 41`` and offered no way to
learn what was wrong with any of the 41.

That is a defect class rather than an incident. A field costs nothing to keep
filling in, nothing complains when its last reader goes, and the loss is
invisible: the number is still computed, still correct, and no longer reaches
anybody. Running this sweep over every dataclass in the package turned up five
more, including the three things QC measures about every spacer, the
limitations that never reached the machine-readable report, and the absolute
signal figure that ``--kinetics-constants`` was advertised as unlocking.

**The rule matters.** The first cut excluded each field's home module, on the
theory that a field is "used" only when something outside reads it. That
returned 138 candidates, nearly all of them fields read a few lines below their
own declaration, and a list that long is a list nobody reads. The rule that
actually describes the defect is stricter: the value is constructed and then
read by *nothing, anywhere*.

**What it cannot see.** Attribute reads and quoted names, only, and it errs in
both directions.

It over-reports: a field consumed through ``dataclasses.asdict`` looks unread,
which is why ``MetricSummary``'s fields appear "read only by tests" while
reaching the report perfectly well.

It under-reports, in two ways that were each demonstrated rather than guessed
at. A name shared between two classes hides a real gap -- ``QCResult.gc``
looked used because ``ControlGuide.gc`` is read somewhere else entirely, and
that one had to be caught by hand. And a field exported under its own name
keeps a matching string constant even after its last read goes: deleting the
line that reads ``r.sensitivity`` leaves ``"sensitivity"`` sitting in the row
dict beside it, and this check stays quiet. Counting quoted names is
deliberate, since dict-key consumption is real consumption, and it costs this.

So: candidates, not findings. It is worth running because five real defects
came out of it, not because a clean run proves anything.

Usage::

    python examples/verify_nothing_is_computed_for_nobody.py

Exits non-zero when a field outside the allowlist is read nowhere.
"""

from __future__ import annotations

import ast
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "cas13guide"
TESTS = ROOT / "tests"

#: Fields that are read nowhere and are meant to be. Each needs a reason, so
#: that adding to this list is a decision somebody made rather than a way to
#: quiet the check.
ALLOWED: dict[str, str] = {
    "collateral_active": (
        "a biological fact about the ortholog, true for all three shipped "
        "effectors. It documents the assumption `predicted_signal` rests on; "
        "an effector without collateral activity would make it load-bearing"
    ),
    "activated_fraction": (
        "kinetics diagnostic. The package's designated absolute figure is "
        "`absolute_signal()`, i.e. cleaved_reporter_nm, and that one is now "
        "reported; a second absolute number nobody asked for is not better"
    ),
    "cleavage_rate": "the same, one field over",
    "blast_mismatches": (
        "parsed from BLAST's own output for completeness of the row. The "
        "mismatch count the pipeline uses is the wobble-aware one from "
        "`profile_duplex`, deliberately, since BLAST does not know G-U pairs "
        "are real base pairs in RNA:RNA"
    ),
    "per_fold": "per-fold detail behind the cross-validation summary",
    "target_length": "preset designability detail, reported through its summary",
    "canonical_length": "the same",
}


def _declared() -> dict[str, list[str]]:
    """Dataclass fields, and the properties beside them.

    Fields only, at first. That missed `SignalPrediction.saturated` -- computed
    on every prediction, read by one test and nothing in the package, while the
    constants' own comment named it as the check that decides whether a
    saturating assay may rank. A property is exactly the shape a
    computed-but-unread *guard* takes, and guards are what this codebase
    accumulates: the standard fix for two surfaces disagreeing is to add one.
    """
    found: dict[str, list[str]] = collections.defaultdict(list)
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_dataclass = any(
                "dataclass" in ast.unparse(d) for d in node.decorator_list
            )
            for stmt in node.body:
                if (is_dataclass and isinstance(stmt, ast.AnnAssign)
                        and isinstance(stmt.target, ast.Name)):
                    name = stmt.target.id
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    ast.unparse(d) == "property" for d in stmt.decorator_list
                ):
                    name = stmt.name
                else:
                    continue
                if not name.startswith("_"):
                    found[name].append(f"{node.name} ({path.relative_to(SRC)})")
    return found


def _mentions(root: Path) -> set[str]:
    """Attribute reads and string constants -- everything a field can be read as."""
    seen: set[str] = set()
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                seen.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                seen.add(node.value)
    return seen


def main() -> int:
    declared = _declared()
    in_src = _mentions(SRC)
    in_tests = _mentions(TESTS)

    unread, tests_only = [], []
    for name, owners in sorted(declared.items()):
        if name in in_src:
            continue
        (tests_only if name in in_tests else unread).append((name, owners))

    print(f"{len(declared)} public fields and properties declared")

    # A scraped inventory can come back empty, and this file's whole result is
    # a statement about what is *missing* from it: with nothing declared,
    # nothing is unread, nothing is flagged, and it exits 0 having compared two
    # empty sets. Same vacuous pass the vocabulary scraper had.
    #
    # Anchored on fields that cannot legitimately disappear rather than on a
    # count, which would need editing every time a field is added. Each of
    # these is load-bearing enough that its absence means the walk broke, not
    # that the field went away.
    load_bearing = {"required_coverage", "worst_off_target", "lower", "upper"}
    if not declared:
        print("FAIL: no public fields or properties found at all. The source")
        print("walk is broken, and an empty inventory reports no dead values")
        print("because it has nothing to report on.")
        return 1
    absent = sorted(load_bearing - set(declared))
    if absent:
        print("FAIL: the walk did not find fields that must exist: "
              + ", ".join(absent))
        print("Its inventory is of something other than this package.")
        return 1
    print()

    flagged = [(n, o) for n, o in unread if n not in ALLOWED]
    excused = [(n, o) for n, o in unread if n in ALLOWED]

    if tests_only:
        print(f"read only by tests, never by the package -- {len(tests_only)}:")
        for name, owners in tests_only:
            print(f"    {name:24} {', '.join(owners)}")
        print("    (some of these reach output through asdict; see the module")
        print("     docstring on what this cannot see)")
        print()

    if excused:
        print(f"read nowhere, and allowed -- {len(excused)}:")
        for name, _ in excused:
            print(f"    {name:24} {ALLOWED[name]}")
        print()

    if flagged:
        print(f"read NOWHERE and not allowed -- {len(flagged)}:")
        for name, owners in flagged:
            print(f"    {name:24} {', '.join(owners)}")
        print()
        print("Each is a value this package computes for nobody. Either give it")
        print("a reader, or add it to ALLOWED with the reason it has none.")
        return 1

    print("Every value this package computes reaches something.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
