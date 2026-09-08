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
EXAMPLES = Path(__file__).resolve().parent
TESTS = ROOT / "tests"

#: Fields that are read nowhere and are meant to be. Each needs a reason, so
#: that adding to this list is a decision somebody made rather than a way to
#: quiet the check.
ALLOWED: dict[str, str] = {
    "do_GET": "dispatched by name from BaseHTTPRequestHandler, so the caller is in the standard library and no walk over src/ can see it. `do_POST` escapes this list only because a docstring happens to name it, which is the string-constant over-report this file documents",
    "log_message": "same: BaseHTTPRequestHandler calls it for every request. Overridden here to keep the access log off stdout, so having no caller in src/ is the point of it",
    "is_structured": "FR3.4 has two implementations and only one path. The gate reads Thresholds.min_accessibility, a spec decision defaulting to off; this reads STRUCTURED_SITE_THRESHOLD, whose own comment calls itself advisory. Kept because the requirement is real and this is its reporting form, but nothing in the package reports it, and the constant exists only to feed this method",
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
    """Dataclass fields, the properties beside them, and the plain methods.

    Fields only, at first. That missed `SignalPrediction.saturated` -- computed
    on every prediction, read by one test and nothing in the package, while the
    constants' own comment named it as the check that decides whether a
    saturating assay may rank. A property is exactly the shape a
    computed-but-unread *guard* takes, and guards are what this codebase
    accumulates: the standard fix for two surfaces disagreeing is to add one.

    Methods came third, and found two more. A method is a value computed for
    somebody just as a field is, and nothing about being callable makes it
    called: ``ViennaRNAAccessibilityModel.is_structured`` implements FR3.4
    against a module constant that exists only to feed it, while the gate
    implements FR3.4 separately against the spec threshold -- so the
    constant, the method and the requirement they name are all real, and the
    path between them is not. ``Transcriptome.covered_universes`` had no
    caller anywhere, not even a test.
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
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
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


#: Module constants nobody reads, and why that is allowed.
ALLOWED_CONSTANTS: dict[str, str] = {
    "DEFAULT_DG_PER_NT_RANGE": (
        "the assay-specific default a spec may adopt. The package does not "
        "invent a range -- `summarize` derives an empirical one from the "
        "candidate set instead -- so this is offered to a spec author and "
        "used by nothing here on purpose"
    ),
    "STANDING_CAVEATS": (
        "deliberately empty, and kept as the module's public name for callers "
        "that imported it. The caveats a run carries are derived from that "
        "run now; a fixed banner kept warning about placeholder constants "
        "after real ones were supplied, and stayed silent about a heuristic "
        "index nobody remembered to add a line for"
    ),
}


def _declared_constants() -> dict[str, str]:
    """Public module-level constants, by the file that declares them."""
    found: dict[str, str] = {}
    for path in SRC.rglob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (isinstance(target, ast.Name) and target.id.isupper()
                        and not target.id.startswith("_")):
                    found.setdefault(target.id, str(path.relative_to(SRC)))
    return found


def _constant_reads(root: Path) -> set[str]:
    """Names actually *used* as code, rather than named in prose.

    Deliberately not ``_mentions``: that counts attribute reads and string
    constants, which is right for a field and wrong here twice over. A
    constant is referenced as a bare name, so attributes would miss every
    real use; and its own docstring names it, so strings would call every
    constant read. ``SOFT_QC_FLAGS`` was invisible for exactly that reason --
    a set describing which QC findings only warn, consulted by nothing, whose
    comment called it a single source of truth.
    """
    seen: set[str] = set()
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                seen.add(node.id)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    seen.add(alias.asname or alias.name)
    return seen


def _report_constants() -> int:
    """Print the constant half. Returns the number that are not allowed."""
    declared = _declared_constants()
    in_src = _constant_reads(SRC)
    in_tests = _constant_reads(ROOT / "tests") | _constant_reads(ROOT / "examples")

    unread = sorted(n for n in declared if n not in in_src)
    tests_only = [n for n in unread if n in in_tests]
    nowhere = [n for n in unread if n not in in_tests]
    flagged = [n for n in nowhere if n not in ALLOWED_CONSTANTS]
    excused = [n for n in nowhere if n in ALLOWED_CONSTANTS]

    print()
    print(f"{len(declared)} public module constants declared")
    if tests_only:
        print(f"read only by tests, never by the package -- {len(tests_only)}:")
        for name in tests_only:
            print(f"    {name:24} {declared[name]}")
    if excused:
        print(f"read nowhere, and allowed -- {len(excused)}:")
        for name in excused:
            print(f"    {name:24} {ALLOWED_CONSTANTS[name]}")
    if flagged:
        print(f"read NOWHERE and not allowed -- {len(flagged)}:")
        for name in flagged:
            print(f"    {name:24} {declared[name]}")
        print("Each is a constant this package states and never consults.")
        print("Either give it a reader, or add it to ALLOWED_CONSTANTS with")
        print("the reason it has none.")
    return len(flagged)


#: Public functions nobody calls, and why that is allowed.
ALLOWED_FUNCTIONS: dict[str, str] = {
    "hamming": (
        "a generic string primitive, kept deliberately and documented as the "
        "*wrong* function for comparing a spacer to a target site: it is "
        "wobble-blind, so it over-counts mismatches and inflates apparent "
        "specificity. `util.duplex.profile_duplex` is what specificity claims "
        "use. Keeping it uncalled is the point"
    ),
}


def _declared_functions() -> dict[str, str]:
    """Public functions defined at module level, by the file declaring them."""
    found: dict[str, str] = {}
    for path in SRC.rglob("*.py"):
        if path.name == "_contrafold_kernel.py":
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    found[node.name] = f"{path.relative_to(SRC)}:{node.lineno}"
    return found


def _called(root: Path) -> set[str]:
    """Names that appear in call position.

    Call sites rather than mentions, because `predicted_signal_nm` is also a
    dataclass field and a CSV column: counting the name would have found it
    everywhere while nothing invoked it.
    """
    seen: set[str] = set()
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                seen.add(func.id)
            elif isinstance(func, ast.Attribute):
                seen.add(func.attr)
    return seen


def _report_functions() -> list[str]:
    """Public module-level functions with no call site anywhere."""
    declared = _declared_functions()
    called = _called(SRC) | _called(TESTS) | _called(EXAMPLES)

    print(f"{len(declared)} public module-level functions declared")
    if not declared:
        print("FAIL: none found at all, so this section proved nothing.")
        return ["no functions discovered"]

    uncalled = sorted(n for n in declared if n not in called)
    flagged = [n for n in uncalled if n not in ALLOWED_FUNCTIONS]
    excused = [n for n in uncalled if n in ALLOWED_FUNCTIONS]

    if excused:
        print(f"never called, and allowed -- {len(excused)}:")
        for name in excused:
            print(f"    {name:24} {ALLOWED_FUNCTIONS[name]}")
    if flagged:
        print(f"never called ANYWHERE -- {len(flagged)}:")
        for name in flagged:
            print(f"    {name:24} {declared[name]}")
        print()
        print("A function nothing calls is a value computed for nobody with a")
        print("parameter list. Give it a caller, or add it to")
        print("ALLOWED_FUNCTIONS with the reason it has none.")
    return flagged


def main() -> int:
    declared = _declared()
    in_src = _mentions(SRC)
    in_tests = _mentions(TESTS)

    unread, tests_only = [], []
    for name, owners in sorted(declared.items()):
        if name in in_src:
            continue
        (tests_only if name in in_tests else unread).append((name, owners))

    print(f"{len(declared)} public fields, properties and methods declared")

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

    stranded = _report_constants()
    if stranded:
        return 1

    print()
    if _report_functions():
        return 1

    print()
    print("Every value this package computes reaches something.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
