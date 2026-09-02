"""Measure line coverage the way the README's figure is measured.

The Verification section quotes a coverage number, and every other figure in
that section ships the procedure that reproduces it. This one did not, and the
procedure is not the obvious one: three separate attempts at re-measuring it
produced three wrong answers, each for a different reason.

**Subprocesses have to be traced.** A good part of this suite drives the CLI
through ``subprocess`` -- that is the point of those tests -- and ``coverage``
does not follow a child process unless the child starts tracing itself. A
plain run therefore reports ``cli.py`` around 6% and ``demo.py`` at 20%, which
says nothing about the tests and everything about the instrument. The child is
told to trace itself by putting a ``sitecustomize`` on its path that calls
``coverage.process_startup()``, which fires only when ``COVERAGE_PROCESS_START``
names a config file.

**The data file has to be absolute.** In parallel mode each process writes its
own data file next to the *current directory*, and several of these tests run
the CLI inside a temporary directory that is deleted immediately afterwards.
Their coverage goes with it: the numbers come out lower than the truth, silently
and only for the subprocess-heavy modules -- exactly the ones the tracing was
turned on for.

**``source`` has to be a package name, not a path.** ``source = src/cas13guide``
is resolved against each process's working directory, so it means nothing in a
child running elsewhere; an absolute path fails differently, matching no module
and collecting nothing at all. ``source_pkgs = cas13guide`` is what survives
both.

Usage::

    python examples/measure_coverage.py            # full suite, traced
    python examples/measure_coverage.py -k report  # narrow it while iterating

Two figures are read as artefacts rather than gaps, and the report says so:
``_contrafold_kernel.py`` is ``@njit`` and numba-compiled code is not traced at
all, and ``__main__.py`` exists to be executed by ``python -m``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHIM = ROOT / "sitecustomize.py"

#: Modules whose coverage number is an artefact of how they run, not a gap.
ARTEFACTS = {
    "_contrafold_kernel.py": "@njit; numba-compiled code is never traced",
    "__main__.py": "exists to be run by `python -m`, not imported",
}


def main(argv: list[str]) -> int:
    if SHIM.exists():
        print(f"refusing to overwrite {SHIM}", file=sys.stderr)
        return 2

    workspace = Path(tempfile.mkdtemp(prefix="cas13guide-coverage-"))
    config = workspace / "coverage.rc"
    data = workspace / "data" / ".coverage"
    data.parent.mkdir(parents=True)
    config.write_text(
        "[run]\n"
        # A package name, not a path: see the module docstring.
        "source_pkgs = cas13guide\n"
        "parallel = true\n"
        f"data_file = {data}\n",
        encoding="utf-8",
    )
    SHIM.write_text("import coverage\ncoverage.process_startup()\n", encoding="utf-8")

    env = {
        **os.environ,
        "COVERAGE_FILE": str(data),
        "COVERAGE_PROCESS_START": str(config),
        # So a child started anywhere still imports the shim.
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
    }
    rc = ["--rcfile", str(config)]

    try:
        run = subprocess.run(
            [sys.executable, "-m", "coverage", "run", *rc, "-m", "pytest", "-q", *argv],
            cwd=ROOT, env=env,
        )
        subprocess.run([sys.executable, "-m", "coverage", "combine", *rc],
                       cwd=ROOT, env=env, capture_output=True, text=True)
        report = subprocess.run(
            [sys.executable, "-m", "coverage", "report", *rc],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
    finally:
        SHIM.unlink(missing_ok=True)
        shutil.rmtree(workspace, ignore_errors=True)

    print(report.stdout, end="")
    if report.returncode != 0:
        print(report.stderr, file=sys.stderr)
        return report.returncode

    total = ""
    measured: dict[str, int] = {}
    for line in report.stdout.splitlines():
        if line.startswith("TOTAL"):
            total = line.split()[-1]
        for name in ARTEFACTS:
            if name in line and line.rstrip().endswith("%"):
                measured[name] = int(line.split()[-1].rstrip("%"))
    print()
    print(f"line coverage, with subprocess tracing: {total}")

    # Only call something an artefact when it actually reads low. `__main__.py`
    # is on the list because it exists to be executed rather than imported --
    # but under subprocess tracing it reads 100%, and announcing it as an
    # excused gap would be excusing a gap that is not there.
    excused = {n: w for n, w in ARTEFACTS.items() if measured.get(n, 0) < 50}
    if excused:
        print("Low numbers that are artefacts of how the code runs, not gaps:")
        for name, why in excused.items():
            print(f"    {name:26} {measured.get(name, 0):>3}%  {why}")
        print("Acting on these would mean writing tests for code already tested.")
    if run.returncode != 0:
        print("\nNote: the suite itself did not pass, so the figure is provisional.")
        return run.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
