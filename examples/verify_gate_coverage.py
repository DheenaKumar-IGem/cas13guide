"""Check that every verdict this package can reach is reached by some test.

A ``ReasonCode`` is the vocabulary of the whole tool: it is what the gate says,
what the report prints, and what a caller switches on. A code no test ever
produces is a verdict a user can receive that nobody has ever seen come out --
either dead code, or a live path with no coverage at all.

This found three of the latter. ``TARGET_INACCESSIBLE`` and
``DUPLEX_ENERGY_OUT_OF_RANGE`` are only reachable once the spec sets a floor or
a window, and ``--min-accessibility`` is a shipped command-line flag, so
turning it on ran code no test had executed. ``BORDERLINE_OFF_TARGET`` needs no
flag at all -- it occupies the narrow band just under the hazard budget, and no
fixture had happened to land a guide there. All three turned out to be correct.
That is the usual outcome and not a reason to skip the check: the point is that
nobody knew.

**What it measures, and what it does not.** This reports whether a code was
*produced*, not whether anything *asserted* it. A code can be produced
incidentally by a test that looks at something else entirely. So a green run
here means "no verdict is completely unvisited", which is a floor rather than a
guarantee -- the same relationship line coverage has to correctness. The
mutation check next door is the instrument that asks whether behaviour is
pinned.

Both places a code can come from are recorded:

* the gate, where it lands in ``GateResult.reasons``;
* the orchestrator, where budget shortfalls are appended to
  ``RunTelemetry.reason_codes`` and exported in the provenance ledger.

The first version of this probe watched only the gate, and duly reported three
orchestrator codes as never produced when all three were exercised -- an
instrument that reports a gap it cannot see is worse than none, so it is worth
saying which blind spot was closed and why.

Usage::

    python examples/verify_gate_coverage.py          # runs the whole suite
    python examples/verify_gate_coverage.py -k gate  # narrow it while iterating

Exits non-zero when some code is never produced.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = "_gate_coverage_plugin"

PLUGIN_SOURCE = '''
"""Records every ReasonCode the suite actually produces. Written by
``verify_gate_coverage.py``; not part of the package."""

import json
import os

SEEN: set[str] = set()


class _Recording(list):
    """A list that notices what is appended to it."""

    def append(self, item):
        SEEN.add(getattr(item, "value", item))
        super().append(item)


def pytest_configure(config):
    import cas13guide.pipeline.f_decide as f_decide
    import cas13guide.pipeline.orchestrator as orchestrator
    from cas13guide.models.evidence import GateResult as RealGate

    def recording_gate(*args, **kwargs):
        result = RealGate(*args, **kwargs)
        SEEN.update(code.value for code in result.reasons)
        return result

    f_decide.GateResult = recording_gate

    real_telemetry = orchestrator.RunTelemetry

    def recording_telemetry(*args, **kwargs):
        telemetry = real_telemetry(*args, **kwargs)
        telemetry.reason_codes = _Recording(telemetry.reason_codes)
        return telemetry

    orchestrator.RunTelemetry = recording_telemetry


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["GATE_COVERAGE_OUT"], "w", encoding="utf-8") as handle:
        json.dump(sorted(SEEN), handle)
'''


def main(argv: list[str]) -> int:
    plugin_path = ROOT / f"{PLUGIN}.py"
    out_path = ROOT / ".gate_coverage.json"
    plugin_path.write_text(PLUGIN_SOURCE, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", PLUGIN, *argv],
            cwd=ROOT,
            env={**__import__("os").environ, "GATE_COVERAGE_OUT": str(out_path)},
        )
        if not out_path.exists():
            print("the suite did not run to completion; nothing to report")
            return 2
        produced = set(__import__("json").loads(out_path.read_text(encoding="utf-8")))
    finally:
        plugin_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)

    sys.path.insert(0, str(ROOT / "src"))
    from cas13guide.contract.enums import ReasonCode

    every = [code.value for code in ReasonCode]
    missing = [code for code in every if code not in produced]

    print()
    print(f"{len(produced)}/{len(every)} reason codes produced by the suite")
    if missing:
        print()
        print("never produced -- dead code, or a live path with no coverage:")
        for code in missing:
            print(f"    {code}")
        print()
        print("Each one is a verdict a user can receive that no test has seen.")
        return 1

    print("Every verdict this package can reach is reached by some test.")
    print("That is a floor, not a guarantee: produced is not the same as")
    print("asserted. verify_mutation_coverage.py is the one that asks whether")
    print("the behaviour is pinned.")
    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
