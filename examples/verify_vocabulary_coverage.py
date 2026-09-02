"""Check that everything this package can say to a user is said in some test.

This package has four vocabularies for talking to a user: ``ReasonCode``, what
the gate says and what a caller switches on; ``Limitation``, what the run
admits it could not establish; ``QCFlag``, what is wrong with a spacer; and
``ReviewTier``, how much of a human a REVIEW needs. A value in any of them
that no test ever produces is a sentence a user can receive that nobody has
watched come out -- either dead code, or a live path with no coverage at all.

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

Every place a value can come from is recorded:

* the gate, where a reason code lands in ``GateResult.reasons`` and a tier in
  ``GateResult.review_tier``;
* the orchestrator, where budget shortfalls are appended to
  ``RunTelemetry.reason_codes`` and exported in the provenance ledger;
* ``limitations_for_run``, which constructs every ``Limitation``;
* ``qc_candidate``, whose flags and warnings are what is wrong with a spacer.

Asking the question of the limitations found two more: the FR6.3
thin-calibration warning, which needs a confidence model no fixture supplied,
and the unverified-preset warning, which cannot fire while ``PRESETS_VERIFIED``
is ``True`` and is exactly what a contributor meets the moment they add a
preset they have not checked. The QC flags and the review tiers came back
clean, all thirteen of them, which is a result worth having rather than a
wasted question.

The first version of this probe watched only the gate, and duly reported three
orchestrator codes as never produced when all three were exercised -- an
instrument that reports a gap it cannot see is worse than none, so it is worth
saying which blind spot was closed and why.

Usage::

    python examples/verify_vocabulary_coverage.py           # the whole suite
    python examples/verify_vocabulary_coverage.py -k gate   # narrow it

Exits non-zero when anything in any of the four is never produced.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN = "_vocabulary_coverage_plugin"

PLUGIN_SOURCE = '''
"""Records everything the suite makes the package say. Written by
``verify_vocabulary_coverage.py``; not part of the package."""

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
        if result.review_tier is not None:
            SEEN.add("tier:" + result.review_tier.value)
        return result

    f_decide.GateResult = recording_gate

    import cas13guide.pipeline.b_qc as b_qc
    from cas13guide.models.evidence import QCResult as RealQC

    def recording_qc(*args, **kwargs):
        qc = RealQC(*args, **kwargs)
        # Warnings count: a flag that only ever deprioritises is still a thing
        # the package says about a guide.
        SEEN.update("qc:" + flag.value for flag in qc.flags)
        SEEN.update("qc:" + flag.value for flag in qc.warnings)
        return qc

    b_qc.QCResult = recording_qc

    import cas13guide.services.limitations as limitations
    real_limitation = limitations.Limitation

    def recording_limitation(*args, **kwargs):
        made = real_limitation(*args, **kwargs)
        SEEN.add("limitation:" + made.id)
        return made

    limitations.Limitation = recording_limitation

    real_telemetry = orchestrator.RunTelemetry

    def recording_telemetry(*args, **kwargs):
        telemetry = real_telemetry(*args, **kwargs)
        telemetry.reason_codes = _Recording(telemetry.reason_codes)
        return telemetry

    orchestrator.RunTelemetry = recording_telemetry


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["VOCABULARY_COVERAGE_OUT"], "w", encoding="utf-8") as handle:
        json.dump(sorted(SEEN), handle)
'''


def main(argv: list[str]) -> int:
    plugin_path = ROOT / f"{PLUGIN}.py"
    out_path = ROOT / ".vocabulary_coverage.json"
    plugin_path.write_text(PLUGIN_SOURCE, encoding="utf-8")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", PLUGIN, *argv],
            cwd=ROOT,
            env={**__import__("os").environ, "VOCABULARY_COVERAGE_OUT": str(out_path)},
        )
        if not out_path.exists():
            print("the suite did not run to completion; nothing to report")
            return 2
        produced = set(__import__("json").loads(out_path.read_text(encoding="utf-8")))
    finally:
        plugin_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)

    sys.path.insert(0, str(ROOT / "src"))
    from cas13guide.contract.enums import QCFlag, ReasonCode, ReviewTier

    # Limitation ids are constructed inline rather than declared in an enum, so
    # the list of them is the source itself. Read rather than imported, because
    # importing tells you which ids exist and not which lines can make one.
    import ast

    tree = ast.parse(
        (ROOT / "src/cas13guide/services/limitations.py").read_text(encoding="utf-8")
    )
    limitation_ids = sorted({
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "Limitation"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    })

    vocabularies = {
        "reason code": [code.value for code in ReasonCode],
        "limitation": [f"limitation:{name}" for name in limitation_ids],
        "QC flag": [f"qc:{flag.value}" for flag in QCFlag],
        "review tier": [f"tier:{tier.value}" for tier in ReviewTier],
    }

    print()
    missing_any = False
    for label, every in vocabularies.items():
        missing = [name for name in every if name not in produced]
        seen = len(every) - len(missing)
        print(f"{seen}/{len(every)} {label}s produced by the suite")
        for name in missing:
            print(f"    {name.split(':', 1)[-1]}")
        missing_any = missing_any or bool(missing)

    print()
    if missing_any:
        print("Each one above is something this package can say to a user that")
        print("no test has ever watched it say: dead code, or a live path with")
        print("no coverage at all.")
        return 1

    print("Everything this package can say to a user is said in some test.")
    print("That is a floor, not a guarantee: produced is not the same as")
    print("asserted. verify_mutation_coverage.py is the one that asks whether")
    print("the behaviour is pinned.")
    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
