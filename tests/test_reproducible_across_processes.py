"""The same input must give the same answer in a *different process*.

Every other test in this suite runs inside one interpreter, and that is exactly
where this class of defect hides. Python randomises the hash of ``str`` and
``bytes`` once per process: iteration order over a set of strings is perfectly
stable within a run and varies between runs. A suite that never leaves its own
interpreter agrees with itself forever, while two users on two machines get two
different rankings for the same FASTA and no error to explain it.

The package leans on that ordering in a lot of places -- ``a_candidates`` says
so in as many words ("deterministic rather than dependent on iteration luck"),
``fmindex`` says so, ``conservation`` says so. Nothing checked it.

Measured when this was written: four configurations (plain, structure, all
sites, detection) across ``PYTHONHASHSEED`` 0, 1, 42 and 12345 produce
byte-identical artefacts -- sixteen runs, no differences. The test below keeps
the cheapest corner of that sweep so the suite stays quick; the full sweep is
worth re-running by hand after anything that touches ordering.
"""

import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

#: Two seeds is the minimum that can show the failure at all, and in practice
#: shows it immediately: an order that depends on string hashing differs
#: between essentially any two seeds.
SEEDS = ("0", "12345")


def _design(directory: Path, fasta: Path, seed: str) -> dict[str, str]:
    out = directory / f"out{seed}"
    result = subprocess.run(
        [sys.executable, "-m", "cas13guide.cli", "design",
         "--fasta", str(fasta), "--task", "knockdown",
         "--assay", "cell_culture_knockdown", "--effector", "RfxCas13d",
         "--no-structure", "--no-signal", "--top", "15", "--seed", "11",
         "--out", str(out), "--stem", "run"],
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"design failed under PYTHONHASHSEED={seed}\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return {
        path.name: path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(out.iterdir())
    }


class TestARunIsReproducibleInAnotherProcess(unittest.TestCase):
    def test_every_artefact_is_identical_under_a_different_hash_seed(self):
        rng = random.Random(7)
        sequence = "".join(rng.choice("ACGU") for _ in range(240))

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            fasta = work / "t.fa"
            # Two records, so the run has more than one target to order.
            fasta.write_text(
                f">t1\n{sequence}\n>t2\n{sequence[60:180]}\n", encoding="utf-8"
            )
            runs = {seed: _design(work, fasta, seed) for seed in SEEDS}

        first, second = (runs[seed] for seed in SEEDS)
        self.assertEqual(
            sorted(first), sorted(second), "the two runs produced different files"
        )
        self.assertTrue(first, "fixture produced no artefacts, so it checks nothing")
        for name in sorted(first):
            with self.subTest(artefact=name):
                self.assertEqual(
                    first[name], second[name],
                    f"{name} differs between PYTHONHASHSEED "
                    f"{SEEDS[0]} and {SEEDS[1]}: something downstream is "
                    f"iterating a set of strings rather than a sorted sequence",
                )


if __name__ == "__main__":
    unittest.main()
