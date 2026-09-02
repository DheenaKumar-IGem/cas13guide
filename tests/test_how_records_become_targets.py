"""How a multi-record FASTA becomes a contract, pinned in one place.

There were two implementations of this. The live one is in ``workflow.py``:
the first record is the REQUIRED target and records 1+ are DESIRED, because a
guide that misses a DESIRED target is worth less and a guide that misses the
REQUIRED one fails.

The other was ``fasta.records_to_target_specs``, added in the commit that first
built the `design` command and called by nothing since. It made **every**
record REQUIRED and gave them a different biotype -- and both differences change
verdicts: under its classification a guide covering only the primary transcript
fails a contract it should pass, and the biotype feeds the hazard weighting.

It could not be fixed where it lived. "First record REQUIRED, the rest DESIRED"
is a policy about what a user meant by handing over a FASTA, not a fact about
FASTA parsing, so the function's own layer had no way to be right. It was
removed, and the rule it contradicted is pinned here instead.
"""

import random
import tempfile
import unittest
from pathlib import Path

from helpers import *  # noqa: F401,F403


def _spec_from(records: list[tuple[str, str]]):
    from cas13guide.services.workflow import DesignRequest, run_request

    text = "".join(f">{name}\n{sequence}\n" for name, sequence in records)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "t.fa"
        path.write_text(text, encoding="utf-8")
        return run_request(
            DesignRequest(fasta_path=str(path), task="knockdown",
                          assay="cell_culture_knockdown", effector="RfxCas13d",
                          use_structure=False, use_signal=False)
        ).spec


def _sequences(count: int, length: int = 150) -> list[tuple[str, str]]:
    rng = random.Random(9)
    return [
        (f"t{i + 1}", "".join(rng.choice("ACGU") for _ in range(length)))
        for i in range(count)
    ]


class TestTheFirstRecordIsTheOneThatMustWork(unittest.TestCase):
    def test_a_single_record_is_required(self):
        from cas13guide.contract.enums import TargetClass

        targets = _spec_from(_sequences(1)).contract.targets
        self.assertEqual(len(targets), 1)
        self.assertIs(targets[0].target_class, TargetClass.REQUIRED)

    def test_later_records_are_desired_not_required(self):
        """The difference the removed builder got wrong. Under all-REQUIRED, a
        guide covering only the primary transcript fails a contract it should
        pass."""
        from cas13guide.contract.enums import TargetClass

        targets = _spec_from(_sequences(4)).contract.targets
        self.assertEqual(len(targets), 4)
        self.assertIs(targets[0].target_class, TargetClass.REQUIRED)
        for target in targets[1:]:
            with self.subTest(target=target.target_id):
                self.assertIs(target.target_class, TargetClass.DESIRED)

    def test_exactly_one_record_is_required(self):
        for count in (1, 2, 5):
            with self.subTest(records=count):
                from cas13guide.contract.enums import TargetClass

                targets = _spec_from(_sequences(count)).contract.targets
                required = [
                    t for t in targets if t.target_class is TargetClass.REQUIRED
                ]
                self.assertEqual(len(required), 1)

    def test_the_order_is_the_order_of_the_file(self):
        """"First" has to mean first in the file, or which target must work
        depends on dictionary ordering."""
        records = _sequences(3)
        targets = _spec_from(records).contract.targets
        self.assertEqual(
            [t.target_id for t in targets], [name for name, _ in records]
        )

    def test_the_second_way_of_building_this_is_gone(self):
        """Two ways to turn records into targets, disagreeing on which must
        work, is the drift this file exists to prevent."""
        import cas13guide.services.fasta as fasta

        self.assertFalse(
            hasattr(fasta, "records_to_target_specs"),
            "a second target-spec builder is back; there should be one rule",
        )


if __name__ == "__main__":
    unittest.main()
