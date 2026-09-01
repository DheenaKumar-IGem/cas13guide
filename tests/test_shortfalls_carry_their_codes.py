"""A shortfall is reported twice, and the two copies must agree.

The orchestrator records each budget shortfall as prose for a human
(``telemetry.shortfalls``) and as a ``ReasonCode`` for a machine
(``telemetry.reason_codes``, exported in the provenance ledger as
``shortfall_codes``). Two representations of one fact, appended on adjacent
lines, is exactly the shape that drifts: delete or move one and everything
still runs, the report still reads correctly to a person, and a caller
switching on the codes silently stops seeing the shortfall.

``test_orchestrator`` already pins the prose -- that the loops fire, that they
are bounded, and that the message is recorded once per run rather than once per
iteration. What nothing checked was the other half: that the code is recorded
at all. This pins the pair.
"""

import unittest

from helpers import *  # noqa: F401,F403
from test_orchestrator import run, world


class TestEveryShortfallHasItsMachineReadableCode(unittest.TestCase):
    """One run that exhausts both budgets.

    ``world(include_target=False)`` searches a transcriptome the guides are not
    in, so every candidate is zero-hit and the search escalation runs out;
    ``step=4`` leaves room for the panel retile loop to run out too. Both
    shortfalls therefore come from one run, and the outer loop goes round three
    times, which is what makes the once-per-run claim falsifiable.
    """

    @classmethod
    def setUpClass(cls):
        cls.spec = make_spec()
        cls.report = run(cls.spec, world(include_target=False), step=4)
        cls.codes = cls._telemetry(cls.report)["detail"]["shortfall_codes"]

    @staticmethod
    def _telemetry(report) -> dict:
        for record in report.provenance["records"]:
            if record.get("kind") == "control_flow" and record.get("name") == "telemetry":
                return record
        raise AssertionError("no control_flow/telemetry record in the ledger")

    def test_the_search_shortfall_ships_with_its_code(self):
        from cas13guide.contract.enums import ReasonCode

        prose = [s for s in self.report.shortfalls if "search escalation" in s]
        self.assertTrue(prose, f"fixture produced no search shortfall: {self.report.shortfalls}")
        self.assertIn(ReasonCode.BUDGET_EXHAUSTED_SEARCH.value, self.codes)

    def test_the_panel_shortfall_ships_with_its_code(self):
        from cas13guide.contract.enums import ReasonCode

        prose = [s for s in self.report.shortfalls if "no feasible panel" in s]
        self.assertTrue(prose, f"fixture produced no panel shortfall: {self.report.shortfalls}")
        self.assertIn(ReasonCode.BUDGET_EXHAUSTED_NO_FEASIBLE_PANEL.value, self.codes)
        self.assertIn("t1", prose[0], "a shortfall that names no target is not actionable")

    def test_each_code_is_recorded_once_not_once_per_iteration(self):
        """The outer loop goes round three times here -- two retiles -- and the
        search branch is reached on each. Without the guard the code would be
        recorded three times and a caller counting them would over-report."""
        self.assertEqual(len(self.codes), len(set(self.codes)), f"duplicated: {self.codes}")

    def test_the_loop_really_did_go_round_more_than_once(self):
        """Otherwise the test above is satisfied by a run that never repeats,
        and proves nothing about the guard it is named for."""
        self.assertEqual(
            self._telemetry(self.report)["detail"]["panel_retiles"],
            self.spec.budget.max_panel_retiles,
        )

    def test_every_code_is_a_real_reason_code(self):
        from cas13guide.contract.enums import ReasonCode

        self.assertTrue(self.codes)
        known = {c.value for c in ReasonCode}
        for code in self.codes:
            self.assertIn(code, known)


if __name__ == "__main__":
    unittest.main()
