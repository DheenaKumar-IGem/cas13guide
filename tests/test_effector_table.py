"""The ortholog table, checked against itself.

`EFFECTORS` is hand-written published constants and every run reads it for
geometry. Two tests reach into it for `spacer_length_range`; nothing checked
that the table is coherent, and each of the three properties below is a real
defect rather than a tidiness rule.
"""

import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.contract.effectors import EFFECTORS

RNA = set("ACGU")


class TestEveryOrthologsParametersAgreeWithEachOther(unittest.TestCase):
    def test_there_are_orthologs_to_check(self):
        """Or every assertion below is true of an empty table."""
        self.assertGreater(len(EFFECTORS), 0)

    def test_every_direct_repeat_is_rna(self):
        """The spacer:DR interaction scan reads these base by base.

        A stray character would be scanned as though it were a base, and the
        flip rate that scan reports is quoted in the README.
        """
        for effector, params in EFFECTORS.items():
            with self.subTest(effector=effector.value):
                self.assertTrue(params.direct_repeat)
                self.assertEqual(
                    set(params.direct_repeat) - RNA, set(),
                    f"{effector.value} direct repeat is not over AUGC",
                )

    def test_the_preferred_length_is_one_of_the_canonical_ones(self):
        """`is_canonical_length` draws the line between validated and merely
        tolerated. A default sitting outside it would make every run that does
        not pass --spacer-length start out of domain."""
        for effector, params in EFFECTORS.items():
            with self.subTest(effector=effector.value):
                self.assertIn(
                    params.preferred_spacer_length, params.spacer_lengths,
                    f"{effector.value} prefers a length it does not call "
                    f"canonical",
                )

    def test_every_canonical_length_is_one_the_ortholog_tolerates(self):
        """`spacer_lengths` is the published set and `spacer_length_range` is
        what the ortholog will accept, so the first has to sit inside the
        second -- otherwise `supports_length` refuses a length the same table
        calls canonical."""
        for effector, params in EFFECTORS.items():
            low, high = params.spacer_length_range
            for length in params.spacer_lengths:
                with self.subTest(effector=effector.value, length=length):
                    self.assertTrue(
                        params.supports_length(length),
                        f"{effector.value} calls {length} canonical and "
                        f"tolerates only {low}-{high}",
                    )

    def test_the_tolerated_range_is_not_inverted(self):
        for effector, params in EFFECTORS.items():
            low, high = params.spacer_length_range
            with self.subTest(effector=effector.value):
                self.assertLessEqual(low, high)


if __name__ == "__main__":
    unittest.main()
