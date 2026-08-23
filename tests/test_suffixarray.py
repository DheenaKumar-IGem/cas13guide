"""SA-IS: the hard part is the recursion, so exercise it deliberately."""

import random
import unittest

from helpers import *  # noqa: F401,F403

from cas13guide.services.suffixarray import build_suffix_array, sais


def brute(text: str) -> list[int]:
    return sorted(range(len(text)), key=lambda i: text[i:])


class TestCorrectness(unittest.TestCase):
    def test_randomized_across_alphabets_and_lengths(self):
        rng = random.Random(1)
        for _ in range(600):
            alphabet = rng.choice(["AUGC", "AB", "A", "AUGC#$", "abcdefgh"])
            text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
            self.assertEqual(build_suffix_array(text), brute(text), repr(text))

    def test_classic_worst_cases(self):
        for text in (
            "banana",
            "mississippi",
            "abracadabra",
            "aaaaaaaaaa",
            "abababababab",
            "aabaaabaaaab",
            "AUAUAUAUAUAU$",
        ):
            self.assertEqual(build_suffix_array(text), brute(text), text)

    def test_edge_cases(self):
        self.assertEqual(build_suffix_array(""), [])
        self.assertEqual(build_suffix_array("A"), [0])
        self.assertEqual(build_suffix_array("AA"), [1, 0])
        self.assertEqual(build_suffix_array("AB"), [0, 1])
        self.assertEqual(build_suffix_array("BA"), [1, 0])

    def test_no_terminator_required_from_the_caller(self):
        """Callers must not have to supply a unique smallest sentinel."""
        for text in ("AUGCAUGC", "AUGC#AUGC", "$$$$", "####"):
            self.assertEqual(build_suffix_array(text), brute(text), text)

    def test_every_suffix_appears_exactly_once(self):
        rng = random.Random(5)
        text = "".join(rng.choice("AUGC") for _ in range(500))
        sa = build_suffix_array(text)
        self.assertEqual(sorted(sa), list(range(len(text))))

    def test_result_is_strictly_increasing_in_suffix_order(self):
        rng = random.Random(6)
        text = "".join(rng.choice("AUG") for _ in range(400))
        sa = build_suffix_array(text)
        for a, b in zip(sa, sa[1:]):
            self.assertLess(text[a:], text[b:])


class TestRecursionPath(unittest.TestCase):
    """Strings whose LMS substrings collide force the recursive branch.

    The non-recursive path handles most random input, so a suite that only
    tests random RNA never executes the algorithm's hardest code.
    """

    def _forces_recursion(self, text: str) -> bool:
        """True when LMS naming produces duplicates (so SA-IS must recurse)."""
        alphabet = sorted(set(text))
        rank = {c: i + 1 for i, c in enumerate(alphabet)}
        s = [rank[c] for c in text] + [0]
        n = len(s)
        is_s = [False] * n
        is_s[n - 1] = True
        for i in range(n - 2, -1, -1):
            is_s[i] = s[i] < s[i + 1] or (s[i] == s[i + 1] and is_s[i + 1])
        lms = [i for i in range(1, n) if is_s[i] and not is_s[i - 1]]
        substrings = []
        for idx, start in enumerate(lms):
            end = lms[idx + 1] + 1 if idx + 1 < len(lms) else n
            substrings.append(tuple(s[start:end]))
        return len(set(substrings)) < len(substrings)

    def test_repetitive_inputs_actually_recurse(self):
        repetitive = "abababababababab"
        self.assertTrue(self._forces_recursion(repetitive))
        self.assertEqual(build_suffix_array(repetitive), brute(repetitive))

    def test_a_batch_of_recursion_forcing_strings(self):
        rng = random.Random(9)
        forced = 0
        for _ in range(300):
            unit = "".join(rng.choice("ab") for _ in range(rng.randint(2, 5)))
            text = unit * rng.randint(3, 8)
            if self._forces_recursion(text):
                forced += 1
                self.assertEqual(build_suffix_array(text), brute(text), text)
        self.assertGreater(forced, 50, "fixture failed to exercise the recursion")

    def test_tandem_repeats_of_rna(self):
        for unit in ("AUGC", "AAUU", "AUAUG", "GGGGA"):
            text = unit * 12
            self.assertEqual(build_suffix_array(text), brute(text), text)


class TestIntegerEntryPoint(unittest.TestCase):
    def test_sais_requires_a_sentinel_and_handles_short_inputs(self):
        self.assertEqual(sais([], 1), [])
        self.assertEqual(sais([0], 1), [0])
        self.assertEqual(sais([1, 0], 2), [1, 0])
        self.assertEqual(sais([2, 1, 0], 3), [2, 1, 0])


if __name__ == "__main__":
    unittest.main()
