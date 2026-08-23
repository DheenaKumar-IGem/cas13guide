"""Linear-time suffix array construction (SA-IS).

Replaces the O(n log^2 n) prefix-doubling build, which cost ~20 s on a 3 Mb
corpus -- fine for a demo, useless for a transcriptome.

SA-IS (Nong, Zhang & Chan, 2009) is O(n): it classifies suffixes as S- or
L-type, sorts the LMS (leftmost-S) substrings by induced sorting, recurses on
the reduced string only when LMS names collide, and induces the full order
from the sorted LMS suffixes.

The public entry point takes a string and returns suffixes ordered exactly as
Python's own string comparison would order them, for any input. It gets there
by mapping characters to ranks ``1..k`` in sorted order and appending an
integer ``0`` sentinel that no real character can equal -- so the caller never
has to guarantee a unique smallest terminator, and the result is directly
comparable with a brute-force sort.
"""

from __future__ import annotations


def _classify(s: list[int], n: int) -> list[bool]:
    """``True`` marks an S-type suffix (smaller than its right neighbour)."""
    is_s = [False] * n
    is_s[n - 1] = True
    for i in range(n - 2, -1, -1):
        if s[i] < s[i + 1]:
            is_s[i] = True
        elif s[i] == s[i + 1]:
            is_s[i] = is_s[i + 1]
    return is_s


def _bucket_sizes(s: list[int], alphabet_size: int) -> list[int]:
    counts = [0] * alphabet_size
    for c in s:
        counts[c] += 1
    return counts


def _bucket_heads(counts: list[int]) -> list[int]:
    heads, total = [0] * len(counts), 0
    for c, count in enumerate(counts):
        heads[c] = total
        total += count
    return heads


def _bucket_tails(counts: list[int]) -> list[int]:
    tails, total = [0] * len(counts), 0
    for c, count in enumerate(counts):
        total += count
        tails[c] = total
    return tails


def _induce(
    s: list[int],
    n: int,
    is_s: list[bool],
    counts: list[int],
    lms_order: list[int],
) -> list[int]:
    """Induced sort: seed with LMS suffixes, then sweep L then S."""
    sa = [-1] * n

    tails = _bucket_tails(counts)
    for i in reversed(lms_order):
        c = s[i]
        tails[c] -= 1
        sa[tails[c]] = i

    heads = _bucket_heads(counts)
    for i in range(n):
        j = sa[i] - 1
        if sa[i] > 0 and not is_s[j]:
            c = s[j]
            sa[heads[c]] = j
            heads[c] += 1

    tails = _bucket_tails(counts)
    for i in range(n - 1, -1, -1):
        j = sa[i] - 1
        if sa[i] > 0 and is_s[j]:
            c = s[j]
            tails[c] -= 1
            sa[tails[c]] = j

    return sa


def sais(s: list[int], alphabet_size: int) -> list[int]:
    """Suffix array of an integer sequence whose last element is a unique 0."""
    n = len(s)
    if n == 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [1, 0] if s[0] > s[1] else [0, 1]

    is_s = _classify(s, n)

    def is_lms(i: int) -> bool:
        return i > 0 and is_s[i] and not is_s[i - 1]

    counts = _bucket_sizes(s, alphabet_size)
    lms = [i for i in range(1, n) if is_lms(i)]

    sa = _induce(s, n, is_s, counts, lms)
    sorted_lms = [i for i in sa if is_lms(i)]

    # Name LMS substrings; equal substrings share a name.
    names = [-1] * n
    name = 0
    if sorted_lms:
        names[sorted_lms[0]] = 0
        previous = sorted_lms[0]
        for current in sorted_lms[1:]:
            different = False
            offset = 0
            while True:
                a, b = previous + offset, current + offset
                if a >= n or b >= n:
                    different = True
                    break
                a_lms, b_lms = is_lms(a), is_lms(b)
                if offset > 0 and (a_lms or b_lms):
                    if a_lms and b_lms:
                        break  # both substrings ended together: identical
                    different = True
                    break
                if s[a] != s[b] or is_s[a] != is_s[b]:
                    different = True
                    break
                offset += 1
            if different:
                name += 1
            names[current] = name
            previous = current

    if lms:
        reduced = [names[i] for i in lms]
        if name + 1 == len(lms):
            # All names distinct: the order is already determined.
            lms_sorted = [0] * len(lms)
            for position, label in zip(lms, reduced):
                lms_sorted[label] = position
        else:
            lms_sorted = [lms[i] for i in sais(reduced, name + 1)]
    else:
        lms_sorted = []

    return _induce(s, n, is_s, counts, lms_sorted)


def build_suffix_array(text: str) -> list[int]:
    """Suffix array of ``text``, ordered as Python would order the suffixes.

    Characters are mapped to ranks ``1..k`` and an integer ``0`` sentinel is
    appended, so no assumption is made about the caller's terminator. The
    sentinel's own suffix sorts first and is stripped before returning.
    """
    n = len(text)
    if n == 0:
        return []

    alphabet = sorted(set(text))
    rank = {c: i + 1 for i, c in enumerate(alphabet)}
    mapped = [rank[c] for c in text]
    mapped.append(0)

    sa = sais(mapped, len(alphabet) + 1)
    return sa[1:]  # drop the appended sentinel suffix
