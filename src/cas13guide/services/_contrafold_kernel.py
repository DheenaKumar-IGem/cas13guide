r"""The CONTRAfold Viterbi kernel, vendored so that it can return its score.

One function, copied from ``gpu_contrafold.cpu.viterbi`` (MIT -- see
THIRD_PARTY_LICENSES.md) with a single changed return statement. Generated
rather than transcribed, so the copy cannot drift by a typo.
"""

import numpy as np
from numba import njit

from gpu_contrafold.cpu import C_MAX_SINGLE, D_HAIRPIN, HALF, NEG

@njit(cache=True)
def viterbi_scored(s, L, forced, bp, stack, tm, hc, dl, dr, hp_cum, cs, i11, b01,
            mb, mu, mp, eu, ep, canon):
    r"""Vendored from ``gpu_contrafold.cpu.viterbi`` (MIT), with ONE change.

    Upstream returns only ``pair`` and throws away the optimum it just
    computed in ``F5[L]``. That scalar *is* the CONTRAfold score, and it is
    the only thing the Arc model consumes, so this returns ``(pair, F5[L])``.

    Copied rather than patched at import time. Every Arc prediction rests on
    this number, and a textual patch against a 0.1.0 single-author package
    would break silently on upgrade. The parity test over 399 known folds is
    what proves the copy still agrees with the model it came from.

    Do not hand-edit this to "improve" it. Its only job is to agree.
    """
    NEGv = NEG
    FC = np.full((L + 2, L + 2), NEGv)
    FM = np.full((L + 2, L + 2), NEGv)
    FM1 = np.full((L + 2, L + 2), NEGv)
    F5 = np.full(L + 2, NEGv)

    for i in range(L, -1, -1):
        for j in range(i, L + 1):
            FM2 = NEGv
            for k in range(i + 1, j):
                if FM1[i, k] > HALF and FM[k, j] > HALF:
                    v = FM1[i, k] + FM[k, j]
                    if v > FM2:
                        FM2 = v
            if 0 < i and j < L and canon[s[i], s[j + 1]] == 1 and forced[i] == 0 and forced[j + 1] == 0:
                best = NEGv
                jB = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
                v = jB + hp_cum[j - i if j - i <= D_HAIRPIN else D_HAIRPIN]
                if v > best:
                    best = v
                pmax = i + C_MAX_SINGLE
                if pmax > j:
                    pmax = j
                for p in range(i, pmax + 1):
                    l1 = p - i
                    qmin = p + 2
                    alt = p - i + j - C_MAX_SINGLE
                    if alt > qmin:
                        qmin = alt
                    for q in range(j, qmin - 1, -1):
                        l2 = j - q
                        if canon[s[p + 1], s[q]] == 1 and forced[p + 1] == 0 and forced[q] == 0 and FC[p + 1, q - 1] > HALF:
                            if p == i and q == j:
                                e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                            else:
                                jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                snuc = 0.0
                                if l1 == 0 and l2 == 1:
                                    snuc = b01[s[j]]
                                elif l1 == 1 and l2 == 0:
                                    snuc = b01[s[i + 1]]
                                elif l1 == 1 and l2 == 1:
                                    snuc = i11[s[i + 1], s[j]]
                                e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB + jB2 + snuc
                            v = e + FC[p + 1, q - 1]
                            if v > best:
                                best = v
                if FM2 > HALF:
                    jA = hc[s[i], s[j + 1]]
                    if i < L:
                        jA += dl[s[i], s[j + 1], s[i + 1]]
                    if j > 0:
                        jA += dr[s[i], s[j + 1], s[j]]
                    v = FM2 + jA + mp + mb
                    if v > best:
                        best = v
                FC[i, j] = best
            if 0 < i and i + 2 <= j and j < L:
                best = NEGv
                if canon[s[i + 1], s[j]] == 1 and forced[i + 1] == 0 and forced[j] == 0 and FC[i + 1, j - 1] > HALF:
                    jAji = hc[s[j], s[i + 1]]
                    if j < L:
                        jAji += dl[s[j], s[i + 1], s[j + 1]]
                    if i > 0:
                        jAji += dr[s[j], s[i + 1], s[i]]
                    v = FC[i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]]
                    if v > best:
                        best = v
                if FM1[i + 1, j] > HALF:
                    v = FM1[i + 1, j] + mu
                    if v > best:
                        best = v
                FM1[i, j] = best
            if 0 < i and i + 2 <= j and j < L:
                best = NEGv
                if FM2 > HALF and FM2 > best:
                    best = FM2
                if FM[i, j - 1] > HALF:
                    v = FM[i, j - 1] + mu
                    if v > best:
                        best = v
                if FM1[i, j] > HALF and FM1[i, j] > best:
                    best = FM1[i, j]
                FM[i, j] = best

    F5[0] = 0.0
    for j in range(1, L + 1):
        best = F5[j - 1] + eu
        for k in range(0, j):
            if canon[s[k + 1], s[j]] == 1 and forced[k + 1] == 0 and forced[j] == 0 and FC[k + 1, j - 1] > HALF and F5[k] > HALF:
                jA = hc[s[j], s[k + 1]]
                if j < L:
                    jA += dl[s[j], s[k + 1], s[j + 1]]
                if k > 0:
                    jA += dr[s[j], s[k + 1], s[k]]
                v = F5[k] + FC[k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA
                if v > best:
                    best = v
        F5[j] = best

    # ---- argmax traceback ----
    pair = np.full(L + 2, -1, np.int64)
    EPS = 1e-6
    st_t = np.empty(4 * L + 16, np.int64)
    st_i = np.empty(4 * L + 16, np.int64)
    st_j = np.empty(4 * L + 16, np.int64)
    st_t[0] = 0; st_i[0] = 0; st_j[0] = L
    sp = 1
    while sp > 0:
        sp -= 1
        typ = st_t[sp]; i = st_i[sp]; j = st_j[sp]
        if typ == 0:                       # F5(j)
            if j == 0:
                continue
            val = F5[j]
            done = False
            if F5[j - 1] > HALF and abs(F5[j - 1] + eu - val) <= EPS:
                st_t[sp] = 0; st_i[sp] = 0; st_j[sp] = j - 1; sp += 1; done = True
            if not done:
                for k in range(0, j):
                    if canon[s[k + 1], s[j]] == 1 and forced[k + 1] == 0 and forced[j] == 0 and FC[k + 1, j - 1] > HALF and F5[k] > HALF:
                        jA = hc[s[j], s[k + 1]]
                        if j < L:
                            jA += dl[s[j], s[k + 1], s[j + 1]]
                        if k > 0:
                            jA += dr[s[j], s[k + 1], s[k]]
                        v = F5[k] + FC[k + 1, j - 1] + ep + bp[s[k + 1], s[j]] + jA
                        if abs(v - val) <= EPS:
                            pair[k + 1] = j; pair[j] = k + 1
                            st_t[sp] = 1; st_i[sp] = k + 1; st_j[sp] = j - 1; sp += 1
                            st_t[sp] = 0; st_i[sp] = 0; st_j[sp] = k; sp += 1
                            done = True; break
        elif typ == 1:                     # FC(i,j): pair (i,j+1) already set
            val = FC[i, j]
            jB = hc[s[i], s[j + 1]] + tm[s[i], s[j + 1], s[i + 1], s[j]]
            d = j - i
            done = abs(jB + hp_cum[d if d <= D_HAIRPIN else D_HAIRPIN] - val) <= EPS
            if not done:
                pmax = i + C_MAX_SINGLE
                if pmax > j:
                    pmax = j
                for p in range(i, pmax + 1):
                    l1 = p - i
                    qmin = p + 2
                    alt = p - i + j - C_MAX_SINGLE
                    if alt > qmin:
                        qmin = alt
                    for q in range(j, qmin - 1, -1):
                        l2 = j - q
                        if canon[s[p + 1], s[q]] == 1 and forced[p + 1] == 0 and forced[q] == 0 and FC[p + 1, q - 1] > HALF:
                            if p == i and q == j:
                                e = bp[s[i + 1], s[j]] + stack[s[i], s[j + 1], s[i + 1], s[j]]
                            else:
                                jB2 = hc[s[q], s[p + 1]] + tm[s[q], s[p + 1], s[q + 1], s[p]]
                                snuc = 0.0
                                if l1 == 0 and l2 == 1:
                                    snuc = b01[s[j]]
                                elif l1 == 1 and l2 == 0:
                                    snuc = b01[s[i + 1]]
                                elif l1 == 1 and l2 == 1:
                                    snuc = i11[s[i + 1], s[j]]
                                e = cs[l1, l2] + bp[s[p + 1], s[q]] + jB + jB2 + snuc
                            if abs(e + FC[p + 1, q - 1] - val) <= EPS:
                                pair[p + 1] = q; pair[q] = p + 1
                                st_t[sp] = 1; st_i[sp] = p + 1; st_j[sp] = q - 1; sp += 1
                                done = True; break
                    if done:
                        break
            if not done:
                jA = hc[s[i], s[j + 1]]
                if i < L:
                    jA += dl[s[i], s[j + 1], s[i + 1]]
                if j > 0:
                    jA += dr[s[i], s[j + 1], s[j]]
                for k in range(i + 1, j):
                    if FM1[i, k] > HALF and FM[k, j] > HALF and abs(FM1[i, k] + FM[k, j] + jA + mp + mb - val) <= EPS:
                        st_t[sp] = 3; st_i[sp] = i; st_j[sp] = k; sp += 1
                        st_t[sp] = 2; st_i[sp] = k; st_j[sp] = j; sp += 1
                        break
        elif typ == 2:                     # FM(i,j)
            val = FM[i, j]
            done = False
            for k in range(i + 1, j):
                if FM1[i, k] > HALF and FM[k, j] > HALF and abs(FM1[i, k] + FM[k, j] - val) <= EPS:
                    st_t[sp] = 3; st_i[sp] = i; st_j[sp] = k; sp += 1
                    st_t[sp] = 2; st_i[sp] = k; st_j[sp] = j; sp += 1
                    done = True; break
            if not done and FM[i, j - 1] > HALF and abs(FM[i, j - 1] + mu - val) <= EPS:
                st_t[sp] = 2; st_i[sp] = i; st_j[sp] = j - 1; sp += 1; done = True
            if not done and FM1[i, j] > HALF and abs(FM1[i, j] - val) <= EPS:
                st_t[sp] = 3; st_i[sp] = i; st_j[sp] = j; sp += 1
        else:                              # FM1(i,j)
            val = FM1[i, j]
            done = False
            if canon[s[i + 1], s[j]] == 1 and forced[i + 1] == 0 and forced[j] == 0 and FC[i + 1, j - 1] > HALF:
                jAji = hc[s[j], s[i + 1]]
                if j < L:
                    jAji += dl[s[j], s[i + 1], s[j + 1]]
                if i > 0:
                    jAji += dr[s[j], s[i + 1], s[i]]
                if abs(FC[i + 1, j - 1] + jAji + mp + bp[s[i + 1], s[j]] - val) <= EPS:
                    pair[i + 1] = j; pair[j] = i + 1
                    st_t[sp] = 1; st_i[sp] = i + 1; st_j[sp] = j - 1; sp += 1
                    done = True
            if not done and FM1[i + 1, j] > HALF and abs(FM1[i + 1, j] + mu - val) <= EPS:
                st_t[sp] = 3; st_i[sp] = i + 1; st_j[sp] = j; sp += 1
    return pair, F5[L]
