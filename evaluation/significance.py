"""Paired significance tests on per-attack pre-echo (NumPy only).

Unit of analysis: one detected attack (attacks are >= 50 ms apart, so their
pre-echo windows do not overlap). For each attack the difference
d = pre-echo(model A) - pre-echo(model B) in dB is formed; d < 0 means A has
less pre-echo.

  * Sign test: exact two-sided binomial test on the number of attacks with
    d < 0 (ties dropped).
  * Wilcoxon signed-rank test: exact two-sided p-value for n <= 25 (dynamic
    programming over rank sums), normal approximation with tie correction
    above that.
  * Holm correction across the 9 excerpt x rate tests of one comparison.
"""
from math import comb

import numpy as np


def sign_test(d):
    d = np.asarray(d)
    d = d[d != 0]
    n = len(d)
    k = int(np.sum(d < 0))
    lo = min(k, n - k)
    p = sum(comb(n, i) for i in range(lo + 1)) / 2 ** n * 2
    return dict(n=n, wins=k, p=min(1.0, p))


def _rank(a):
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return r


def wilcoxon(d):
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return dict(n=0, W=0.0, p=1.0)
    r = _rank(np.abs(d))
    w_pos = float(np.sum(r[d > 0]))
    if n <= 25 and np.allclose(r, np.round(r)):
        # exact null distribution of W+ over integer ranks 1..n
        total = n * (n + 1) // 2
        dist = np.zeros(total + 1)
        dist[0] = 1
        for k in range(1, n + 1):          # number of subsets of {1..k} with each rank sum
            new = dist.copy()
            new[k:] += dist[: total + 1 - k]
            dist = new
        dist = dist / dist.sum()
        cdf_lo = dist[: int(np.floor(min(w_pos, total - w_pos))) + 1].sum()
        p = min(1.0, 2 * cdf_lo)
    else:
        mean = n * (n + 1) / 4
        _, counts = np.unique(np.abs(d), return_counts=True)
        var = n * (n + 1) * (2 * n + 1) / 24 - np.sum(counts ** 3 - counts) / 48
        z = (w_pos - mean) / np.sqrt(var)
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return dict(n=n, W=w_pos, p=float(p))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj
