"""Polyphase rational resampler (NumPy only).

Equivalent in design to scipy.signal.resample_poly with its default
Kaiser(beta=5) window and half-length 10*max(up, down): the signal is
(conceptually) upsampled by `up`, low-pass filtered at the lower of the two
Nyquist rates, and downsampled by `down`. Only the needed output samples are
computed, using one sub-filter per polyphase branch.
"""
from math import gcd
import numpy as np


def design_filter(up, down, beta=5.0):
    max_rate = max(up, down)
    half_len = 10 * max_rate
    n = np.arange(2 * half_len + 1) - half_len
    cutoff = 1.0 / max_rate  # relative to the upsampled Nyquist frequency
    h = cutoff * np.sinc(cutoff * n) * np.kaiser(2 * half_len + 1, beta)
    h /= h.sum()  # unity DC gain (as scipy.signal.firwin)
    return h * up, half_len


def resample_poly(x, up, down, chunk=65536):
    g = gcd(up, down)
    up //= g
    down //= g
    x = np.asarray(x, dtype=np.float64)
    h, half_len = design_filter(up, down)
    L = len(h)
    n_out = -(-len(x) * up // down)  # ceil
    K = -(-L // up)  # taps per polyphase branch
    # Sub-filters: branch p uses taps h[p], h[p+up], ...
    hp = np.zeros((up, K))
    for p in range(up):
        taps = h[p::up]
        hp[p, : len(taps)] = taps
    # Pad input so every index below is valid.
    pad_front = K
    xp = np.concatenate([np.zeros(pad_front), x, np.zeros(K + 2)])
    y = np.empty(n_out)
    for start in range(0, n_out, chunk):
        m = np.arange(start, min(start + chunk, n_out))
        n0 = m * down + half_len  # index into the virtual upsampled signal
        phase = n0 % up
        base = n0 // up  # input index paired with tap 0 of the branch
        idx = base[:, None] - np.arange(K)[None, :] + pad_front
        y[m] = np.einsum("ij,ij->i", xp[idx], hp[phase])
    return y
