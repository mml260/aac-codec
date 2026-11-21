"""Quantisation and PAM-guided bit allocation for one frame.

Band-dependent precision: every band b gets a uniform mid-tread quantiser
whose step Delta_b makes its noise (Delta^2/12 per line) equal to the PAM's
allowed noise a_b raised by a global offset g (dB) shared by all bands:

    Delta_b^2 / 12 = a_b * 10^(g/10),   Delta_b = 2^(sf_b/4)  (sf rounded to 1.5 dB steps)

So the noise spectrum follows the masking threshold and g sets how far above
(g > 0) or below (g < 0) the threshold the whole frame sits: the classic
"threshold adaptation to the bit budget" (constant noise-to-mask ratio).
The bit cost of every candidate g is computed exactly (same code lengths as
the bitstream writer), and a bisection on g finds the finest quantisation
whose frame size fits the target. A band whose lines all quantise to 0 is
switched off and costs only its 1-bit flag.

Every function takes the frame's coding layout (blockswitch.Layout): the 49
long bands by default, or the (group, short band) pairs of a short frame,
whose coefficients are given in the layout's coded (interleaved) order.
"""
import numpy as np

from .blockswitch import LONG_LAYOUT
from .bitstream import side_info_bits

SF_MIN, SF_MAX = -100, 60
K_RANGE = np.arange(0, 40)


def _sum_bands(v, layout, axis=-1):
    return np.add.reduceat(v, layout.offsets[:-1], axis=axis)


def scalefactors(allow, g_db):
    step = np.sqrt(12.0 * allow * 10 ** (g_db / 10.0))
    sf = np.round(4.0 * np.log2(np.maximum(step, 1e-300)))
    return np.clip(sf, SF_MIN, SF_MAX).astype(np.int64)


def quantize(X, sf, layout=LONG_LAYOUT):
    step = 2.0 ** (sf[layout.band_of_line] / 4.0)
    return np.round(X / step).astype(np.int64)


def dequantize(q, sf, layout=LONG_LAYOUT):
    return q * 2.0 ** (sf[layout.band_of_line] / 4.0)


def rice_band_costs(q, layout=LONG_LAYOUT):
    """Best Rice parameter and its exact payload bits for every band."""
    m = np.abs(q)
    nz = _sum_bands((m > 0).astype(np.int64), layout)
    shifted = _sum_bands(m[:, None] >> K_RANGE[None, :], layout, axis=0)       # (bands, K)
    cost = shifted + layout.width[:, None] * (K_RANGE[None, :] + 1) + nz[:, None]
    kb = np.argmin(cost, axis=1)
    return K_RANGE[kb], cost[np.arange(layout.n_bands), kb]


def evaluate(X, allow, g_db, layout=LONG_LAYOUT, header_bits=0):
    """Quantise one frame at offset g; header_bits = window-sequence/grouping fields (v2)."""
    sf = scalefactors(allow, g_db)
    q = quantize(X, sf, layout)
    active = _sum_bands(np.abs(q), layout) > 0
    k, rice = rice_band_costs(q, layout)
    q[~active[layout.band_of_line]] = 0
    side = header_bits + side_info_bits(active, sf, k, layout)
    payload = int(np.sum(rice[active]))
    return dict(g=g_db, sf=sf, k=k, q=q, active=active, side=side, payload=payload,
                bits=side + payload)


def allocate(X, allow, target_bits, g_lo=-60.0, g_hi=90.0, iters=24, layout=LONG_LAYOUT,
             header_bits=0):
    """Smallest global offset g (finest quantisation) whose exact size <= target_bits."""
    ev = lambda g: evaluate(X, allow, g, layout, header_bits)  # noqa: E731
    best = ev(g_hi)
    if best["bits"] > target_bits:
        return best  # cannot go coarser than this; caller accounts for overshoot
    lo, hi = g_lo, g_hi
    cand = ev(lo)
    if cand["bits"] <= target_bits:
        return cand
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r = ev(mid)
        if r["bits"] <= target_bits:
            hi, best = mid, r
        else:
            lo = mid
        if hi - lo < 0.02:
            break
    return best


def perceptual_entropy(X, allow, layout=LONG_LAYOUT):
    """Bits needed to bring every band's noise down to its allowance (per frame).

    PE = sum_b width_b * max(0, 0.5*log2(e_b / a_b)), with e_b the mean
    energy per line of band b: the familiar ~6 dB-per-bit rule applied
    band-wise. Reported in the encoder statistics (not used for decisions).
    """
    e = _sum_bands(X ** 2, layout, axis=-1) / layout.width
    return np.sum(layout.width * np.maximum(0.0, 0.5 * np.log2(np.maximum(e, 1e-300) / allow)), axis=-1)
