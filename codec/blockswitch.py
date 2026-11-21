"""Block switching (AAC-style window switching) for the optional v2 mode.

The required codec uses one 2048-sample sine window on every frame. A long
window spreads quantisation noise over 42.7 ms, so noise from an attack leaks
into the quiet part before it (pre-echo). AAC's fix, reproduced here, is to
switch frames that contain an attack to eight short MDCTs of 256 samples
(5.3 ms), with transition windows so perfect reconstruction is kept.

Every frame still has a hop of 1024 samples and 1024 coefficients, and
frame f still spans padded samples [fM, fM + 2M) (mdct.py's framing).
Window sequences (AAC numbering):

  ONLY_LONG    sine window, 2048 samples (the v1 window)
  LONG_START   long sine rise | 1 for 448 | short sine fall (128) | 0 for 448
  EIGHT_SHORT  8 sine windows of 256 at offsets 448 + 128 w (w = 0..7),
               each an orthonormal 256 -> 128 MDCT; 8 x 128 = 1024 coefficients
  LONG_STOP    time reverse of LONG_START

Allowed transitions (the half-windows that overlap must be time-reversed
copies of each other and satisfy Princen-Bradley):
  ONLY_LONG / LONG_STOP  ->  ONLY_LONG or LONG_START
  LONG_START / EIGHT_SHORT  ->  EIGHT_SHORT or LONG_STOP

Choosing the sequence (encoder only)
  An attack at original sample t is assigned to frame f = floor((t + 448) / M),
  i.e. fM - 448 <= t < fM + 576. That places it inside the short-window region
  of frame f (which covers fM - 576 .. fM + 576) and *after* the end of the
  preceding LONG_START window (which reaches fM - 448), so no long window
  contains the attack's onset. A single long frame between two short frames
  would have to be both a stop and a start window, which AAC does not have,
  so such a frame is made short too.

Short frames: bands and grouping
  Each short window has the 14 AAC short-window scalefactor bands for 48 kHz.
  Consecutive short windows with similar energy are grouped and share one
  scalefactor per band (AAC's scale_factor_grouping, 7 bits). The coded
  order is AAC's interleaving: for each group, for each band, the lines of
  that band from every window of the group. A "coding band" is one
  (group, band) pair.
"""
from functools import lru_cache

import numpy as np

from . import M, N_WIN
from . import mdct as _mdct
from .bands import SFB_OFFSETS

ONLY_LONG, LONG_START, EIGHT_SHORT, LONG_STOP = 0, 1, 2, 3
SEQ_NAMES = {ONLY_LONG: "long", LONG_START: "start", EIGHT_SHORT: "short", LONG_STOP: "stop"}
N_SHORT = 256                     # short window length
M_S = N_SHORT // 2                # 128 coefficients per short window
N_SUBWIN = M // M_S               # 8 short windows per frame
SHORT_OFFSET = N_WIN // 4 - M_S // 2   # 448: first short window's start inside the frame

# AAC short-window scalefactor bands at 48 kHz (swb_offset_short_48), 14 bands
SFB_OFFSETS_SHORT = np.array([0, 4, 8, 12, 16, 20, 28, 36, 44, 56, 68, 80, 96, 112, 128])
N_BANDS_SHORT = len(SFB_OFFSETS_SHORT) - 1

# --------------------------------------------------------------------- windows
_long_sine = np.sin(np.pi * (np.arange(N_WIN) + 0.5) / N_WIN)
SHORT_WINDOW = np.sin(np.pi * (np.arange(N_SHORT) + 0.5) / N_SHORT)
_start = np.concatenate([_long_sine[:M], np.ones(SHORT_OFFSET), SHORT_WINDOW[M_S:],
                         np.zeros(SHORT_OFFSET)])
WINDOWS = {ONLY_LONG: _long_sine, LONG_START: _start, LONG_STOP: _start[::-1].copy()}

_n = np.arange(N_SHORT)[None, :]
_k = np.arange(M_S)[:, None]
_basis_short = np.sqrt(2.0 / M_S) * np.cos(np.pi / M_S * (_n + 0.5 + M_S / 2) * (_k + 0.5))
_basis_long = _mdct._basis


def short_window_span(w):
    """Sample range [a, b) of short window w inside its frame."""
    a = SHORT_OFFSET + M_S * w
    return a, a + N_SHORT


def noise_shape(seq):
    """Power envelope w^2(n) (length 2048) of the noise a frame of this type spreads in time.

    For a short frame it is the sum of the eight short windows squared
    (the noise of each window is confined to that window).
    """
    if seq != EIGHT_SHORT:
        return WINDOWS[seq] ** 2
    out = np.zeros(N_WIN)
    for w in range(N_SUBWIN):
        a, b = short_window_span(w)
        out[a:b] += SHORT_WINDOW ** 2
    return out


# --------------------------------------------------------------------- sequences
def legal(prev, cur):
    if prev in (ONLY_LONG, LONG_STOP):
        return cur in (ONLY_LONG, LONG_START)
    return cur in (EIGHT_SHORT, LONG_STOP)


def check_sequence(seq):
    seq = list(seq)
    for f in range(1, len(seq)):
        if not legal(seq[f - 1], seq[f]):
            raise ValueError(f"illegal window transition {SEQ_NAMES.get(seq[f-1])} -> "
                             f"{SEQ_NAMES.get(seq[f])} at frame {f}")
    if seq and seq[0] not in (ONLY_LONG, LONG_START, EIGHT_SHORT, LONG_STOP):
        raise ValueError("bad window sequence")


def sequence_from_short_frames(short):
    """Window sequence for a boolean 'needs short windows' per frame."""
    s = np.array(short, dtype=bool).copy()
    F = len(s)
    for f in range(1, F - 1):            # no stop+start window: fill single long gaps
        if s[f - 1] and s[f + 1] and not s[f]:
            s[f] = True
    seq = np.full(F, ONLY_LONG, dtype=int)
    for f in range(F):
        if s[f]:
            seq[f] = EIGHT_SHORT
        elif f + 1 < F and s[f + 1]:
            seq[f] = LONG_START
        elif f > 0 and s[f - 1]:
            seq[f] = LONG_STOP
    check_sequence(seq)
    return seq


# --------------------------------------------------------------------- transient detector
ATTACK_BLOCK = 128          # 2.7 ms analysis blocks
ATTACK_HISTORY = 8          # compared with the mean of the previous 8 blocks (21 ms)
ATTACK_RATIO_DB = 6.0       # chosen on the dev excerpts by scripts/tune_block_switching.py, then frozen
ATTACK_FLOOR_DBFS = -60.0


def detect_attacks(x, ratio_db=None):
    """Original-signal sample positions where an attack block starts.

    Energy of the high-passed signal (first difference) in 128-sample blocks;
    a block is an attack if it is ratio_db above the mean of the previous 8
    blocks and above ATTACK_FLOOR_DBFS (so noise in near-silence is ignored).
    """
    ratio_db = ATTACK_RATIO_DB if ratio_db is None else ratio_db
    x = np.asarray(x, dtype=np.float64)
    hp = np.diff(x, prepend=0.0)
    nb = len(hp) // ATTACK_BLOCK
    e = np.mean(hp[: nb * ATTACK_BLOCK].reshape(nb, ATTACK_BLOCK) ** 2, axis=1)
    c = np.concatenate([[0.0], np.cumsum(e)])
    j = np.arange(1, nb)
    lo = np.maximum(j - ATTACK_HISTORY, 0)
    prev = (c[j] - c[lo]) / (j - lo)
    hit = (e[j] > prev * 10 ** (ratio_db / 10)) & (e[j] > 10 ** (ATTACK_FLOOR_DBFS / 10))
    return j[hit] * ATTACK_BLOCK


def short_frames_for(x, ratio_db=None):
    F = _mdct.n_frames(len(x))
    short = np.zeros(F, dtype=bool)
    for t in detect_attacks(x, ratio_db):
        f = (t + SHORT_OFFSET) // M
        if 0 <= f < F:
            short[f] = True
    return short


def choose_sequence(x, ratio_db=None):
    return sequence_from_short_frames(short_frames_for(x, ratio_db))


# --------------------------------------------------------------------- transform
def analysis(x, seq):
    """MDCT of every frame with its window; coefficients in natural order, shape (F, M).

    Short frames hold their 8 windows one after another (window-major, 8 x 128).
    """
    fr = _mdct.frames(np.asarray(x, dtype=np.float64))
    X = np.empty((fr.shape[0], M))
    for f, s in enumerate(seq):
        if s == EIGHT_SHORT:
            segs = np.stack([fr[f, slice(*short_window_span(w))] for w in range(N_SUBWIN)])
            X[f] = ((segs * SHORT_WINDOW) @ _basis_short.T).reshape(-1)
        else:
            X[f] = (fr[f] * WINDOWS[s]) @ _basis_long.T
    return X


def synthesis(X, seq, length):
    """Inverse of analysis(): per-frame inverse MDCT, window, overlap-add, trim."""
    F = X.shape[0]
    out = np.zeros((F + 1) * M)
    for f, s in enumerate(seq):
        y = np.zeros(N_WIN)
        if s == EIGHT_SHORT:
            ys = (X[f].reshape(N_SUBWIN, M_S) @ _basis_short) * SHORT_WINDOW
            for w in range(N_SUBWIN):
                a, b = short_window_span(w)
                y[a:b] += ys[w]
        else:
            y = (X[f] @ _basis_long) * WINDOWS[s]
        out[f * M: f * M + N_WIN] += y
    return out[M: M + length]


# --------------------------------------------------------------------- coding layouts
class Layout:
    """How a frame's 1024 coefficients are split into coding bands.

    offsets   band boundaries in *coded* order (length n_bands + 1)
    sections  (first_band, n_bands, max_band_bits): each section sends its own
              max_band field (one section for a long frame, one per group for
              a short frame)
    perm      coded position -> natural index (None = identity, long frames)
    """

    def __init__(self, offsets, sections, perm=None):
        self.offsets = np.asarray(offsets)
        self.width = np.diff(self.offsets)
        self.n_bands = len(self.width)
        self.band_of_line = np.repeat(np.arange(self.n_bands), self.width)
        self.sections = tuple(sections)
        self.perm = perm

    def to_coded(self, X_nat):
        return X_nat if self.perm is None else X_nat[self.perm]

    def to_natural(self, X_coded):
        if self.perm is None:
            return X_coded
        out = np.empty_like(X_coded)
        out[self.perm] = X_coded
        return out


LONG_LAYOUT = Layout(SFB_OFFSETS, [(0, len(SFB_OFFSETS) - 1, 6)])
SHORT_MAX_BAND_BITS = 4
GROUPING_BITS = N_SUBWIN - 1


@lru_cache(maxsize=None)
def short_layout(groups):
    """Layout of a short frame for a grouping such as (3, 1, 4) (window counts per group)."""
    groups = tuple(int(g) for g in groups)
    if sum(groups) != N_SUBWIN or min(groups) < 1:
        raise ValueError(f"bad grouping {groups}")
    perm, offsets, sections = [], [0], []
    w0 = 0
    for gi, n in enumerate(groups):
        sections.append((gi * N_BANDS_SHORT, N_BANDS_SHORT, SHORT_MAX_BAND_BITS))
        for b in range(N_BANDS_SHORT):
            lo, hi = SFB_OFFSETS_SHORT[b], SFB_OFFSETS_SHORT[b + 1]
            for w in range(w0, w0 + n):
                perm.extend(w * M_S + np.arange(lo, hi))
            offsets.append(offsets[-1] + n * (hi - lo))
        w0 += n
    return Layout(np.array(offsets), sections, np.array(perm))


def grouping_to_bits(groups):
    """7 bits; bit i (MSB first) = 1 if window i+1 continues the group of window i."""
    v = 0
    for n in groups:
        for j in range(n):
            v = (v << 1) | (1 if j > 0 else 0)
    return v & ((1 << GROUPING_BITS) - 1)   # drop window 0's bit (always a new group)


def bits_to_grouping(v):
    groups = [1]
    for i in range(GROUPING_BITS - 1, -1, -1):
        if (v >> i) & 1:
            groups[-1] += 1
        else:
            groups.append(1)
    return tuple(groups)


GROUP_DB = 6.0   # a window starts a new group when its energy differs this much from the group's first


def choose_grouping(X_nat):
    """Group consecutive short windows of similar energy (encoder decision)."""
    e = 10 * np.log10(np.sum(X_nat.reshape(N_SUBWIN, M_S) ** 2, axis=1) + 1e-12)
    groups, ref = [1], e[0]
    for w in range(1, N_SUBWIN):
        if abs(e[w] - ref) > GROUP_DB:
            groups.append(1)
            ref = e[w]
        else:
            groups[-1] += 1
    return tuple(groups)
