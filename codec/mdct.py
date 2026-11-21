"""Orthonormal MDCT / IMDCT with sine window, 2048-sample frames, 1024 hop.

Forward (frame f, k = 0..M-1, n = 0..2M-1):
    X_f[k] = sqrt(2/M) * sum_n w[n] x[fM + n] cos(pi/M (n + 1/2 + M/2)(k + 1/2))
Inverse + overlap-add:
    y[fM + n] += sqrt(2/M) * w[n] * sum_k X_f[k] cos(...)

The sine window w[n] = sin(pi (n + 1/2) / 2M) satisfies the Princen-Bradley
condition w[n]^2 + w[n+M]^2 = 1, so time-domain aliasing cancels between
neighbouring frames and reconstruction is perfect wherever a sample is
covered by two frames.

Boundaries / arbitrary lengths: the signal is padded with M zeros in front
and with zeros at the end up to a whole number of hops plus another M, so
every real sample lies in the fully-overlapped region:
    n_frames = ceil(L / M) + 1,   padded length = (n_frames + 1) * M.
The decoder removes the M leading samples and truncates to the original
length L that is stored in the bitstream header.
"""
import numpy as np

from . import M, N_WIN

_window = np.sin(np.pi * (np.arange(N_WIN) + 0.5) / N_WIN)
_n = np.arange(N_WIN)[None, :]
_k = np.arange(M)[:, None]
_basis = np.sqrt(2.0 / M) * np.cos(np.pi / M * (_n + 0.5 + M / 2) * (_k + 0.5))  # (M, 2M)


def window():
    return _window.copy()


def n_frames(length):
    return -(-length // M) + 1


def pad(x):
    """Zero-pad a 1-D signal for framing. Returns padded signal."""
    L = len(x)
    F = n_frames(L)
    total = (F + 1) * M
    out = np.zeros(total)
    out[M : M + L] = x
    return out


def frames(x):
    """Padded, *unwindowed* frames, shape (F, 2M). Frame f starts at f*M of the padded signal."""
    xp = pad(np.asarray(x, dtype=np.float64))
    F = n_frames(len(x))
    idx = np.arange(F)[:, None] * M + np.arange(N_WIN)[None, :]
    return xp[idx]


def mdct(x):
    """MDCT of a whole signal -> coefficients of shape (F, M)."""
    fr = frames(x) * _window[None, :]
    return fr @ _basis.T


def imdct(X, length):
    """Inverse MDCT with windowed overlap-add; returns `length` samples."""
    X = np.atleast_2d(X)
    F = X.shape[0]
    y_frames = (X @ _basis) * _window[None, :]  # (F, 2M)
    out = np.zeros((F + 1) * M)
    for f in range(F):
        out[f * M : f * M + N_WIN] += y_frames[f]
    return out[M : M + length]


def frame_time(f):
    """Centre time (s, relative to the original signal start) of MDCT frame f."""
    from . import FS
    # Frame f spans padded samples [fM, fM+2M) whose centre is padded index fM+M;
    # original sample 0 sits at padded index M, so the centre is original sample fM.
    return f * M / FS
