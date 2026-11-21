"""Short-time (2.67 ms) energies in 10 frequency groups, shared by the PAMs.

256-point Hann FFTs every 128 samples of the padded signal; sub-block t is
centred on padded sample 128 t + 64. Inside MDCT frame f (padded samples
[fM, fM + 2M)) the 16 sub-blocks are t = 8f .. 8f + 15, and the short window
w of a block-switched frame (padded fM + 448 + 128 w, 256 samples) has
exactly the span of sub-block t = 8f + 4 + w.
"""
import numpy as np

from . import M, N_WIN
from .bands import FFT_FREQS

SUB = 128                       # sub-block hop (2.67 ms)
SUB_PER_FRAME = N_WIN // SUB    # 16 sub-blocks span one MDCT window
SUB_PER_HOP = M // SUB          # 8
SFFT = 256
GROUP_EDGES_HZ = np.array([0, 400, 800, 1600, 3200, 4800, 6400, 9600, 12800, 16000, 24001])
N_GROUPS = len(GROUP_EDGES_HZ) - 1

_sfreqs = np.arange(SFFT // 2 + 1) * 48000 / SFFT
_SGROUP = np.clip(np.searchsorted(GROUP_EDGES_HZ, _sfreqs, side="right") - 1, 0, N_GROUPS - 1)
GROUP_OF_BIN = np.clip(np.searchsorted(GROUP_EDGES_HZ, FFT_FREQS, side="right") - 1, 0, N_GROUPS - 1)
_SHANN = np.hanning(SFFT)

# Weight of each sub-block in the long (2048-point Hann) FFT that Model 1 uses
H2 = (np.hanning(N_WIN) ** 2).reshape(SUB_PER_FRAME, SUB).mean(axis=1)
H2 /= H2.sum()


def subblock_energies(xp):
    """Group energies of 2.67 ms sub-blocks of the padded signal, shape (T, groups)."""
    T = len(xp) // SUB
    pad = np.concatenate([np.zeros(SFFT), xp, np.zeros(SFFT)])
    centres = np.arange(T) * SUB + SUB // 2 + SFFT
    idx = centres[:, None] + np.arange(-SFFT // 2, SFFT // 2)[None, :]
    P = np.abs(np.fft.rfft(pad[idx] * _SHANN, axis=1)) ** 2
    E = np.zeros((T, N_GROUPS))
    for g in range(N_GROUPS):
        E[:, g] = P[:, _SGROUP == g].sum(axis=1)
    return E + 1e-30


def frame_slice(f):
    return slice(f * SUB_PER_HOP, f * SUB_PER_HOP + SUB_PER_FRAME)
