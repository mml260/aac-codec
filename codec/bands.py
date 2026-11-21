"""Frequency tables shared by encoder, PAMs and decoder."""
import numpy as np

from . import FS, M, N_WIN

# Coder (scalefactor) bands: AAC long-window scalefactor band offsets for 48 kHz
# (ISO/IEC 13818-7 / 14496-3, swb_offset_long_48). 49 bands over 1024 MDCT lines,
# 4 lines (94 Hz) wide at the bottom, 32 lines (750 Hz) at the top: roughly
# 1/3 to 1/2 critical band wide over most of the range.
SFB_OFFSETS = np.array([
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 108, 120,
    132, 144, 160, 176, 196, 216, 240, 264, 292, 320, 352, 384, 416, 448, 480, 512,
    544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 1024])
N_BANDS = len(SFB_OFFSETS) - 1
BAND_WIDTH = np.diff(SFB_OFFSETS)
BAND_OF_LINE = np.repeat(np.arange(N_BANDS), BAND_WIDTH)

# Frequencies
MDCT_FREQS = (np.arange(M) + 0.5) * FS / (2 * M)       # centre of MDCT line k
FFT_FREQS = np.arange(N_WIN // 2 + 1) * FS / N_WIN      # FFT bin k (same 23.4 Hz grid)
BAND_LO_HZ = SFB_OFFSETS[:-1] * FS / (2 * M)
BAND_HI_HZ = SFB_OFFSETS[1:] * FS / (2 * M)

# Critical bands (Table 1.1 of Thiagarajan & Spanias / Zwicker), capped at 20 kHz.
CRITICAL_BAND_EDGES_HZ = np.array([
    0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320,
    2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000])


def bark(f):
    """Zwicker/Terhardt critical-band rate (Bark)."""
    f = np.asarray(f, dtype=np.float64)
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def ath_db_spl(f):
    """Terhardt threshold in quiet (dB SPL). Frequencies below 20 Hz are clamped."""
    f = np.maximum(np.asarray(f, dtype=np.float64), 20.0) / 1000.0
    return 3.64 * f ** -0.8 - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2) + 1e-3 * f ** 4


def sum_bands(v, axis=-1):
    """Sum a per-MDCT-line quantity over the coder bands."""
    return np.add.reduceat(v, SFB_OFFSETS[:-1], axis=axis)
