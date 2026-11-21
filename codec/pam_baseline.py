"""Baseline psychoacoustic model: MPEG-1 Psychoacoustic Model 1 (as taught in
the course slides / Painter & Spanias), adapted to the codec's 2048-sample frame.

Per MDCT frame (the same 2048 samples the MDCT sees):
  1. Spectral analysis + SPL normalisation. Hann window, 2048-point FFT.
     Calibration: a full-scale sinusoid (0 dBFS) gives 90.3 dB SPL, i.e. a
     +/-1 LSB 16-bit sine sits at ~0 dB SPL, where the threshold in quiet is
     lowest. Bin powers are normalised so that, for noise, the sum of the
     one-sided bin powers equals the signal's mean-square power.
  2. Tonal maskers: local maxima at least 7 dB above the bins in a
     frequency-dependent neighbourhood; power = sum of the peak bin and its
     two neighbours. Noise maskers: remaining power summed in each critical
     band and placed at the band's geometric-mean bin.
  3. Decimation: maskers below the threshold in quiet are removed; of any two
     maskers closer than 0.5 Bark only the stronger survives.
  4. Individual masking thresholds with the level-dependent two-slope
     spreading function and the tonal/noise masking indices
        T_TM = P - 0.275 z - 6.025 + SF,   T_NM = P - 0.175 z - 2.025 + SF.
  5. Global masking threshold = power sum of the threshold in quiet and all
     individual thresholds, evaluated at every FFT bin.

Mapping to band allowances (the part that connects the model to the coder):
  A threshold of T dB SPL at an FFT bin is the power a component in that bin
  may have and still be masked. Converted to full-scale power
  p = 0.5 * 10^((T - 90.3)/10). Quantisation noise of variance s^2 on an
  orthonormal MDCT coefficient produces white time-domain noise of variance
  s^2 whose power per FFT bin is s^2 / M. A uniform quantiser spreads its
  noise evenly over a band, so every line in the band must stay below the
  *lowest* threshold in that band (Model 1's "minimum threshold in the
  subband" rule):
        a_b = M * min_{bins in band b} p          (allowed noise per MDCT line)
        A_b = width_b * a_b                       (allowed noise energy of band b)

Block-switching mode (optional v2): a short frame needs a threshold for each
of its eight 5.3 ms windows, not one for the whole 42.7 ms frame. The frame's
Model-1 masking curve (from the long FFT) keeps its frequency shape, and its
level follows each window's own energy in 10 frequency groups:
    mask_w(k) = mask(k) * E_w,g / E_bar,g
with E_w,g the energy of the 256-sample sub-block that coincides with short
window w and E_bar,g the energy the long Hann FFT saw (same weighting as the
improved PAM). So a quiet window before an attack is masked only by its own
quiet signal. This is simultaneous masking at short-window resolution; the
improved PAM replaces E_w,g by its temporal masking envelope (pre/post-masking).
A noise variance per coefficient is transform-independent (white noise of
variance s^2 gives coefficients of variance s^2 in any orthonormal transform),
so the same per-line conversion gives the allowance of a short band.
"""
import numpy as np

from . import M, N_WIN
from .bands import (BAND_HI_HZ, BAND_LO_HZ, CRITICAL_BAND_EDGES_HZ, FFT_FREQS,
                    N_BANDS, BAND_WIDTH, ath_db_spl, bark)
from . import mdct as _mdct
from .subblocks import GROUP_OF_BIN, H2, SUB_PER_HOP, frame_slice, subblock_energies

PN_DB = 90.302              # dB SPL of a full-scale sinusoid
EPS_POW = 1e-20
N_BINS = N_WIN // 2 + 1
K_MAX = int(np.searchsorted(FFT_FREQS, 20000.0))    # analyse maskers up to 20 kHz
Z_BIN = bark(FFT_FREQS)
ATH_DB = ath_db_spl(FFT_FREQS)
ATH_LIN = 10 ** (ATH_DB / 10)
_HANN = np.hanning(N_WIN)
_SUM_W2 = np.sum(_HANN ** 2)

# Tonal neighbourhood: ISO 11172-3 Layer II table for a 1024-point FFT at 48 kHz,
# scaled x2 for our 2048-point FFT (same Hz widths). Peak must exceed P(k+-j)+7 dB
# for all j in 2..DELTA_MAX(k).
DELTA_MAX = np.where(FFT_FREQS < 3000, 4,
            np.where(FFT_FREQS < 6000, 6,
            np.where(FFT_FREQS < 12000, 12, 24)))

# FFT bins belonging to each coder band (same 23.4 Hz grid as the MDCT lines)
BAND_BINS = [np.flatnonzero((FFT_FREQS >= lo) & (FFT_FREQS < hi))
             for lo, hi in zip(BAND_LO_HZ, BAND_HI_HZ)]
CB_BINS = [np.flatnonzero((FFT_FREQS >= lo) & (FFT_FREQS < hi) & (np.arange(N_BINS) >= 1))
           for lo, hi in zip(CRITICAL_BAND_EDGES_HZ[:-1], CRITICAL_BAND_EDGES_HZ[1:])]


def spectrum_db(seg):
    """Calibrated power spectrum P(k) in dB SPL for one 2048-sample segment."""
    X = np.fft.rfft(_HANN * seg)
    p = 2.0 * np.abs(X) ** 2 / (N_WIN * _SUM_W2)
    p[0] *= 0.5
    p[-1] *= 0.5
    return 10 * np.log10(np.maximum(p, EPS_POW) / 0.5) + PN_DB


def find_maskers(P):
    """Return (tonal_bins, tonal_db, noise_bins, noise_db) before decimation."""
    k = np.arange(2, K_MAX)
    is_peak = (P[k] > P[k - 1]) & (P[k] >= P[k + 1])
    for j in range(2, int(DELTA_MAX.max()) + 1):
        need = DELTA_MAX[k] >= j
        lo = np.clip(k - j, 0, N_BINS - 1)
        hi = np.clip(k + j, 0, N_BINS - 1)
        ok = (P[k] - P[lo] >= 7.0) & (P[k] - P[hi] >= 7.0)
        is_peak &= ~need | ok
    tk = k[is_peak]
    lin = 10 ** (P / 10)
    t_db = 10 * np.log10(lin[tk - 1] + lin[tk] + lin[tk + 1])

    # Noise maskers: everything outside the tonal neighbourhoods, per critical band
    excluded = np.zeros(N_BINS, dtype=bool)
    for kk in tk:
        d = DELTA_MAX[kk]
        excluded[max(kk - d, 0): kk + d + 1] = True
    nk, n_db = [], []
    for bins in CB_BINS:
        sel = bins[~excluded[bins]]
        if len(sel) == 0:
            continue
        pw = lin[sel].sum()
        loc = int(round(np.exp(np.mean(np.log(bins)))))  # geometric-mean bin of the band
        nk.append(loc)
        n_db.append(10 * np.log10(pw))
    return tk, t_db, np.array(nk, dtype=int), np.array(n_db)


def decimate(tk, t_db, nk, n_db):
    """Remove maskers below the threshold in quiet, then the weaker of any pair < 0.5 Bark apart."""
    bins = np.concatenate([tk, nk]).astype(int)
    pw = np.concatenate([t_db, n_db])
    tonal = np.concatenate([np.ones(len(tk), bool), np.zeros(len(nk), bool)])
    keep = pw >= ATH_DB[bins]
    bins, pw, tonal = bins[keep], pw[keep], tonal[keep]
    order = np.argsort(Z_BIN[bins], kind="stable")
    bins, pw, tonal = list(bins[order]), list(pw[order]), list(tonal[order])
    i = 0
    while i < len(bins) - 1:
        if Z_BIN[bins[i + 1]] - Z_BIN[bins[i]] < 0.5:
            drop = i if pw[i] < pw[i + 1] else i + 1
            del bins[drop], pw[drop], tonal[drop]
        else:
            i += 1
    bins = np.array(bins, dtype=int)
    pw = np.array(pw)
    tonal = np.array(tonal, dtype=bool)
    return bins[tonal], pw[tonal], bins[~tonal], pw[~tonal]


def spreading_db(dz, P):
    """Level-dependent two-slope spreading function SF(i, j) in dB (Model 1)."""
    sf = np.full(dz.shape, -np.inf)
    m = (dz >= -3) & (dz < -1)
    sf[m] = (17 * dz - 0.4 * P + 11)[m]
    m = (dz >= -1) & (dz < 0)
    sf[m] = ((0.4 * P + 6) * dz)[m]
    m = (dz >= 0) & (dz < 1)
    sf[m] = (-17 * dz)[m]
    m = (dz >= 1) & (dz < 8)
    sf[m] = ((0.15 * P - 17) * dz - 0.15 * P)[m]
    return sf


def individual_thresholds_lin(bins, pw, tonal):
    """Sum over maskers of 10^(T/10) at every FFT bin (linear 'SPL power')."""
    if len(bins) == 0:
        return np.zeros(N_BINS)
    zj = Z_BIN[bins][:, None]
    Pj = pw[:, None]
    dz = Z_BIN[None, :] - zj
    idx = (-0.275 * zj - 6.025) if tonal else (-0.175 * zj - 2.025)
    T = Pj + idx + spreading_db(dz, Pj)
    return np.sum(10 ** (T / 10), axis=0)


def analyze_segment(seg):
    """Full Model-1 analysis of one 2048-sample segment."""
    P = spectrum_db(seg)
    tk, t_db, nk, n_db = find_maskers(P)
    tk, t_db, nk, n_db = decimate(tk, t_db, nk, n_db)
    mask_lin = individual_thresholds_lin(tk, t_db, True) + individual_thresholds_lin(nk, n_db, False)
    return dict(P_db=P, tonal_bins=tk, tonal_db=t_db, noise_bins=nk, noise_db=n_db,
                mask_lin=mask_lin)


def global_threshold_db(mask_lin):
    return 10 * np.log10(ATH_LIN + mask_lin)


def critical_bandwidth_hz(f):
    """Zwicker critical bandwidth BW_c(f) = 25 + 75 (1 + 1.4 (f/kHz)^2)^0.69  (Hz)."""
    return 25.0 + 75.0 * (1.0 + 1.4 * (np.asarray(f) / 1000.0) ** 2) ** 0.69


# Number of FFT bins (= MDCT lines) in one critical band around each bin (>= 1)
LINES_PER_CB = np.maximum(1.0, critical_bandwidth_hz(FFT_FREQS) / (FFT_FREQS[1] - FFT_FREQS[0]))


def allowance_per_line(Tg_db):
    """Global threshold (dB SPL per FFT bin) -> allowed noise variance per MDCT line, per band.

    Model 1's threshold at bin i is the level at which a probe at that
    frequency is just masked. Quantisation noise is not a single line: it
    fills every line of the band, and its audibility is governed by the total
    noise power falling inside one critical band. The threshold is therefore
    spread over the critical bandwidth (divided by the number of lines per
    critical band) so that a critical band full of noise integrates to the
    threshold, and the band takes the minimum over its lines.
    """
    p = 0.5 * 10 ** ((Tg_db - PN_DB) / 10) / LINES_PER_CB
    return np.array([M * p[b].min() for b in BAND_BINS])


# FFT bins (23.4 Hz grid) inside each short-window band (187.5 Hz lines)
def _short_band_bins():
    from .blockswitch import SFB_OFFSETS_SHORT, M_S
    hz = SFB_OFFSETS_SHORT * (FFT_FREQS[-1] / M_S)
    return [np.flatnonzero((FFT_FREQS >= lo) & (FFT_FREQS < hi)) for lo, hi in zip(hz[:-1], hz[1:])]


SHORT_BAND_BINS = _short_band_bins()
SHORT_RATIO_MIN_DB, SHORT_RATIO_MAX_DB = -60.0, 15.0


def allowance_per_line_short(Tg_db):
    """Same as allowance_per_line, for the 14 short-window bands."""
    p = 0.5 * 10 ** ((Tg_db - PN_DB) / 10) / LINES_PER_CB
    return np.array([M * p[b].min() for b in SHORT_BAND_BINS])


class BaselinePAM:
    name = "baseline"

    def analyze(self, x, seq=None):
        """Run the model on every MDCT frame of signal x.

        seq: window sequence (block-switching mode); adds out["allow_short"],
        a dict frame -> (8, 14) allowances for every EIGHT_SHORT frame.

        Returns dict with per-frame arrays:
          P_db (F, 1025), mask_lin (F, 1025), Tg_db (F, 1025),
          allow (F, 49)  allowed noise variance per MDCT line in each band,
          maskers  list of per-frame masker tuples (for plotting).
        """
        segs = _mdct.frames(x)
        F = segs.shape[0]
        out = dict(P_db=np.empty((F, N_BINS)), mask_lin=np.empty((F, N_BINS)), maskers=[])
        for f in range(F):
            a = analyze_segment(segs[f])
            out["P_db"][f] = a["P_db"]
            out["mask_lin"][f] = a["mask_lin"]
            out["maskers"].append((a["tonal_bins"], a["tonal_db"], a["noise_bins"], a["noise_db"]))
        self.finalize(x, out, seq)
        return out

    def finalize(self, x, out, seq=None):
        out["Tg_db"] = global_threshold_db(out["mask_lin"])
        out["allow"] = np.stack([allowance_per_line(t) for t in out["Tg_db"]])
        if seq is not None:
            self.short_allowances(x, out, seq)
        return out

    def short_masker_energy(self, E):
        """Energy that masks each short window: its own (simultaneous masking only)."""
        return E

    def short_allowances(self, x, out, seq):
        from .blockswitch import EIGHT_SHORT, N_SUBWIN
        E = subblock_energies(_mdct.pad(np.asarray(x, dtype=np.float64)))
        Em = self.short_masker_energy(E)
        lo, hi = 10 ** (SHORT_RATIO_MIN_DB / 10), 10 ** (SHORT_RATIO_MAX_DB / 10)
        out["allow_short"], out["short_ratio"] = {}, {}
        for f in np.flatnonzero(np.asarray(seq) == EIGHT_SHORT):
            e_bar = H2 @ E[frame_slice(f)]
            t = f * SUB_PER_HOP + 4 + np.arange(N_SUBWIN)       # sub-block of each short window
            ratio = np.clip(Em[t] / e_bar[None, :], lo, hi)     # (8, groups)
            Tg = 10 * np.log10(ATH_LIN[None, :] + out["mask_lin"][f][None, :] * ratio[:, GROUP_OF_BIN])
            out["allow_short"][int(f)] = np.stack([allowance_per_line_short(tw) for tw in Tg])
            out["short_ratio"][int(f)] = ratio
        return out


assert len(BAND_BINS) == N_BANDS
