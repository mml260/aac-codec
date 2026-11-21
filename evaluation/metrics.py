"""Objective measurements used to compare decoded files with the original.

None of these use either codec PAM, so neither model is graded by itself.

1. Neutral noise-to-mask ratio (NMR) with an independent Johnston-style
   masking model (Johnston 1988 / PEAQ-like measurement):
     * 512-point Hann FFT, hop 256 (5.3 ms): finer time resolution than the
       codec, so noise smeared before an attack is seen as unmasked;
     * energy in 25 critical bands, spread with the Schroeder spreading
       function, tonality from the spectral flatness measure,
       offset O_b = a (14.5 + b) + (1 - a) 5.5 dB, floored by the threshold
       in quiet (same 90.3 dB SPL full-scale calibration);
     * NMR_b = coding-error energy / masking threshold in band b.
   Reported: Total NMR (dB) = 10 log10(mean NMR over frames and bands) and
   the percentage of "disturbed" frames whose largest band NMR exceeds
   1.5 dB (PEAQ's relative-disturbed-frames criterion). Frames where the
   original is below -70 dBFS are skipped.
2. Pre-echo, measured on high-passed signals (first difference, +6 dB/oct),
   because pre-echo is heard as high-frequency noise that sustained low
   notes cannot mask. Attacks are detected in the high-passed original as
   1 ms energy jumps of more than 10 dB above the maximum of the previous
   20 ms, above -55 dBFS, at least 50 ms apart. For each attack, the
   high-passed coding-error energy in the window from 20 ms to 2 ms *before*
   it is compared with the high-passed original's energy there, floored at
   the 16-bit noise level (pre-echo-to-signal ratio, dB). Higher = more
   audible pre-echo; > 0 dB means the error is louder than the signal it
   precedes.
3. SNR and segmental SNR, for reference only (not perceptual).
"""
import numpy as np

FS = 48000
PN_DB = 90.302


def _bark(f):
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


def _ath(f):
    f = np.maximum(f, 20.0) / 1000.0
    return 3.64 * f ** -0.8 - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2) + 1e-3 * f ** 4


_CB_EDGES = np.array([0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
                      2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20000])
_NFFT, _HOP = 512, 256
_FREQS = np.arange(_NFFT // 2 + 1) * FS / _NFFT
_BAND_OF_BIN = np.searchsorted(_CB_EDGES, _FREQS, side="right") - 1
_VALID = (_BAND_OF_BIN >= 0) & (_BAND_OF_BIN < len(_CB_EDGES) - 1) & (_FREQS > 0)
_NB = len(_CB_EDGES) - 1
_Z = _bark(0.5 * (_CB_EDGES[:-1] + _CB_EDGES[1:]))
_dz = _Z[:, None] - _Z[None, :]  # maskee - masker
_SF = 15.81 + 7.5 * (_dz + 0.474) - 17.5 * np.sqrt(1 + (_dz + 0.474) ** 2)
_SPREAD = 10 ** (_SF / 10)
_WIN = np.hanning(_NFFT)
_SW2 = np.sum(_WIN ** 2)
# threshold in quiet per band (power, full-scale units): minimum over the band, times its bins
_ATH_BAND = np.array([
    0.5 * 10 ** ((_ath(_FREQS[_VALID & (_BAND_OF_BIN == b)]).min() - PN_DB) / 10)
    * np.sum(_VALID & (_BAND_OF_BIN == b)) for b in range(_NB)])


def _frames(x):
    n = 1 + max(0, (len(x) - _NFFT) // _HOP)
    idx = np.arange(n)[:, None] * _HOP + np.arange(_NFFT)[None, :]
    return x[idx]


def _band_power(fr):
    P = 2 * np.abs(np.fft.rfft(fr * _WIN, axis=1)) ** 2 / (_NFFT * _SW2)
    P[:, ~_VALID] = 0.0
    B = np.zeros((fr.shape[0], _NB))
    for b in range(_NB):
        B[:, b] = P[:, _BAND_OF_BIN == b].sum(axis=1)
    return P, B


def masking_threshold(x):
    fr = _frames(x)
    P, B = _band_power(fr)
    Pv = P[:, _VALID] + 1e-30
    sfm_db = 10 * np.log10(np.exp(np.mean(np.log(Pv), axis=1)) / np.mean(Pv, axis=1))
    alpha = np.minimum(sfm_db / -60.0, 1.0)
    C = B @ _SPREAD.T
    b_idx = np.arange(_NB)[None, :]
    O = alpha[:, None] * (14.5 + b_idx) + (1 - alpha[:, None]) * 5.5
    T = np.maximum(C * 10 ** (-O / 10), _ATH_BAND[None, :])
    level = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-30)
    return T, level


def nmr(ref, test):
    T, level = masking_threshold(ref)
    _, N = _band_power(_frames(test - ref))
    keep = level > -70.0
    r = N[keep] / T[keep]
    total = 10 * np.log10(np.mean(r))
    disturbed = 100.0 * np.mean(10 * np.log10(np.max(r, axis=1) + 1e-30) > 1.5)
    per_frame = np.full(len(level), np.nan)
    per_frame[keep] = 10 * np.log10(np.mean(r, axis=1))
    return dict(total_nmr_db=float(total), disturbed_frames_pct=float(disturbed),
                per_frame_nmr_db=per_frame)


def _hp(x):
    return np.diff(np.asarray(x, dtype=np.float64), prepend=0.0)


def detect_attacks(x, rise_db=10.0, floor_dbfs=-55.0, min_gap_ms=50.0):
    x = _hp(x)
    blk = FS // 1000  # 1 ms
    n = len(x) // blk
    e = 10 * np.log10(np.mean(x[: n * blk].reshape(n, blk) ** 2, axis=1) + 1e-20)
    attacks, last = [], -10 ** 9
    for t in range(20, n):
        if e[t] > floor_dbfs and e[t] - np.max(e[t - 20:t]) > rise_db and t - last >= min_gap_ms:
            attacks.append(t * blk)
            last = t
    return np.array(attacks, dtype=int)


def pre_echo(ref, test, attacks=None, pre_ms=20.0, guard_ms=2.0):
    if attacks is None:
        attacks = detect_attacks(ref)
    err = _hp(test) - _hp(ref)
    ref = _hp(ref)
    floor = (1.0 / 32768) ** 2 / 12  # 16-bit quantisation noise power per sample
    ratios, levels = [], []
    for a in attacks:
        s, e = a - int(pre_ms * FS / 1000), a - int(guard_ms * FS / 1000)
        if s < 0:
            continue
        pe = np.sum(err[s:e] ** 2)
        ps = max(np.sum(ref[s:e] ** 2), floor * (e - s))
        ratios.append(10 * np.log10(pe / ps + 1e-20))
        levels.append(10 * np.log10(pe / (e - s) + 1e-20))
    ratios = np.array(ratios)
    return dict(n_attacks=int(len(ratios)),
                mean_pre_echo_to_signal_db=float(np.mean(ratios)) if len(ratios) else float("nan"),
                median_pre_echo_to_signal_db=float(np.median(ratios)) if len(ratios) else float("nan"),
                mean_pre_echo_level_dbfs=float(np.mean(levels)) if len(levels) else float("nan"),
                per_attack_db=ratios)


def snr(ref, test):
    e = test - ref
    s = 10 * np.log10(np.sum(ref ** 2) / (np.sum(e ** 2) + 1e-30))
    blk = 1024
    n = len(ref) // blk
    r = ref[: n * blk].reshape(n, blk)
    d = e[: n * blk].reshape(n, blk)
    pr = np.sum(r ** 2, axis=1)
    keep = pr > blk * 10 ** (-60 / 10)
    seg = 10 * np.log10(pr[keep] / (np.sum(d[keep] ** 2, axis=1) + 1e-30))
    return dict(snr_db=float(s), seg_snr_db=float(np.mean(np.clip(seg, -10, 35))))


def all_metrics(ref, test):
    out = {}
    m = nmr(ref, test)
    out.update({k: v for k, v in m.items() if k != "per_frame_nmr_db"})
    p = pre_echo(ref, test)
    out.update({k: v for k, v in p.items() if k != "per_attack_db"})
    out.update(snr(ref, test))
    return out
