"""Improved PAM: baseline (Model 1) + temporal masking with pre-echo control.

Problem with the baseline
  Model 1 is a purely simultaneous model computed from one 2048-sample
  (42.7 ms) Hann-windowed spectrum. It therefore sees the *average* energy of
  the window. But the quantisation noise of an MDCT frame is not confined to
  where the signal is loud: after the inverse MDCT it is spread over the whole
  synthesis window with power envelope w^2(n) (sine window squared). If an
  attack falls inside the window, the noise also fills the quiet part before
  the attack, where only the quiet signal and a few ms of pre-masking can
  hide it (pre-echo). Conversely, after a loud event the ear's post-masking
  (~100-200 ms) hides noise that a simultaneous model forbids.

Model (applied to the masking part of the global threshold; ATH unchanged)
  1. Short-time energies: 256-point Hann FFTs every 128 samples (2.67 ms
     "sub-blocks"), summed into 10 frequency groups
     (0-0.4-0.8-1.6-3.2-4.8-6.4-9.6-12.8-16-24 kHz).
  2. Temporal masking envelope per group: each sub-block's energy also masks
       * later sub-blocks (post-masking), attenuated by
         A_post(t) = 10 dB * min(t, 5 ms)/5 ms + 32 dB * log10(max(t, 5 ms)/5 ms)
         (about -10 dB at 5 ms, -29 dB at 20 ms, -42 dB at 50 ms, -61 dB at 200 ms,
          following Zwicker & Fastl's post-masking decay), up to 200 ms;
       * earlier sub-blocks (pre-masking), attenuated by 8 dB per ms
         (effective only for the ~2-3 ms before an attack).
     e_mask[t] = max over these contributions (including e[t] itself).
  3. For MDCT frame f, with sub-blocks s = 0..15 inside its window:
       e_bar  = window-weighted mean energy the long-FFT baseline effectively saw
                (Hann^2 weights),
       w2[s]  = share of the frame's quantisation noise falling in sub-block s
                (mean of the sine window squared over s; 1 at the centre),
     the noise of the frame stays below the local masking everywhere if
       F = min_s  e_mask[s] / (e_bar * w2[s]).
     For a stationary signal F = 1 (no change). Before an attack F << 1
     (thresholds tightened by the quiet part); in a decay after a loud event
     F > 1 (post-masking relaxes them). F is limited to [-35 dB, +15 dB].
  4. Improved global threshold per FFT bin:  T = ATH + F_group * Mask_Model1,
     then the same band-allowance mapping as the baseline.

Block-switching mode (optional v2 codec)
  * Long, start and stop frames: as above, with w2[s] taken from that frame's
    own window (start/stop windows put no noise in their zero parts).
  * Short frames: each 5.3 ms window is masked by the temporal masking
    envelope e_mask at its own sub-block instead of by its own energy only
    (the baseline's short-window rule, see pam_baseline.py), i.e. the
    post-masking of an attack also covers the windows that follow it.
"""
import numpy as np

from . import M, N_WIN
from . import mdct as _mdct
from .pam_baseline import ATH_LIN, BaselinePAM, allowance_per_line, global_threshold_db
from .subblocks import (GROUP_EDGES_HZ, GROUP_OF_BIN, H2 as _H2, N_GROUPS, SUB,  # noqa: F401
                        SUB_PER_FRAME, frame_slice, subblock_energies)

POST_MAX_MS = 200.0
PRE_DB_PER_MS = 8.0
PRE_MAX_MS = 8.0
F_MIN_DB, F_MAX_DB = -35.0, 15.0

_SUB_MS = 1000.0 * SUB / 48000


def noise_share(window_sq):
    """Mean of a frame's noise envelope w^2(n) in each of its 16 sub-blocks."""
    return np.asarray(window_sq).reshape(SUB_PER_FRAME, SUB).mean(axis=1)


_n = np.arange(N_WIN)
_W2 = noise_share(np.sin(np.pi * (_n + 0.5) / N_WIN) ** 2)   # sine window (v1 / ONLY_LONG)


def post_attenuation_db(t_ms):
    t = np.asarray(t_ms, dtype=float)
    return 10.0 * np.minimum(t, 5.0) / 5.0 + 32.0 * np.log10(np.maximum(t, 5.0) / 5.0)


def masking_envelope(E):
    """Temporal spread (max over pre- and post-masking contributions)."""
    T = E.shape[0]
    out = E.copy()
    n_post = int(POST_MAX_MS / _SUB_MS)
    for d in range(1, n_post + 1):
        a = 10 ** (-post_attenuation_db(d * _SUB_MS) / 10)
        out[d:] = np.maximum(out[d:], E[:-d] * a)
    n_pre = int(PRE_MAX_MS / _SUB_MS)
    for d in range(1, n_pre + 1):
        a = 10 ** (-PRE_DB_PER_MS * d * _SUB_MS / 10)
        out[:-d] = np.maximum(out[:-d], E[d:] * a)
    return out


def temporal_factors(x, seq=None):
    """Per-frame, per-group threshold factor F (linear), shape (frames, groups).

    seq (block-switching mode) gives each frame's window type, whose noise
    envelope replaces the sine window's; without it every frame is ONLY_LONG.
    """
    x = np.asarray(x, dtype=np.float64)
    xp = _mdct.pad(x)
    F_frames = _mdct.n_frames(len(x))
    E = subblock_energies(xp)
    Em = masking_envelope(E)
    fac = np.empty((F_frames, N_GROUPS))
    if seq is not None:
        from .blockswitch import noise_shape
        shares = {s: noise_share(noise_shape(s)) for s in set(int(v) for v in seq)}
    for f in range(F_frames):
        sl = frame_slice(f)
        e_bar = _H2 @ E[sl]                                  # (groups,)
        w2 = _W2 if seq is None else shares[int(seq[f])]
        with np.errstate(divide="ignore"):
            ratio = Em[sl] / (e_bar[None, :] * w2[:, None])  # (16, groups); inf where no noise
        fac[f] = ratio.min(axis=0)
    return np.clip(fac, 10 ** (F_MIN_DB / 10), 10 ** (F_MAX_DB / 10))


class TemporalPAM(BaselinePAM):
    name = "improved"

    def finalize(self, x, out, seq=None):
        fac = temporal_factors(x, seq)
        out["temporal_factor"] = fac
        out["Tg_db_baseline"] = global_threshold_db(out["mask_lin"])
        mask_imp = out["mask_lin"] * fac[:, GROUP_OF_BIN]
        out["mask_lin_improved"] = mask_imp
        out["Tg_db"] = 10 * np.log10(ATH_LIN + mask_imp)
        out["allow"] = np.stack([allowance_per_line(t) for t in out["Tg_db"]])
        if seq is not None:
            self.short_allowances(x, out, seq)
        return out

    def short_masker_energy(self, E):
        """Short windows: each window is masked by the temporal masking envelope
        (its own energy plus pre/post-masking from its neighbours)."""
        return masking_envelope(E)
