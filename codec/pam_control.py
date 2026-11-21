"""Control model for the experiment: "just give attack frames more bits".

Baseline PAM (Model 1) plus a crude, encoder-style transient detector:
  * the high-passed signal (first difference) is cut into 256-sample blocks
    (8 per MDCT window, like AAC's short blocks);
  * a frame is an "attack frame" if any block in its 2048-sample window has
    more than ATTACK_RATIO_DB more energy than the mean of the 4 blocks
    before it, and lies above -60 dBFS;
  * on attack frames the whole masking part of the threshold is lowered by
    BOOST_DB, uniformly across frequency.

Like the improved PAM, this lowers thresholds on frames with attacks, so the
same reservoir moves bits to them. Unlike the improved PAM it has no model of
where in the window the attack sits, how quiet the part before it is, which
frequencies need it, or of pre/post-masking. Comparing the two at the same
bitrate separates "more bits on attacks" from "the right bits, from a
temporal masking model".

BOOST_DB was chosen on the development excerpts only (castanets, speech):
boosts of 6, 12, 20, 30, 40 and 50 dB were tried at 64/96/128 kb/s, and the
rule fixed in advance was "lowest mean pre-echo, with mean NMR no worse than
the baseline". 30 and 40 dB tied on pre-echo (4.62 dB); 30 dB had the lower
NMR, so 30 dB was frozen before the held-out excerpt was encoded.
"""
import numpy as np

from . import M, N_WIN
from . import mdct as _mdct
from .pam_baseline import ATH_LIN, BaselinePAM, allowance_per_line

BLOCK = 256
ATTACK_RATIO_DB = 10.0
FLOOR_DBFS = -60.0
BOOST_DB = 30.0


def attack_frames(x):
    xp = _mdct.pad(np.asarray(x, dtype=np.float64))
    hp = np.diff(xp, prepend=0.0)
    nb = len(hp) // BLOCK
    e = np.mean(hp[: nb * BLOCK].reshape(nb, BLOCK) ** 2, axis=1) + 1e-20
    prev = np.array([e[max(0, j - 4):j].mean() if j > 0 else np.inf for j in range(nb)])
    is_attack = (e > prev * 10 ** (ATTACK_RATIO_DB / 10)) & (e > 10 ** (FLOOR_DBFS / 10))
    F = _mdct.n_frames(len(x))
    per = N_WIN // BLOCK
    hop = M // BLOCK
    return np.array([is_attack[f * hop: f * hop + per].any() for f in range(F)])


class ControlPAM(BaselinePAM):
    name = "control"

    def finalize(self, x, out, seq=None):
        att = attack_frames(x)
        out["attack_frame"] = att
        scale = np.where(att, 10 ** (-BOOST_DB / 10), 1.0)[:, None]
        mask = out["mask_lin"] * scale
        out["Tg_db"] = 10 * np.log10(ATH_LIN + mask)
        out["allow"] = np.stack([allowance_per_line(t) for t in out["Tg_db"]])
        if seq is not None:
            self.short_allowances(x, out, seq)
        return out
