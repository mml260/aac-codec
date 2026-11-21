"""Make all report figures (PNG) into figures/.

Requires results/ from scripts/run_all.py.
Usage:  python scripts/make_figures.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec import FS, M, N_WIN, mdct  # noqa: E402
from codec import blockswitch as bs  # noqa: E402
from codec.bands import (BAND_HI_HZ, BAND_LO_HZ, BAND_WIDTH, FFT_FREQS, SFB_OFFSETS,  # noqa: E402
                         sum_bands)
from codec.encoder import encode  # noqa: E402
from codec.pam_baseline import ATH_DB, PN_DB, BaselinePAM  # noqa: E402
from codec import pam_temporal as pt  # noqa: E402
from codec.wavio import read_wav  # noqa: E402
from evaluation.metrics import detect_attacks  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
# Palette (validated categorical slots 1-2) + ink
C_BASE, C_IMP, C_CTL = "#2a78d6", "#eb6834", "#1baf7a"
# Each condition: (label, colour, marker, line style). Colour = ear model, style = transform.
STYLE = {
    "baseline": ("Baseline, long windows", C_BASE, "o", "--"),
    "improved": ("Improved, long windows", C_IMP, "o", "--"),
    "control": ("Control (attack boost), long windows", C_CTL, "s", ":"),
    "baseline_bs": ("Baseline + block switching", C_BASE, "D", "-"),
    "improved_bs": ("Improved + block switching", C_IMP, "D", "-"),
}
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 1.6, "figure.dpi": 150,
})


def load(name):
    x, _ = read_wav(ROOT / "data" / "prepared" / f"{name}.wav")
    return x[:, 0]


def decoded(name, model, rate):
    p = next((RES / name).glob(f"{name}__{model}__target{rate:03d}k__actual*.wav"))
    y, _ = read_wav(p)
    return y[:, 0], p.name


def per_line_db(var_per_line):
    """MDCT noise variance per line -> dB SPL per FFT bin (same scale as the spectrum)."""
    return 10 * np.log10(np.maximum(var_per_line / M, 1e-30) / 0.5) + PN_DB


def freq_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(40, 22000)
    ticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["50", "100", "200", "500", "1k", "2k", "5k", "10k", "20k"])
    ax.set_xlabel("Frequency (Hz)")


# ----------------------------------------------------------------------------- 1
def fig_mdct():
    x = load("castanets")[48000:48000 + 123457]  # starts mid-rhythm, so both edges carry signal
    y = mdct.imdct(mdct.mdct(x), len(x))
    err = np.abs(y - x) + 1e-18
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.0), gridspec_kw=dict(width_ratios=[2.2, 1, 1]))
    n = np.arange(len(x))
    for ax, sl, title in [(axs[0], slice(None), f"Whole signal (L = {len(x)} samples, not a multiple of 1024)"),
                          (axs[1], slice(0, 3000), "First 3000 samples (start boundary)"),
                          (axs[2], slice(len(x) - 3000, len(x)), "Last 3000 samples (end boundary)")]:
        ax.semilogy(n[sl] / FS, err[sl], color=C_BASE, lw=0.6)
        ax.axhline(1 / 32768, color=INK2, ls="--", lw=1)
        ax.set_ylim(1e-17, 1e-3)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("Time (s)")
    axs[0].set_ylabel("|reconstructed − original|")
    axs[0].text(0.01, 1.6 / 32768, "1 LSB of 16-bit audio", color=INK2, fontsize=8.5,
                transform=axs[0].get_yaxis_transform())
    fig.suptitle(f"MDCT → IMDCT without quantization: max error {err.max():.1e} "
                 f"(castanets 1.0–3.57 s, sine window 2048, hop 1024, zero-padded boundaries)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_mdct_reconstruction.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- frame choice
def pick_attack_frame(x, out_imp, rate=64):
    """A castanets frame inside the rhythm (after 1 s, so the signal before the attack is
    real, not digital silence), whose window has the attack in its second half and whose
    temporal factor is not clipped in most groups (mean > -30 dB); among those, the one where the improved PAM moved the
    most extra bits into the frame (from the per-frame encoder statistics)."""
    stb = json.loads(next((RES / "castanets" / "stats").glob(f"castanets__baseline__target{rate:03d}k*.json")).read_text())
    sti = json.loads(next((RES / "castanets" / "stats").glob(f"castanets__improved__target{rate:03d}k*.json")).read_text())
    extra = np.array(sti["frame_bits"]) - np.array(stb["frame_bits"])
    attacks = [a for a in detect_attacks(x) if a > FS]
    best, best_val = None, -np.inf
    for a in attacks:
        for f in range(max(1, a // M - 1), a // M + 2):
            pos = a - (f * M - M)  # attack position inside the frame's 2048-sample window
            fdb = 10 * np.log10(out_imp["temporal_factor"][f])
            if 0.5 * N_WIN < pos < 0.85 * N_WIN and np.mean(fdb) > -30 and extra[f] > best_val:
                best, best_val = (f, a), extra[f]
    return best


# ----------------------------------------------------------------------------- 2
def fig_masking(name, f, attack=None, rate=64, fname=None):
    x = load(name)
    out_b = BaselinePAM().analyze(x)
    out_i = pt.TemporalPAM().analyze(x)
    X = mdct.mdct(x)
    _, st_b = encode(x, rate * 1000, "baseline", pam_out=out_b, keep_coeffs=True)
    _, st_i = encode(x, rate * 1000, "improved", pam_out=out_i, keep_coeffs=True)
    t_c = mdct.frame_time(f)
    P = out_b["P_db"][f]
    fig, axs = plt.subplots(2, 1, figsize=(12.5, 8.2), sharex=True)
    ax = axs[0]
    ax.plot(FFT_FREQS[1:], P[1:], color=MUTED, lw=0.7, label="Input spectrum")
    ax.plot(FFT_FREQS[1:], ATH_DB[1:], color=INK2, ls="--", lw=1.1, label="Threshold in quiet")
    ax.plot(FFT_FREQS[1:], out_b["Tg_db"][f][1:], color=C_BASE, lw=1.8, label="Baseline global masking threshold")
    ax.plot(FFT_FREQS[1:], out_i["Tg_db"][f][1:], color=C_IMP, lw=1.8, label="Improved global masking threshold")
    tb, tdb, nb, ndb = out_b["maskers"][f]
    ax.plot(FFT_FREQS[tb], tdb, "x", color=INK, ms=6, mew=1.4, label="Tonal maskers")
    ax.plot(FFT_FREQS[nb], ndb, "o", mfc="none", mec=INK, ms=6, mew=1.2, label="Noise maskers")
    ax.set_ylabel("Level (dB SPL per FFT bin)")
    band = (FFT_FREQS > 100) & (FFT_FREQS < 16000)
    lo = max(-30, np.floor(min(np.percentile(P[1:850], 2), out_i["Tg_db"][f][band].min()) / 10) * 10 - 5)
    top = max(P[band].max(), out_b["Tg_db"][f][band].max()) + 12
    ax.set_ylim(lo, top)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    ax.set_title("(a) Spectrum and global masking curves", loc="left")

    ax = axs[1]
    widths = BAND_WIDTH
    for out, st, col, lab in [(out_b, st_b, C_BASE, "Baseline"), (out_i, st_i, C_IMP, "Improved")]:
        allow_db = per_line_db(out["allow"][f])
        edges = np.r_[BAND_LO_HZ, BAND_HI_HZ[-1]]
        ax.stairs(allow_db, np.maximum(edges, 1), color=col, lw=1.6, baseline=None,
                  label=f"{lab}: allowed noise\nper line (band allowance)")
        noise = sum_bands((X[f] - st["Xq"][f]) ** 2) / widths
        centres = np.sqrt(np.maximum(BAND_LO_HZ, 10) * BAND_HI_HZ)
        ax.plot(centres, per_line_db(noise), "o", color=col, ms=4.5, mec=SURF, mew=0.8,
                label=f"{lab}: actual coding\nnoise per line at {rate} kb/s")
    sig = sum_bands(X[f] ** 2) / widths
    ax.stairs(per_line_db(sig), np.maximum(np.r_[BAND_LO_HZ, BAND_HI_HZ[-1]], 1), color=MUTED,
              lw=1.0, baseline=None, label="Signal energy\nper line (MDCT)")
    ax.set_ylabel("Level (dB SPL per line)")
    ax.set_ylim(lo - 30, per_line_db(sig).max() + 12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5)
    ax.set_title("(b) Band-level allowances (after spreading over the critical band and taking the band "
                 "minimum) and resulting noise", loc="left")
    freq_axis(ax)
    att = (f", attack at {attack/FS:.3f} s ({(attack - (f*M - M))/FS*1000:.1f} ms into the window)"
           if attack is not None else "")
    fig.suptitle(f"{name.capitalize()}, MDCT frame {f}: window {(f*M-M)/FS:.3f}–{(f*M+M)/FS:.3f} s, "
                 f"centre {t_c:.3f} s{att}\n"
                 f"Bits spent on this frame at {rate} kb/s: baseline {st_b['frame_bits'][f]}, "
                 f"improved {st_i['frame_bits'][f]} (mean {st_b['mean_bits_per_frame']:.0f})\n"
                 "Normalization: 2048-pt Hann FFT; 0 dBFS sine = 90.3 dB SPL (±1 LSB ≈ 0 dB SPL)",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / (fname or f"fig2_masking_{name}_frame{f}.png"))
    plt.close(fig)
    return dict(frame=f, centre_s=t_c,
                baseline_bits=st_b["frame_bits"][f], improved_bits=st_i["frame_bits"][f],
                g_baseline=st_b["g"][f], g_improved=st_i["g"][f],
                factor_db=(10 * np.log10(out_i["temporal_factor"][f])).round(1).tolist())


# ----------------------------------------------------------------------------- 3
def fig_temporal(f, attack):
    x = load("castanets")
    xp = mdct.pad(x)
    E = pt.subblock_energies(xp)
    Em = pt.masking_envelope(E)
    sl = slice(f * (M // pt.SUB), f * (M // pt.SUB) + pt.SUB_PER_FRAME)
    seg = xp[f * M: f * M + N_WIN]
    t0 = (f * M - M) / FS
    tt = t0 + np.arange(N_WIN) / FS
    g = 5  # 4.8-6.4 kHz group
    fac = pt.temporal_factors(x)[f]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw=dict(height_ratios=[1, 1.2, 1]))
    ax = axs[0]
    ax.plot(tt * 1000, seg, color=INK2, lw=0.6, label="Input (frame window)")
    w2 = np.sin(np.pi * (np.arange(N_WIN) + 0.5) / N_WIN) ** 2
    ax.plot(tt * 1000, w2 * np.max(np.abs(seg)), color=C_IMP, lw=1.6,
            label="Spread of this frame's quantization noise (sine window²)")
    ax.axvline(attack / FS * 1000, color=INK, ls=":", lw=1)
    ax.text(attack / FS * 1000, ax.get_ylim()[1] * 0.85, " attack", fontsize=8.5)
    ax.set_ylabel("Amplitude")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_title("(a) The 42.7 ms window spreads coding noise over the quiet part before the attack", loc="left")
    ax = axs[1]
    tsb = t0 * 1000 + (np.arange(pt.SUB_PER_FRAME) + 0.5) * pt.SUB / FS * 1000
    e_bar = pt._H2 @ E[sl]
    ref = e_bar[g]
    ax.step(tsb, 10 * np.log10(E[sl, g] / ref), where="mid", color=MUTED, lw=1.4,
            label="Sub-block energy (2.67 ms), 4.8–6.4 kHz")
    ax.step(tsb, 10 * np.log10(Em[sl, g] / ref), where="mid", color=C_IMP, lw=1.8,
            label="Temporal masking envelope (pre/post-masking)")
    ax.axhline(0, color=C_BASE, lw=1.8, label="What the baseline assumes: window-average energy")
    ax.step(tsb, 10 * np.log10(np.maximum(Em[sl, g] / (ref * pt._W2), 1e-12)), where="mid",
            color=INK, lw=1, ls="--", label="Envelope ÷ noise share  →  min = factor F")
    ax.axvline(attack / FS * 1000, color=INK, ls=":", lw=1)
    ax.set_ylim(-60, 25)
    ax.set_xlim(tt[0] * 1000, tt[-1] * 1000)
    axs[0].set_xlim(tt[0] * 1000, tt[-1] * 1000)
    ax.set_ylabel("dB re window average")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("(b) Local masking vs. the window average in one frequency group", loc="left")
    ax = axs[2]
    labels = [f"{a/1000:g}–{b/1000:g}" for a, b in zip(pt.GROUP_EDGES_HZ[:-1], np.minimum(pt.GROUP_EDGES_HZ[1:], 24000))]
    ax.bar(np.arange(len(fac)), 10 * np.log10(fac), color=C_IMP, width=0.6)
    ax.axhline(0, color=INK2, lw=0.8)
    for i, v in enumerate(10 * np.log10(fac)):
        ax.text(i, v - 1.5, f"{v:.0f}", ha="center", va="top", fontsize=8, color=INK)
    ax.set_xticks(np.arange(len(fac)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Frequency group (kHz)")
    ax.set_ylabel("Threshold change (dB)")
    ax.set_ylim(min(-40, 10 * np.log10(fac).min() - 6), 5)
    ax.set_title("(c) Resulting change to the masking part of the threshold in this frame (F, clipped at −35 dB)", loc="left")
    fig.suptitle(f"Castanets frame {f}: how the temporal model tightens the threshold before an attack",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_temporal_model.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 4
def fig_bits(name="castanets", rate=64, t_range=(0.5, 3.0)):
    x = load(name)
    stb = json.loads(next((RES / name / "stats").glob(f"{name}__baseline__target{rate:03d}k*.json")).read_text())
    sti = json.loads(next((RES / name / "stats").glob(f"{name}__improved__target{rate:03d}k*.json")).read_text())
    F = len(stb["frame_bits"])
    t = np.arange(F) * M / FS
    sel = (t >= t_range[0]) & (t <= t_range[1])
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True, gridspec_kw=dict(height_ratios=[0.8, 1.2, 1]))
    ts = np.arange(len(x)) / FS
    s2 = (ts >= t_range[0]) & (ts <= t_range[1])
    axs[0].plot(ts[s2], x[s2], color=INK2, lw=0.4)
    for a in detect_attacks(x):
        if t_range[0] <= a / FS <= t_range[1]:
            for ax in axs:
                ax.axvline(a / FS, color=MUTED, lw=0.6, ls=":")
    axs[0].set_ylabel("Input")
    axs[0].set_title(f"{name.capitalize()} at {rate} kb/s (dotted lines: detected attacks)", loc="left")
    axs[1].step(t[sel], np.array(stb["frame_bits"])[sel], where="mid", color=C_BASE, label="Baseline")
    axs[1].step(t[sel], np.array(sti["frame_bits"])[sel], where="mid", color=C_IMP, label="Improved")
    axs[1].axhline(stb["mean_bits_per_frame"], color=INK2, ls="--", lw=1)
    axs[1].text(t_range[0], stb["mean_bits_per_frame"] - 60, "mean bits per frame", va="top",
                fontsize=8.5, color=INK2)
    axs[1].set_ylabel("Bits in frame")
    axs[1].legend(loc="upper left", ncol=2)
    axs[1].set_title("(a) Bits per frame: the improved PAM concentrates bits on frames whose window contains an attack", loc="left")
    axs[2].plot(t[sel], np.array(stb["reservoir"])[sel], color=C_BASE, label="Baseline")
    axs[2].plot(t[sel], np.array(sti["reservoir"])[sel], color=C_IMP, label="Improved")
    axs[2].set_ylabel("Reservoir (bits)")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc="upper left", ncol=2)
    axs[2].set_title("(b) Bit-reservoir content (cap 6144 − mean frame bits)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / f"fig4_bits_per_frame_{name}_{rate}k.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 5
def fig_pre_echo(name, attack, rate=64, fname=None):
    x = load(name)
    yb, _ = decoded(name, "baseline", rate)
    yi, _ = decoded(name, "improved", rate)
    s, e = attack - int(0.045 * FS), attack + int(0.015 * FS)
    t = (np.arange(s, e) - attack) / FS * 1000
    hp = lambda v: np.diff(v, prepend=0.0)  # same +6 dB/oct weighting as the pre-echo metric
    blk = FS // 1000

    def env(v):
        v = hp(v)[s:e]
        n = len(v) // blk
        r = np.sqrt(np.mean(v[: n * blk].reshape(n, blk) ** 2, axis=1))
        return (np.arange(n) + 0.5) - 45, 20 * np.log10(r + 1e-9)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw=dict(height_ratios=[0.8, 1.4]))
    axs[0].plot(t, x[s:e], color=INK2, lw=0.7)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title(f"(a) {name.capitalize()}: original around the attack at {attack/FS:.3f} s", loc="left")
    tt, eo = env(x)
    _, eb = env(yb - x)
    _, ei = env(yi - x)
    ax = axs[1]
    ax.step(tt, eo, where="mid", color=MUTED, lw=1.6, label="Original (high-passed)")
    ax.step(tt, eb, where="mid", color=C_BASE, lw=1.8, label=f"Coding error, baseline PAM")
    ax.step(tt, ei, where="mid", color=C_IMP, lw=1.8, label=f"Coding error, improved PAM")
    ax.set_ylabel("1 ms RMS level (dBFS, high-passed)")
    ax.set_xlabel("Time relative to attack (ms)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title(f"(b) Error level vs. signal level at {rate} kb/s (same file size for both PAMs)", loc="left")
    lo = np.floor(min(eo.min(), eb.min(), ei.min()) / 10) * 10
    ax.set_ylim(max(lo, -110), max(eo.max(), eb.max()) + 8)
    for a_ in axs:
        a_.axvspan(-20, -2, color=GRID, alpha=0.7, lw=0)
        a_.axvline(0, color=INK, ls=":", lw=1)
    axs[0].text(-19.5, axs[0].get_ylim()[1] * 0.8, "pre-echo window (−20…−2 ms)", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / (fname or f"fig5_pre_echo_{name}_{rate}k.png"))
    plt.close(fig)


# ----------------------------------------------------------------------------- 6
def _rate_panels(rows, metrics, models, fname, title, figsize=(10.5, 8)):
    names = ["castanets", "speech", "harpsichord"]
    fig, axs = plt.subplots(len(metrics), 3, figsize=figsize, sharex=True, squeeze=False)
    for j, n in enumerate(names):
        for i, (key, lab) in enumerate(metrics):
            ax = axs[i, j]
            for model in models:
                label, col, mk, ls = STYLE[model]
                rr = sorted([r for r in rows if r["excerpt"] == n and r["model"] == model],
                            key=lambda r: int(r["target_kbps"]))
                if not rr:
                    continue
                xs = [float(r["actual_kbps"]) for r in rr]
                ys = [float(r[key]) for r in rr]
                ax.plot(xs, ys, marker=mk, ls=ls, color=col, ms=5, mec=SURF, mew=1, label=label,
                        lw=1.2 if ls != "-" else 1.8)
            if i == 0:
                ax.set_title(n.capitalize() + (" (held out)" if n == "harpsichord" else ""))
            if j == 0:
                ax.set_ylabel(lab)
            ax.set_xticks([64, 96, 128])
            ax.set_xlim(52, 134)
        axs[-1, 1].set_xlabel("Actual bitrate (kb/s)")
    return fig, axs


def fig_rate_quality():
    rows = list(csv.DictReader(open(RES / "summary.csv")))
    metrics = [("total_nmr_db", "Total NMR (dB)\nlower is better"),
               ("mean_pre_echo_to_signal_db", "Pre-echo-to-signal (dB)\nlower is better"),
               ("disturbed_frames_pct", "Disturbed frames (%)\nlower is better")]
    fig, axs = _rate_panels(rows, metrics, list(STYLE), "fig6_rate_quality.png", None)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Objective comparison at matched bitrates (independent Johnston-style NMR model; "
                 "high-passed pre-echo measure)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig6_rate_quality.png")
    plt.close(fig)


def fig_peaq():
    """PEAQ ODG (BS.1387 basic version, GstPEAQ core) vs bitrate."""
    rows = list(csv.DictReader(open(RES / "peaq.csv")))
    fig, axs = _rate_panels(rows, [("ODG", "PEAQ ODG (higher is better)")], list(STYLE),
                            "fig9_peaq.png", None, figsize=(10.5, 4.2))
    grades = [(0, "imperceptible"), (-1, "perceptible, not annoying"), (-2, "slightly annoying"),
              (-3, "annoying"), (-4, "very annoying")]
    for ax in axs[0]:
        ax.set_ylim(-4, 0.3)
        ax.set_yticks([g for g, _ in grades])
    axs[0, 0].set_yticklabels([f"{g}  {t}" for g, t in grades], fontsize=7.5)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("PEAQ objective difference grade (ITU-R BS.1387-1 basic version)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "fig9_peaq.png")
    plt.close(fig)


# ----------------------------------------------------------------------------- 8
def fig_block_switching(name, attack, rate=64):
    """Window sequence around an attack and the coding error with and without block switching."""
    x = load(name)
    seq = bs.choose_sequence(x)
    s, e = attack - int(0.060 * FS), attack + int(0.030 * FS)
    t_ms = lambda n: (np.asarray(n) - attack) / FS * 1000  # noqa: E731
    fig, axs = plt.subplots(3, 1, figsize=(10, 7.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.8, 0.7, 1.5]))
    axs[0].plot(t_ms(np.arange(s, e)), x[s:e], color=INK2, lw=0.7)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title(f"(a) {name.capitalize()}: original around the attack at {attack/FS:.3f} s", loc="left")

    # (b) the windows of the frames that overlap the plotted span
    ax = axs[1]
    cols = {bs.ONLY_LONG: MUTED, bs.LONG_START: C_CTL, bs.EIGHT_SHORT: INK, bs.LONG_STOP: C_CTL}
    n_frame = np.arange(N_WIN)
    for f in range(len(seq)):
        a0 = (f - 1) * M                      # original-sample index of the frame's first sample
        if a0 + N_WIN < s or a0 > e:
            continue
        if seq[f] == bs.EIGHT_SHORT:
            for w in range(bs.N_SUBWIN):
                lo, hi = bs.short_window_span(w)
                ax.plot(t_ms(a0 + np.arange(lo, hi)), bs.SHORT_WINDOW, color=cols[seq[f]], lw=1.1)
        else:
            ax.plot(t_ms(a0 + n_frame), bs.WINDOWS[seq[f]], color=cols[seq[f]], lw=1.3)
        mid = t_ms(a0 + M)
        if t_ms(s) < mid < t_ms(e):
            ax.text(mid, 1.08, bs.SEQ_NAMES[seq[f]].upper(), ha="center", fontsize=7.5,
                    color=INK2 if seq[f] != bs.EIGHT_SHORT else INK)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 1])
    ax.set_ylabel("Window")
    ax.set_title("(b) Window sequence chosen by the transient detector (one label per 1024-sample frame)",
                 loc="left")

    hp = lambda v: np.diff(v, prepend=0.0)  # noqa: E731
    blk = FS // 1000

    def env(v):
        v = hp(v)[s:e]
        n = len(v) // blk
        r = np.sqrt(np.mean(v[: n * blk].reshape(n, blk) ** 2, axis=1))
        return t_ms(s + (np.arange(n) + 0.5) * blk), 20 * np.log10(r + 1e-9)

    ax = axs[2]
    tt, eo = env(x)
    ax.step(tt, eo, where="mid", color=MUTED, lw=1.6, label="Original (high-passed)")
    lows = [eo.min()]
    for model in ["baseline", "improved", "baseline_bs", "improved_bs"]:
        y, _ = decoded(name, model, rate)
        _, ee = env(y - x)
        lows.append(ee.min())
        label, col, _, ls = STYLE[model]
        ax.step(tt, ee, where="mid", color=col, ls=ls, lw=1.8 if ls == "-" else 1.1,
                label=f"Error, {label[0].lower() + label[1:]}")
    ax.set_ylabel("1 ms RMS level (dBFS, high-passed)")
    ax.set_xlabel("Time relative to attack (ms)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"(c) Coding error at {rate} kb/s (all four files the same size)", loc="left")
    ax.set_ylim(max(np.floor(min(lows) / 10) * 10, -110), eo.max() + 8)
    for a_ in (axs[0], axs[2]):
        a_.axvspan(-20, -2, color=GRID, alpha=0.7, lw=0)
    for a_ in axs:
        a_.axvline(0, color=INK, ls=":", lw=1)
    axs[2].set_xlim(t_ms(s), t_ms(e))
    fig.tight_layout()
    fig.savefig(FIG / f"fig8_block_switching_{name}_{rate}k.png")
    plt.close(fig)
    return dict(window_sequence_near_attack=[bs.SEQ_NAMES[int(seq[f])] for f in
                                            range((attack + 448) // M - 3, (attack + 448) // M + 4)])


# ----------------------------------------------------------------------------- 7
def fig_error_spectrogram(name="castanets", rate=64, t_range=(0.75, 1.65)):
    x = load(name)
    yb, _ = decoded(name, "baseline", rate)
    yi, _ = decoded(name, "improved", rate)
    n, hop = 512, 128
    win = np.hanning(n)

    def spec(s):
        a, b = int(t_range[0] * FS), int(t_range[1] * FS)
        s = s[a:b]
        idx = np.arange(0, len(s) - n, hop)[:, None] + np.arange(n)[None, :]
        S = np.abs(np.fft.rfft(s[idx] * win, axis=1)) ** 2
        return 10 * np.log10(S.T + 1e-12)

    panels = [("Original", spec(x)), ("Error, baseline", spec(yb - x)), ("Error, improved", spec(yi - x))]
    vmax = panels[0][1].max()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (title, S) in zip(axs, panels):
        im = ax.imshow(S, origin="lower", aspect="auto", cmap="Greys", vmin=vmax - 80, vmax=vmax,
                       extent=[t_range[0], t_range[1], 0, FS / 2000])
        ax.set_title(f"{title} ({rate} kb/s)", loc="left")
        ax.set_xlabel("Time (s)")
        ax.grid(False)
    axs[0].set_ylabel("Frequency (kHz)")
    cb = fig.colorbar(im, ax=axs, shrink=0.9, pad=0.01)
    cb.set_label("dB (relative)")
    fig.savefig(FIG / f"fig7_error_spectrogram_{name}_{rate}k.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    info = {}
    fig_mdct()
    x = load("castanets")
    out_i = pt.TemporalPAM().analyze(x)
    f, attack = pick_attack_frame(x, out_i)
    info["castanets_attack_frame"] = fig_masking("castanets", f, attack, 64,
                                                 fname="fig2a_masking_castanets_attack_frame.png")
    info["castanets_attack_frame"]["attack_s"] = attack / FS
    # A sustained harpsichord frame (no attack in the window): curves should nearly coincide
    xh = load("harpsichord")
    ah = detect_attacks(xh)
    cand = [fr for fr in range(60, 400) if not any(abs(a - fr * M) < 1.5 * N_WIN for a in ah)
            and np.mean(mdct.frames(xh)[fr] ** 2) > 10 ** (-40 / 10)]
    fh = cand[len(cand) // 2]
    info["harpsichord_sustained_frame"] = fig_masking("harpsichord", fh, None, 64,
                                                      fname="fig2b_masking_harpsichord_sustained_frame.png")
    fig_temporal(f, attack)
    fig_bits("castanets", 64)
    fig_pre_echo("castanets", attack, 64)
    ah7 = [a for a in ah if a / FS > 1.0]
    fig_pre_echo("harpsichord", ah7[0], 64)
    fig_rate_quality()
    fig_error_spectrogram("castanets", 64)
    if any((RES / "castanets").glob("castanets__improved_bs__*.wav")):
        info["block_switching"] = fig_block_switching("castanets", attack, 64)
    if (RES / "peaq.csv").exists():
        fig_peaq()
    (FIG / "figure_info.json").write_text(json.dumps(info, indent=1, default=float))
    print(json.dumps(info, indent=1, default=float))


if __name__ == "__main__":
    main()
