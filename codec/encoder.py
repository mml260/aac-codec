"""Encoder: PCM -> .pacc bitstream at a fixed target bitrate.

Rate control
  The whole file must not exceed  target_bps * duration  bits, counting the
  24-byte header, all side information, stuffing and the final byte pad.
  After the header, the remaining budget is shared evenly over the frames
  (mean_bits per frame) and a bit reservoir lets a frame use bits saved by
  earlier frames, as in MP3/AAC:
    * reservoir content is never negative (no borrowing from the future) and
      never exceeds RES_MAX = 6144 - mean_bits (AAC's 6144-bit decoder buffer
      per channel), so the stream is a valid constant-bitrate stream for a
      decoder with that buffer;
    * constant-quality target (two passes, the encoder works offline):
      pass 1 finds one global noise-to-mask offset g* such that the frames'
      total demand at g* equals the budget; pass 2 codes every frame at g*
      unless the reservoir forbids it, in which case that frame alone is
      quantised coarser (not enough bits) or finer (reservoir full);
    * if a frame cannot absorb a reservoir overflow (e.g. near-silence),
      stuffing bits are written; they are counted in the file size;
    * the reservoir cap ramps to zero over the last frames so the file ends
      within a byte of the exact budget.
  Where a frame's bits go is decided by the PAM's thresholds: frames whose
  thresholds are low relative to their energy need more bits at g*. The
  rate-control mechanism is identical for both PAMs.

Block switching (optional, block_switching=True, file version 2)
  The transient detector in blockswitch.py picks the window sequence, short
  frames get per-window allowances from the PAM and a grouping of their
  windows, and every frame is coded with its own band layout. Rate control
  is unchanged: short frames simply need more bits at g*, which the
  reservoir supplies, as in AAC.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import FS, M
from . import blockswitch as bs
from . import mdct
from .allocation import allocate, dequantize, evaluate, perceptual_entropy
from .bitio import BitWriter
from .bitstream import (HEADER_BITS, VERSION, VERSION_BS, fill_bits, frame_header_bits,
                        pack_header, write_frame)
from .wavio import read_wav

G_MIN, G_MAX = -60.0, 90.0   # search range of the noise-to-mask offset (dB)
DECODER_BUFFER = 6144
DRAIN_FRAMES = 16  # reservoir cap ramps to 0 over the last frames


def get_pam(model):
    if model == "baseline":
        from .pam_baseline import BaselinePAM
        return BaselinePAM()
    if model == "improved":
        from .pam_temporal import TemporalPAM
        return TemporalPAM()
    if model == "control":
        from .pam_control import ControlPAM
        return ControlPAM()
    raise ValueError(model)


def prepare_frames(x, model="baseline", block_switching=False, pam_out=None, seq=None):
    """Transform + PAM for every frame.

    Returns a list of per-frame dicts (coded-order coefficients X, allowances,
    layout, window sequence, grouping, v2 header bits) and the PAM output.
    """
    x = np.asarray(x, dtype=np.float64)
    if not block_switching:
        X = mdct.mdct(x)
        if pam_out is None:
            pam_out = get_pam(model).analyze(x)
        frames = [dict(X=X[f], allow=pam_out["allow"][f], layout=bs.LONG_LAYOUT, seq=None,
                       groups=None, hdr=0) for f in range(X.shape[0])]
        return frames, pam_out
    if seq is None:
        seq = bs.choose_sequence(x)
    X = bs.analysis(x, seq)
    if pam_out is None:
        pam_out = get_pam(model).analyze(x, seq=seq)
    frames = []
    for f, s in enumerate(seq):
        s = int(s)
        if s == bs.EIGHT_SHORT:
            groups = bs.choose_grouping(X[f])
            layout = bs.short_layout(groups)
            a8 = pam_out["allow_short"][f]                       # (8 windows, 14 bands)
            edges = np.cumsum((0,) + groups)
            allow = np.concatenate([a8[edges[i]:edges[i + 1]].min(axis=0)  # shared sf: strictest window
                                    for i in range(len(groups))])
            frames.append(dict(X=layout.to_coded(X[f]), allow=allow, layout=layout, seq=s,
                               groups=groups, hdr=frame_header_bits(s)))
        else:
            frames.append(dict(X=X[f], allow=pam_out["allow"][f], layout=bs.LONG_LAYOUT, seq=s,
                               groups=None, hdr=frame_header_bits(s)))
    pam_out["seq"] = np.asarray(seq)
    return frames, pam_out


def encode(x, target_bps, model="baseline", pam_out=None, keep_coeffs=False,
           block_switching=False, seq=None):
    x = np.asarray(x, dtype=np.float64)
    L = len(x)
    frames, pam_out = prepare_frames(x, model, block_switching, pam_out, seq)
    F = len(frames)
    pe = np.array([perceptual_entropy(fr["X"], fr["allow"], fr["layout"]) for fr in frames])

    def ev(f, g):
        fr = frames[f]
        return evaluate(fr["X"], fr["allow"], g, fr["layout"], fr["hdr"])

    def alloc(f, target, **kw):
        fr = frames[f]
        return allocate(fr["X"], fr["allow"], target, layout=fr["layout"], header_bits=fr["hdr"], **kw)

    total_bits = int(np.floor(target_bps * L / FS))
    mean_bits = (total_bits - HEADER_BITS - 7) / F   # 7 = worst-case final byte pad
    res_max = max(DECODER_BUFFER - mean_bits, 0.0)

    # Pass 1: global noise-to-mask offset g* whose total demand equals the budget.
    def total_demand(g):
        return sum(ev(f, g)["bits"] for f in range(F))

    budget_frames = mean_bits * F
    lo, hi = G_MIN, G_MAX
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if total_demand(mid) > budget_frames:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.01:
            break
    g_star = hi

    # Pass 2: encode frame by frame at g*, within the reservoir limits.
    bw = BitWriter()
    reservoir = 0.0
    Xq = np.zeros((F, M)) if keep_coeffs else None
    stats = dict(frame_bits=[], side_bits=[], payload_bits=[], fill_bits=[], reservoir=[],
                 demand=[], g=[], active_bands=[], pe=pe.tolist(), g_star=g_star)
    if block_switching:
        stats["window_sequence"] = [fr["seq"] for fr in frames]
        stats["groups"] = [list(fr["groups"]) if fr["groups"] else None for fr in frames]
    for f in range(F):
        fr = frames[f]
        available = reservoir + mean_bits
        cap = res_max * min(1.0, (F - 1 - f) / DRAIN_FRAMES)
        must_use = max(0.0, available - cap)
        r = ev(f, g_star)
        demand = r["bits"]
        if demand > available:                       # buffer limit: quantise coarser
            r = alloc(f, int(np.floor(available)), g_lo=g_star)
        elif demand < must_use:                      # reservoir full: spend the excess
            r = alloc(f, int(np.floor(must_use)), g_hi=g_star)
        used = r["bits"]
        n_fill = 0
        if used < must_use:
            # stuffing: just enough to keep the reservoir at its cap
            deficit = int(np.ceil(must_use - used))
            room = int(np.floor(available - used))
            n_fill = 1
            while fill_bits(n_fill) < deficit:
                n_fill += 1
            while n_fill > 0 and fill_bits(n_fill) > room:
                n_fill -= 1
        start = len(bw)
        write_frame(bw, r["active"], r["sf"], r["k"], r["q"], n_fill,
                    layout=fr["layout"], seq=fr["seq"], groups=fr["groups"])
        written = len(bw) - start
        expect = used + fill_bits(n_fill)
        if written != expect:
            raise AssertionError(f"frame {f}: bit count mismatch {written} != {expect}")
        reservoir = available - written
        if keep_coeffs:
            Xq[f] = fr["layout"].to_natural(dequantize(r["q"], r["sf"], fr["layout"]))
        stats["frame_bits"].append(written)
        stats["side_bits"].append(r["side"])
        stats["payload_bits"].append(r["payload"])
        stats["fill_bits"].append(fill_bits(n_fill))
        stats["reservoir"].append(reservoir)
        stats["demand"].append(demand)
        stats["g"].append(r["g"])
        stats["active_bands"].append(int(r["active"].sum()))

    payload = bw.tobytes()
    version = VERSION_BS if block_switching else VERSION
    data = pack_header(L, F, target_bps, model, version) + payload
    stats.update(
        model=model, block_switching=bool(block_switching), target_bps=target_bps, samples=L,
        n_frames=F, total_budget_bits=total_bits, mean_bits_per_frame=mean_bits,
        header_bits=HEADER_BITS, frames_bits=len(bw),
        pad_bits=8 * len(payload) - len(bw), file_bytes=len(data),
        actual_bps=8 * len(data) * FS / L,
    )
    if keep_coeffs:
        stats["Xq"] = Xq
        if block_switching:
            stats["seq"] = np.array([fr["seq"] for fr in frames])
    return data, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description="Encode a 48 kHz mono 16-bit WAV.")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--rate", type=float, required=True, help="target bitrate in kb/s")
    ap.add_argument("--model", choices=["baseline", "improved", "control"], default="baseline")
    ap.add_argument("--block-switching", action="store_true",
                    help="switch to eight short windows on attacks (file version 2)")
    ap.add_argument("--stats", help="optional JSON file for per-frame statistics")
    a = ap.parse_args(argv)
    x, fs = read_wav(a.input)
    if fs != FS or x.shape[1] != 1:
        sys.exit(f"input must be {FS} Hz mono (got {fs} Hz, {x.shape[1]} ch); "
                 "run scripts/prepare_inputs.py first")
    data, stats = encode(x[:, 0], a.rate * 1000, a.model, block_switching=a.block_switching)
    Path(a.output).write_bytes(data)
    if a.stats:
        Path(a.stats).write_text(json.dumps(stats))
    print(f"{a.input} -> {a.output}: {stats['file_bytes']} bytes, "
          f"{stats['actual_bps']/1000:.3f} kb/s (target {a.rate:g})")


if __name__ == "__main__":
    main()
