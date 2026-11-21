"""Encode + decode all conditions and measure them.

  3 excerpts x 2 PAMs (baseline, improved) x 3 target rates (64, 96, 128 kb/s)
  (the 18 required conditions), plus the optional extra models:
    control                         baseline PAM + crude attack boost
    baseline_bs, improved_bs        the two PAMs with block switching (v2 files)

Outputs (results/):
  <excerpt>/<excerpt>__<model>__target<RRR>k__actual<AAA.AAA>k.pacc   encoded file
  <excerpt>/<excerpt>__<model>__target<RRR>k__actual<AAA.AAA>k.wav    decoded, 48 kHz/16-bit
  <excerpt>/<excerpt>__original.wav                                   the prepared input
  <excerpt>/stats/<...>.json      per-frame encoder statistics
  summary.csv, summary.json       sizes, rates, overhead, objective metrics

The decoded WAVs are produced by the stand-alone decoder (python -m codec.decoder)
run in a separate process, sample-aligned with the original (no delay) and at
the original level (no gain or normalisation; any clipping is counted).

Usage:  python scripts/run_all.py [--excerpts castanets,speech] [--models ...] [--rates 64,96,128]
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec import FS  # noqa: E402
from codec import blockswitch as bs  # noqa: E402
from codec.encoder import encode, get_pam  # noqa: E402
from codec.wavio import read_wav  # noqa: E402
from evaluation.metrics import all_metrics  # noqa: E402
from scripts.prepare_inputs import pcm_sha256  # noqa: E402

UNCOMPRESSED_BPS = FS * 16  # 768 kb/s reference (48 kHz mono 16-bit PCM)


def check_manifest(name, path):
    man = json.loads((ROOT / "data" / "prepared" / "manifest.json").read_text())
    digest = pcm_sha256(path)
    if man[name]["sha256"] != digest:
        sys.exit(f"{path} does not match manifest; re-run scripts/prepare_inputs.py")
    return digest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--excerpts", default="castanets,harpsichord,speech")
    ap.add_argument("--models", default="baseline,improved")
    ap.add_argument("--rates", default="64,96,128")
    a = ap.parse_args(argv)
    out_root = ROOT / "results"
    rows = []
    for name in a.excerpts.split(","):
        src = ROOT / "data" / "prepared" / f"{name}.wav"
        digest = check_manifest(name, src)
        x, _ = read_wav(src)
        x = x[:, 0]
        d = out_root / name
        (d / "stats").mkdir(parents=True, exist_ok=True)
        shutil.copy(src, d / f"{name}__original.wav")
        seq = None
        for model in a.models.split(","):
            t0 = time.time()
            block_switching = model.endswith("_bs")
            pam_name = model[:-3] if block_switching else model
            if block_switching:
                seq = bs.choose_sequence(x) if seq is None else seq
                pam_out = get_pam(pam_name).analyze(x, seq=seq)
            else:
                pam_out = get_pam(pam_name).analyze(x)
            for rate in [int(r) for r in a.rates.split(",")]:
                data, st = encode(x, rate * 1000, pam_name, pam_out=pam_out,
                                  block_switching=block_switching, seq=seq)
                stem = f"{name}__{model}__target{rate:03d}k__actual{st['actual_bps']/1000:.3f}k"
                for old in d.glob(f"{name}__{model}__target{rate:03d}k__*"):
                    old.unlink()
                pacc = d / f"{stem}.pacc"
                wav = d / f"{stem}.wav"
                pacc.write_bytes(data)
                # decode with the stand-alone decoder in a separate process
                r = subprocess.run([sys.executable, "-m", "codec.decoder", str(pacc), str(wav)],
                                   cwd=ROOT, capture_output=True, text=True)
                if r.returncode != 0:
                    sys.exit(r.stderr)
                y, _ = read_wav(wav)
                m = all_metrics(x, y[:, 0])
                st_small = {k: v for k, v in st.items()}
                (d / "stats" / f"{stem}.json").write_text(json.dumps(st_small))
                frame_bits = np.array(st["frame_bits"])
                side = int(np.sum(st["side_bits"]))
                payload = int(np.sum(st["payload_bits"]))
                fill = int(np.sum(st["fill_bits"]))
                row = dict(
                    excerpt=name, model=model, target_kbps=rate,
                    file=pacc.name, bytes=len(data),
                    actual_kbps=round(st["actual_bps"] / 1000, 3),
                    rate_error_pct=round(100 * (st["actual_bps"] / (rate * 1000) - 1), 4),
                    compression_ratio=round(UNCOMPRESSED_BPS / st["actual_bps"], 3),
                    header_bits=st["header_bits"], side_info_bits=side, payload_bits=payload,
                    stuffing_bits=fill, pad_bits=st["pad_bits"],
                    side_info_pct=round(100 * side / (8 * len(data)), 2),
                    stuffing_pct=round(100 * fill / (8 * len(data)), 2),
                    g_star_db=round(st["g_star"], 2),
                    mean_active_bands=round(float(np.mean(st["active_bands"])), 1),
                    max_frame_bits=int(frame_bits.max()),
                    short_frames=int(np.sum(np.array(st.get("window_sequence", [0])) == bs.EIGHT_SHORT)),
                    clipped_samples=int(np.sum(np.abs(y[:, 0]) >= 32767 / 32768)),
                    input_sha256=digest[:16],
                    **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()},
                )
                rows.append(row)
                print(f"{name:12s} {model:11s} {rate:3d}k -> {row['actual_kbps']:8.3f} kb/s "
                      f"{row['bytes']:6d} B  NMR {row['total_nmr_db']:6.2f} dB  "
                      f"pre-echo {row['mean_pre_echo_to_signal_db']:6.2f} dB")
            print(f"  ({model} on {name}: {time.time()-t0:.1f} s)")
    if rows:
        keys = list(rows[0].keys())
        summary_csv = out_root / "summary.csv"
        existing = []
        if summary_csv.exists():
            with open(summary_csv) as fh:
                existing = [r for r in csv.DictReader(fh)
                            if not any(r["excerpt"] == n["excerpt"] and r["model"] == n["model"]
                                       and int(r["target_kbps"]) == n["target_kbps"] for n in rows)]
        allrows = existing + [{k: str(v) for k, v in r.items()} for r in rows]
        order = {"castanets": 0, "speech": 1, "harpsichord": 2}
        allrows.sort(key=lambda r: (order.get(r["excerpt"], 9), r["model"], int(r["target_kbps"])))
        with open(summary_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(allrows)
        (out_root / "summary.json").write_text(json.dumps(allrows, indent=1))


if __name__ == "__main__":
    main()
