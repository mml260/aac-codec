"""PEAQ (ITU-R BS.1387-1, basic version) for every decoded file in results/.

Uses the core of GstPEAQ (Holters & Zoelzer, DAFx 2015; LGPL), built without
GStreamer: tools/peaq/ holds a stand-alone driver that mirrors GstPEAQ's own
basic-version processing loop, and a small FFT stand-in for GStreamer's FFT.

Build once (needs git, gcc, glib/gobject development files):

    git clone https://github.com/HSU-ANT/gstpeaq tools/peaq/gstpeaq
    python3 scripts/run_peaq.py --build

Then:  python3 scripts/run_peaq.py          -> results/peaq.csv

Check of the build (run with --selftest): GstPEAQ's own regression values are
reproduced (440 Hz sine vs itself: 0.171; 440 Hz sawtooth vs triangle: -2.007).
Playback level: full-scale sine = 92 dB SPL (the BS.1387 / GstPEAQ default).
"""
import argparse
import csv
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec.wavio import read_wav  # noqa: E402

TOOL = ROOT / "tools" / "peaq"
BIN = TOOL / "peaq_basic"
CORE = ["earmodel.c", "fftearmodel.c", "fbearmodel.c", "leveladapter.c",
        "modpatt.c", "movaccum.c", "movs.c", "nn.c"]
MOVS = ["BandwidthRefB", "BandwidthTestB", "TotalNMRB", "WinModDiff1B", "ADBB",
        "EHSB", "AvgModDiff1B", "AvgModDiff2B", "RmsNoiseLoudB", "MFPDB",
        "RelDistFramesB", "DI", "ODG"]
NAME = re.compile(r"(?P<ex>\w+?)__(?P<model>\w+?)__target(?P<rate>\d+)k__actual(?P<act>[\d.]+)k\.wav$")


def build(src):
    src = Path(src) / "src"
    flags = subprocess.run(["pkg-config", "--cflags", "--libs", "gobject-2.0"],
                           capture_output=True, text=True, check=True).stdout.split()
    cmd = (["gcc", "-O2", f"-I{TOOL}", f"-I{src}", "-o", str(BIN),
            str(TOOL / "peaq_basic_cli.c"), str(TOOL / "gstfftf64_shim.c")]
           + [str(src / c) for c in CORE] + flags + ["-lm"])
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    print("built", BIN)


def peaq(ref, test, tmp):
    paths = []
    for i, x in enumerate((ref, test)):
        p = Path(tmp) / f"{i}.f32"
        np.asarray(x, dtype="<f4").tofile(p)
        paths.append(str(p))
    out = subprocess.run([str(BIN)] + paths, capture_output=True, text=True, check=True).stdout
    return {k: float(v) for k, v in (ln.split() for ln in out.strip().splitlines())}


def to_float(path):
    x = read_wav(path)
    x = x[0] if isinstance(x, tuple) else x
    x = np.asarray(x)
    return x / 32768.0 if np.issubdtype(x.dtype, np.integer) else x


def selftest(tmp):
    fs, n, f = 48000, 128 * 1024, 440.0
    a = np.mod(2 * np.pi * f / fs * np.arange(1, n + 1), 2 * np.pi)
    sine = 0.8 * np.sin(a)
    saw = np.where(a < np.pi, 0.8 / np.pi * a, -0.8 / np.pi * (2 * np.pi - a))
    k = 0.8 / (np.pi / 2)
    tri = np.where(a < np.pi / 2, k * a, np.where(a < 1.5 * np.pi, k * (np.pi - a), k * (a - 2 * np.pi)))
    r1, r2 = peaq(sine, sine, tmp)["ODG"], peaq(saw, tri, tmp)["ODG"]
    print(f"sine vs sine: {r1:.3f} (GstPEAQ: 0.171); saw vs triangle: {r2:.3f} (GstPEAQ: -2.007)")
    return round(r1, 3) == 0.171 and round(r2, 3) == -2.007


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--src", default=str(TOOL / "gstpeaq"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.build or not BIN.exists():
        build(a.src)
    with tempfile.TemporaryDirectory() as tmp:
        if not selftest(tmp):
            sys.exit("PEAQ self-test failed")
        if a.selftest:
            return
        rows = []
        for exdir in sorted(p for p in (ROOT / "results").iterdir() if p.is_dir()):
            ref = to_float(exdir / f"{exdir.name}__original.wav")
            for w in sorted(exdir.glob("*__target*k__actual*k.wav")):
                m = NAME.match(w.name)
                r = peaq(ref, to_float(w), tmp)
                rows.append({"excerpt": m["ex"], "model": m["model"], "target_kbps": int(m["rate"]),
                             "actual_kbps": float(m["act"]), **{k: round(r[k], 4) for k in MOVS}})
                print(f"{m['ex']:12s} {m['model']:9s} {int(m['rate']):4d}k  ODG {r['ODG']:7.3f}")
            # anchor: the reference against itself
            r = peaq(ref, ref, tmp)
            rows.append({"excerpt": exdir.name, "model": "original", "target_kbps": 768,
                         "actual_kbps": 768.0, **{k: round(r[k], 4) for k in MOVS}})
    out = ROOT / "results" / "peaq.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
