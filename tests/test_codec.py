"""Codec tests: bit I/O, exact bit accounting, decoder == encoder reconstruction,
and decoding in a fresh process that has no access to the encoder, the PAMs,
or the original audio.

Run:  python tests/test_codec.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec import FS, mdct  # noqa: E402
from codec import blockswitch as bs  # noqa: E402
from codec.bitio import BitReader, BitWriter, eg0_len, seg0_len  # noqa: E402
from codec.decoder import decode  # noqa: E402
from codec.encoder import encode  # noqa: E402
from codec.wavio import read_wav  # noqa: E402

# Only these modules are copied next to the bitstream for the fresh-session decode
DECODER_MODULES = ["__init__.py", "decoder.py", "bitstream.py", "bitio.py", "bands.py",
                   "mdct.py", "blockswitch.py", "wavio.py"]


def report(ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    return ok


def test_bitio():
    rng = np.random.default_rng(0)
    bw = BitWriter()
    vals_u = rng.integers(0, 5000, 300)
    vals_s = rng.integers(-300, 300, 300)
    q = rng.laplace(0, 6, 500).round().astype(int)
    for u in vals_u:
        bw.eg0(u)
    for s in vals_s:
        bw.seg0(s)
    bw.rice_block(q, 2)
    bw.fixed(0b1011, 4)
    expected = int(np.sum(eg0_len(vals_u)) + np.sum(seg0_len(vals_s)))
    br = BitReader(bw.tobytes())
    ok = all(br.eg0() == u for u in vals_u)
    ok &= all(br.seg0() == s for s in vals_s)
    ok &= np.array_equal(br.rice_block(len(q), 2), q)
    ok &= br.fixed(4) == 0b1011
    ok &= expected <= len(bw)
    return report(ok, "bit writer/reader round trip (EG0, signed EG0, Rice, fixed)")


def test_codec_consistency(x, name, rate, model, block_switching=False):
    data, st = encode(x, rate, model, keep_coeffs=True, block_switching=block_switching)
    y, hdr = decode(data)
    if block_switching:
        y_enc = bs.synthesis(st["Xq"], st["seq"], len(x))
        model = model + "+BS"
    else:
        y_enc = mdct.imdct(st["Xq"], len(x))
    same = np.max(np.abs(y - y_enc)) < 1e-9
    budget_ok = 8 * len(data) <= rate * len(x) / FS + 1e-9
    acct_ok = st["header_bits"] + st["frames_bits"] + st["pad_bits"] == 8 * len(data)
    return report(same and budget_ok and acct_ok and len(y) == len(x),
                  f"{name:10s} {model:11s} {rate/1000:5.0f} kb/s: decoder == encoder reconstruction, "
                  f"{len(data)} bytes <= budget, bit accounting closes")


def test_fresh_session(x, rate, model, block_switching=False):
    data, _ = encode(x, rate, model, block_switching=block_switching)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pkg = td / "codec"
        pkg.mkdir()
        for m in DECODER_MODULES:
            shutil.copy(ROOT / "codec" / m, pkg / m)
        (td / "in.pacc").write_bytes(data)
        r = subprocess.run([sys.executable, "-E", "-s", "-m", "codec.decoder", "in.pacc", "out.wav"],
                           cwd=td, capture_output=True, text=True)
        ok = r.returncode == 0 and (td / "out.wav").exists()
        if ok:
            y, fs = read_wav(td / "out.wav")
            y_ref, _ = decode(data)
            ok = fs == FS and len(y) == len(x) and np.max(np.abs(y[:, 0] - np.clip(np.round(y_ref * 32768), -32768, 32767) / 32768)) < 1e-12
        files = sorted(p.name for p in pkg.iterdir() if p.suffix == ".py")
    kind = "block-switching (v2)" if block_switching else "long-window (v1)"
    return report(ok, f"fresh-process decode of a {kind} file with only {files} + the .pacc file "
                      f"(no encoder, PAM or original audio){'' if ok else ': ' + r.stderr[-300:]}")


def run():
    res = [test_bitio()]
    rng = np.random.default_rng(3)
    odd = rng.standard_normal(30001) * 0.05  # arbitrary length, not a multiple of 1024
    res.append(test_codec_consistency(odd, "noise", 64000, "baseline"))
    prepared = ROOT / "data" / "prepared"
    for name in ["castanets", "speech"]:
        x, _ = read_wav(prepared / f"{name}.wav")
        for rate in (64000, 128000):
            res.append(test_codec_consistency(x[:, 0], name, rate, "baseline"))
    res.append(test_codec_consistency(odd, "noise", 64000, "improved", block_switching=True))
    for name in ["castanets", "speech"]:
        x, _ = read_wav(prepared / f"{name}.wav")
        res.append(test_codec_consistency(x[:, 0], name, 64000, "improved", block_switching=True))
    x, _ = read_wav(prepared / "speech.wav")
    res.append(test_fresh_session(x[:96000, 0], 96000, "baseline"))
    x, _ = read_wav(prepared / "castanets.wav")
    res.append(test_fresh_session(x[:96000, 0], 96000, "improved", block_switching=True))
    print(f"{sum(res)}/{len(res)} codec tests passed")
    return all(res)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
