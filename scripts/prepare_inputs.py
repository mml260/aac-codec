"""Prepare the three common SQAM excerpts: 48 kHz, mono, first 10 s, 16-bit.

  * channels are averaged ((L+R)/2), no other gain is applied;
  * 44.1 kHz -> 48 kHz with a polyphase Kaiser-windowed sinc resampler
    (codec/resample.py, numerically identical to scipy.signal.resample_poly);
  * exactly 480000 output samples (10.000 s) are kept;
  * a SHA-256 of each prepared file's 16-bit sample data is written to
    data/prepared/manifest.json so every experiment can verify it used
    identical inputs (hashing the samples, not the file bytes, keeps the check
    valid if a tool adds metadata chunks to the WAV).

Usage:  python scripts/prepare_inputs.py "<folder with the SQAM wavs>"
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec import FS  # noqa: E402
from codec.resample import resample_poly  # noqa: E402
from codec.wavio import read_wav, write_wav  # noqa: E402

EXCERPTS = {
    "castanets": "27_castanets_single_tone_rhythm_stereo.wav",
    "harpsichord": "40_harpsichord_arpegio_melodious_phrase_stereo.wav",
    "speech": "49_female_speech_english_mono.wav",
}
DURATION_S = 10.0


def pcm_sha256(path):
    x, _ = read_wav(path)
    return hashlib.sha256(np.round(x * 32768).astype("<i2").tobytes()).hexdigest()


def main(src_dir):
    src_dir = Path(src_dir)
    out_dir = ROOT / "data" / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    n_out = int(round(DURATION_S * FS))
    for name, fname in EXCERPTS.items():
        x, fs_in = read_wav(src_dir / fname)
        mono = x.mean(axis=1)
        # Resample a little more than 10 s so the end of the excerpt is not
        # affected by the filter's edge, then cut exactly 480000 samples.
        n_in = int(np.ceil((DURATION_S + 0.1) * fs_in))
        seg = mono[:n_in]
        if fs_in != FS:
            y = resample_poly(seg, FS, fs_in)
        else:
            y = seg.copy()
        y = y[:n_out]
        out = out_dir / f"{name}.wav"
        clipped = write_wav(out, y, FS)
        digest = pcm_sha256(out)
        peak = float(np.max(np.abs(y)))
        manifest[name] = {
            "source": fname,
            "source_fs": fs_in,
            "source_channels": x.shape[1],
            "samples": n_out,
            "fs": FS,
            "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
            "clipped_samples": clipped,
            "sha256": digest,
        }
        print(f"{name:12s} {fname}  {fs_in} Hz x{x.shape[1]} -> {out.name}  "
              f"peak {manifest[name]['peak_dbfs']} dBFS  sha256 {digest[:12]}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "audio excerpts")
