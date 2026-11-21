"""Decoder: .pacc bitstream -> 48 kHz mono 16-bit WAV.

Needs nothing but the encoded file: no original audio, no PAM, no encoder
state. It imports only the shared format definition (bitstream, bitio), the
band tables and the inverse transform (mdct, plus blockswitch for version-2
files with window switching).
"""
import argparse
from pathlib import Path

import numpy as np

from . import M
from . import blockswitch as bs
from . import mdct
from .bitio import BitReader
from .bitstream import HEADER_BYTES, VERSION_BS, read_frame, unpack_header
from .wavio import write_wav


def decode(data):
    hdr = unpack_header(data)
    if hdr["hop"] != M:
        raise ValueError("unsupported frame size")
    br = BitReader(data[HEADER_BYTES:])
    F = hdr["n_frames"]
    X = np.zeros((F, M))
    seq = np.zeros(F, dtype=int)
    for f in range(F):
        X[f], seq[f] = read_frame(br, hdr["version"])
    if br.bits_left() > 7:
        raise ValueError(f"{br.bits_left()} unexpected trailing bits")
    if hdr["version"] == VERSION_BS:
        bs.check_sequence(seq)
        y = bs.synthesis(X, seq, hdr["length"])
    else:
        y = mdct.imdct(X, hdr["length"])
    hdr["window_sequence"] = seq
    return y, hdr


def main(argv=None):
    ap = argparse.ArgumentParser(description="Decode a .pacc file to WAV.")
    ap.add_argument("input")
    ap.add_argument("output")
    a = ap.parse_args(argv)
    data = Path(a.input).read_bytes()
    y, hdr = decode(data)
    clipped = write_wav(a.output, y, hdr["fs"])
    kbps = 8 * len(data) * hdr["fs"] / hdr["length"] / 1000
    print(f"{a.input} -> {a.output}: {hdr['length']} samples, {kbps:.3f} kb/s"
          + (f", {clipped} samples clipped" if clipped else ""))


if __name__ == "__main__":
    main()
