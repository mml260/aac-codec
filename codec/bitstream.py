"""Binary file format (shared by encoder and decoder).

File  = header (24 bytes) + frames (bit-packed, back to back) + zero pad to a byte.

Header, big-endian:
  magic      4s   b"PACC"
  version    u8   1 = long windows only (the required codec), 2 = block switching
  fs         u32  48000
  hop        u16  1024   (MDCT coefficients per frame; window = 2*hop)
  length     u32  number of original samples (used to trim the decoder output)
  n_frames   u32  number of MDCT frames = ceil(length/hop) + 1
  target     u32  target bitrate in b/s (informational only)
  model      u8   0 = baseline PAM, 1 = improved PAM, 2 = control (informational only)

Frame (variable length, self-delimiting):
  max_band   6 bits     number of bands up to and including the last active band
  for b < max_band:
    active   1 bit
    if active:
      d_sf   seg0       scalefactor delta vs previous active band (first: vs SF_INIT)
      d_k    seg0       Rice parameter delta vs previous active band (first: vs K_INIT)
      q[...]  Rice(k)   BAND_WIDTH[b] signed integers; sign bit after each nonzero
  fill       1 bit
  if fill:
    n_fill   eg0, followed by n_fill zero bits (reservoir overflow stuffing)

Dequantisation: X[line] = q[line] * 2 ** (sf_band / 4)   (1.5 dB scalefactor steps)
Lines in inactive bands and bands >= max_band are zero.

Version 2 (block switching, see blockswitch.py) adds to the front of every frame
  window_sequence  2 bits   0 ONLY_LONG, 1 LONG_START, 2 EIGHT_SHORT, 3 LONG_STOP
  if EIGHT_SHORT:
    grouping       7 bits   bit i = 1: short window i+1 shares the group of window i
and an EIGHT_SHORT frame has one section per group instead of the single long
section: max_band (4 bits) + the active flags / data of the group's 14 short
bands, each band's integers in AAC's interleaved order (all windows of the
group, band by band). Scalefactor and Rice deltas continue from section to
section. Long-type frames are coded exactly as in version 1.
"""
import struct

import numpy as np

from . import FS, M
from . import blockswitch as bs
from .bitio import eg0_len, seg0_len

MAGIC = b"PACC"
VERSION = 1             # long windows only (the required codec)
VERSION_BS = 2          # block switching
SEQ_BITS = 2
HEADER_FMT = ">4sBIHIIIB"
HEADER_BYTES = struct.calcsize(HEADER_FMT)  # 24
HEADER_BITS = 8 * HEADER_BYTES
SF_INIT = 0
K_INIT = 2
MAX_BAND_BITS = 6
MODEL_IDS = {"baseline": 0, "improved": 1, "control": 2}


def step_size(sf):
    return 2.0 ** (np.asarray(sf, dtype=np.float64) / 4.0)


# ------------------------------------------------------------------ header
def pack_header(length, n_frames, target_bps, model, version=VERSION):
    return struct.pack(HEADER_FMT, MAGIC, version, FS, M, length, n_frames,
                       int(target_bps), MODEL_IDS.get(model, 255))


def unpack_header(data):
    magic, ver, fs, hop, length, n_frames, target, model = struct.unpack(
        HEADER_FMT, data[:HEADER_BYTES])
    if magic != MAGIC:
        raise ValueError("not a PACC file")
    if ver not in (VERSION, VERSION_BS):
        raise ValueError(f"unsupported version {ver}")
    return dict(version=ver, fs=fs, hop=hop, length=length, n_frames=n_frames,
                target_bps=target, model=model)


# ------------------------------------------------------------------ frame cost
def frame_header_bits(seq):
    """Version-2 window fields of one frame (0 for version 1: pass seq=None)."""
    if seq is None:
        return 0
    return SEQ_BITS + (bs.GROUPING_BITS if seq == bs.EIGHT_SHORT else 0)


def _section_max_bands(active, layout):
    out = []
    for b0, nb, _ in layout.sections:
        idx = np.flatnonzero(active[b0:b0 + nb])
        out.append(int(idx[-1]) + 1 if len(idx) else 0)
    return out


def side_info_bits(active, sf, k, layout=bs.LONG_LAYOUT):
    """Exact side-information bits of one frame (everything except Rice payload and
    the version-2 window fields)."""
    bits = 1  # fill flag
    for (b0, nb, mbits), mb in zip(layout.sections, _section_max_bands(active, layout)):
        bits += mbits + mb          # max_band field + active flags
    idx = np.flatnonzero(active)
    if len(idx):
        sfa = sf[idx]
        ka = k[idx]
        dsf = np.diff(np.concatenate([[SF_INIT], sfa]))
        dk = np.diff(np.concatenate([[K_INIT], ka]))
        bits += int(np.sum(seg0_len(dsf)) + np.sum(seg0_len(dk)))
    return int(bits)


def fill_bits(n_fill):
    """Extra bits used when n_fill stuffing bits are inserted (excluding the fill flag)."""
    return 0 if n_fill <= 0 else int(eg0_len(n_fill)) + int(n_fill)


# ------------------------------------------------------------------ frame write/read
def write_frame(bw, active, sf, k, q, n_fill=0, layout=bs.LONG_LAYOUT, seq=None, groups=None):
    """Write one frame. seq/groups only for version 2 (q in the layout's coded order)."""
    if seq is not None:
        bw.fixed(int(seq), SEQ_BITS)
        if seq == bs.EIGHT_SHORT:
            bw.fixed(bs.grouping_to_bits(groups), bs.GROUPING_BITS)
    prev_sf, prev_k = SF_INIT, K_INIT
    off = layout.offsets
    for (b0, nb, mbits), mb in zip(layout.sections, _section_max_bands(active, layout)):
        bw.fixed(mb, mbits)
        for b in range(b0, b0 + mb):
            if active[b]:
                bw.fixed(1, 1)
                bw.seg0(int(sf[b]) - prev_sf)
                bw.seg0(int(k[b]) - prev_k)
                prev_sf, prev_k = int(sf[b]), int(k[b])
                bw.rice_block(q[off[b]:off[b + 1]], int(k[b]))
            else:
                bw.fixed(0, 1)
    if n_fill > 0:
        bw.fixed(1, 1)
        bw.eg0(n_fill)
        bw.raw_bits(n_fill, 0)
    else:
        bw.fixed(0, 1)


def read_frame(br, version=VERSION):
    """Parse one frame. Returns (dequantised coefficients in natural order, window sequence)."""
    seq, layout = bs.ONLY_LONG, bs.LONG_LAYOUT
    if version == VERSION_BS:
        seq = br.fixed(SEQ_BITS)
        if seq == bs.EIGHT_SHORT:
            layout = bs.short_layout(bs.bits_to_grouping(br.fixed(bs.GROUPING_BITS)))
    X = np.zeros(M)
    off = layout.offsets
    prev_sf, prev_k = SF_INIT, K_INIT
    for b0, nb, mbits in layout.sections:
        max_band = br.fixed(mbits)
        if max_band > nb:
            raise ValueError("corrupt frame: max_band out of range")
        for b in range(b0, b0 + max_band):
            if br.fixed(1):
                sf = prev_sf + br.seg0()
                k = prev_k + br.seg0()
                prev_sf, prev_k = sf, k
                q = br.rice_block(int(off[b + 1] - off[b]), k)
                X[off[b]:off[b + 1]] = q * step_size(sf)
    if br.fixed(1):
        br.skip(br.eg0())
    return layout.to_natural(X), seq
