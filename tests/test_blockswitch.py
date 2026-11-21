"""Block-switching transform tests (no quantisation).

  * window shapes: Princen-Bradley across every allowed transition, and the
    START/STOP halves are time-reversed copies of the halves they overlap
  * perfect reconstruction for random legal window sequences (including all
    short, alternating, single short frames, first/last frame short) and
    arbitrary signal lengths
  * the all-long sequence reproduces the v1 MDCT exactly
  * the transform is orthonormal frame by frame (energy preserved)
  * sequence rules: illegal transitions are rejected; the encoder's sequence
    builder only produces legal ones
  * short-frame layouts: every grouping is a permutation of the 1024 lines,
    and the 7-bit grouping code round-trips

Run:  python tests/test_blockswitch.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec import M, mdct  # noqa: E402
from codec import blockswitch as bs  # noqa: E402

TOL = 1e-10


def report(ok, msg):
    print(f"  {'PASS' if ok else 'FAIL'}  {msg}")
    return ok


def test_windows():
    ok = True
    left = {s: bs.noise_shape(s)[:M] for s in range(4)}     # squared first halves
    right = {s: bs.noise_shape(s)[M:] for s in range(4)}    # squared second halves
    for prev, cur in itertools.product(range(4), range(4)):
        if bs.legal(prev, cur):
            pb = right[prev] + left[cur]
            ok &= np.max(np.abs(pb - 1)) < 1e-12
            ok &= np.allclose(right[prev], left[cur][::-1])
    return report(ok, "Princen-Bradley and time-reversal symmetry across all 8 allowed transitions")


def test_reconstruction():
    rng = np.random.default_rng(7)
    worst = 0.0
    cases = 0
    for L in (1, 777, 2048, 10000, 30001, 48000):
        F = mdct.n_frames(L)
        x = rng.standard_normal(L)
        patterns = [np.zeros(F, bool), np.ones(F, bool), np.arange(F) % 2 == 0,
                    np.arange(F) % 3 == 0, rng.random(F) < 0.3]
        if F > 2:
            p = np.zeros(F, bool)
            p[[0, F // 2, F - 1]] = True
            patterns.append(p)
        for short in patterns:
            seq = bs.sequence_from_short_frames(short)
            y = bs.synthesis(bs.analysis(x, seq), seq, L)
            worst = max(worst, np.max(np.abs(y - x)))
            cases += 1
    return report(worst < TOL, f"perfect reconstruction, {cases} sequence/length cases, max error {worst:.1e}")


def test_long_equals_v1():
    x = np.random.default_rng(1).standard_normal(20000)
    seq = np.zeros(mdct.n_frames(len(x)), int)
    d = np.max(np.abs(bs.analysis(x, seq) - mdct.mdct(x)))
    return report(d < 1e-12, f"all-long sequence == v1 MDCT (max diff {d:.1e})")


def test_orthonormal():
    rng = np.random.default_rng(2)
    ok = True
    x = rng.standard_normal(30000)
    for short in (np.zeros(mdct.n_frames(len(x)), bool), rng.random(mdct.n_frames(len(x))) < 0.5):
        seq = bs.sequence_from_short_frames(short)
        X = bs.analysis(x, seq)
        ok &= abs(np.sum(X ** 2) / np.sum(x ** 2) - 1) < 1e-10
    return report(ok, "transform preserves energy (orthonormal / tight frame) for long and mixed sequences")


def test_sequence_rules():
    ok = True
    try:
        bs.check_sequence([bs.ONLY_LONG, bs.EIGHT_SHORT])
        ok = False
    except ValueError:
        pass
    try:
        bs.check_sequence([bs.LONG_START, bs.ONLY_LONG])
        ok = False
    except ValueError:
        pass
    rng = np.random.default_rng(3)
    for _ in range(500):
        short = rng.random(rng.integers(1, 40)) < rng.random()
        seq = bs.sequence_from_short_frames(short)
        bs.check_sequence(seq)
        ok &= bool(np.all(seq[short] == bs.EIGHT_SHORT))
    return report(ok, "illegal transitions rejected; 500 random detector outputs give legal sequences "
                      "with every attack frame short")


def test_layouts():
    ok = True
    n = 0
    for bits in range(1 << bs.GROUPING_BITS):
        g = bs.bits_to_grouping(bits)
        ok &= bs.grouping_to_bits(g) == bits and sum(g) == bs.N_SUBWIN
        lay = bs.short_layout(g)
        ok &= np.array_equal(np.sort(lay.perm), np.arange(M))
        ok &= lay.offsets[-1] == M and lay.n_bands == len(g) * bs.N_BANDS_SHORT
        v = np.random.default_rng(bits).standard_normal(M)
        ok &= np.array_equal(lay.to_natural(lay.to_coded(v)), v)
        n += 1
    return report(ok, f"all {n} groupings: 7-bit code round-trips, layout is a permutation of 1024 lines")


def run():
    res = [test_windows(), test_reconstruction(), test_long_equals_v1(), test_orthonormal(),
           test_sequence_rules(), test_layouts()]
    print(f"{sum(res)}/{len(res)} block-switching tests passed")
    return all(res)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
