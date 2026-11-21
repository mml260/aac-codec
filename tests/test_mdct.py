"""Perfect-reconstruction tests for the MDCT (no quantization).

Run:  python tests/test_mdct.py      (also produces figures/mdct_reconstruction.png
                                      when run via scripts/make_figures.py)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec import M, N_WIN  # noqa: E402
from codec import mdct  # noqa: E402
from codec.wavio import read_wav  # noqa: E402

TOL = 1e-12


def roundtrip(x):
    X = mdct.mdct(x)
    y = mdct.imdct(X, len(x))
    return X, y


def check(name, x):
    X, y = roundtrip(x)
    err = np.max(np.abs(y - x)) if len(x) else 0.0
    # Energy preservation (orthonormal TDAC): sum of squared coefficients == signal energy
    e_ratio = np.sum(X**2) / max(np.sum(x**2), 1e-300)
    ok = err < TOL and len(y) == len(x)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:42s} L={len(x):7d} frames={X.shape[0]:4d} "
          f"max|err|={err:.2e}  energy ratio={e_ratio:.12f}")
    return ok


def run():
    rng = np.random.default_rng(1)
    results = []
    print("Princen-Bradley check: max |w[n]^2 + w[n+M]^2 - 1| =",
          f"{np.max(np.abs(mdct.window()[:M]**2 + mdct.window()[M:]**2 - 1)):.2e}")
    for L in [1, 2, 7, 1023, 1024, 1025, 2047, 2048, 2049, 4096, 48000 + 7, 480000, 480001]:
        results.append(check(f"white noise, arbitrary length", rng.standard_normal(L) * 0.3))
    # Boundary stress: impulses at the very first and last samples, a full-scale step,
    # DC and a sine that starts/ends abruptly.
    L = 5000
    x = np.zeros(L); x[0] = 1.0; x[-1] = -1.0
    results.append(check("impulses at first/last sample", x))
    results.append(check("DC (full scale)", np.ones(3333)))
    x = np.zeros(4000); x[1500:] = 1.0
    results.append(check("step", x))
    t = np.arange(10007) / 48000
    results.append(check("1 kHz sine, abrupt edges", 0.99 * np.sin(2 * np.pi * 1000 * t)))
    prepared = ROOT / "data" / "prepared"
    for name in ["castanets", "harpsichord", "speech"]:
        p = prepared / f"{name}.wav"
        if p.exists():
            x, _ = read_wav(p)
            results.append(check(f"prepared excerpt: {name}", x[:, 0]))
            results.append(check(f"prepared excerpt: {name} (odd length)", x[:123457, 0]))
    n_pass = sum(results)
    print(f"{n_pass}/{len(results)} reconstruction tests passed (tolerance {TOL:g})")
    return n_pass == len(results)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
