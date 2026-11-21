"""Where does the improved PAM's gain come from once block switching is on?

Variants of the improved PAM with block switching, on the development excerpts
at 64 and 96 kb/s:
  full                    the improved PAM as used in results/
  tighten only            long-frame factor F limited to <= 1 (pre-echo control only)
  relax only              long-frame factor F limited to >= 1 (post-masking only)
  short = own energy      short windows use their own energy (as the baseline),
                          long frames keep the full factor F

Usage:  python3 scripts/ablation_block_switching.py  -> results/ablation_block_switching.csv
(needs the PEAQ build from scripts/run_peaq.py)
"""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec import blockswitch as bs  # noqa: E402
from codec import pam_temporal as pt  # noqa: E402
from codec.decoder import decode  # noqa: E402
from codec.encoder import encode  # noqa: E402
from codec.wavio import read_wav  # noqa: E402
from evaluation.metrics import all_metrics  # noqa: E402
from scripts.run_peaq import peaq  # noqa: E402

VARIANTS = {
    "full": (pt.F_MIN_DB, pt.F_MAX_DB, True),
    "tighten only (F<=1)": (pt.F_MIN_DB, 0.0, True),
    "relax only (F>=1)": (0.0, pt.F_MAX_DB, True),
    "short windows = own energy": (pt.F_MIN_DB, pt.F_MAX_DB, False),
}


def main():
    rows = []
    defaults = (pt.F_MIN_DB, pt.F_MAX_DB)
    with tempfile.TemporaryDirectory() as tmp:
        for name in ["castanets", "speech"]:
            x, _ = read_wav(ROOT / "data" / "prepared" / f"{name}.wav")
            x = x[:, 0]
            seq = bs.choose_sequence(x)
            for label, (fmin, fmax, short_env) in VARIANTS.items():
                pt.F_MIN_DB, pt.F_MAX_DB = fmin, fmax
                pam = pt.TemporalPAM()
                if not short_env:
                    pam.short_masker_energy = lambda E: E
                out = pam.analyze(x, seq=seq)
                for rate in (64, 96):
                    data, _ = encode(x, rate * 1000, "improved", pam_out=out, block_switching=True, seq=seq)
                    y = np.clip(np.round(decode(data)[0] * 32768), -32768, 32767) / 32768
                    m = all_metrics(x, y)
                    odg = peaq(x, y, tmp)["ODG"]
                    rows.append(dict(excerpt=name, rate=rate, variant=label, odg=round(odg, 3),
                                     nmr_db=round(m["total_nmr_db"], 2)))
                    print(f"{name:10s} {rate:3d}k  {label:28s} ODG {odg:6.2f}  NMR {m['total_nmr_db']:6.2f}")
            pt.F_MIN_DB, pt.F_MAX_DB = defaults
    with open(ROOT / "results" / "ablation_block_switching.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
