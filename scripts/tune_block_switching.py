"""Choose the transient detector's threshold on the development excerpts only.

The detector (codec/blockswitch.py) marks a 2.7 ms block as an attack when its
high-passed energy is ATTACK_RATIO_DB above the mean of the previous 21 ms.
Too low: stationary passages get short windows, which code less efficiently.
Too high: attacks stay in long windows and pre-echo returns.

Rule fixed before running: try 6, 8, 10, 13, 16 and 20 dB on castanets and
speech (the development excerpts; harpsichord stays held out), with both PAMs
at 64, 96 and 128 kb/s, and keep the threshold with the highest mean PEAQ ODG.
The chosen value is then written into blockswitch.py by hand and frozen.

Result (results/block_switching_tuning.csv): mean ODG -0.725 / -0.742 / -0.759 /
-0.784 / -0.942 / -0.980 for 6 / 8 / 10 / 13 / 16 / 20 dB, so 6 dB was frozen.
Castanets is insensitive below 16 dB (its clicks trigger the detector at any
of these thresholds); speech gains from the lower thresholds (79 short frames
of 470 at 6 dB vs 36 at 10 dB). 6 dB is the edge of the grid; lower values
were not tried, so as not to change the rule after seeing the results.

Usage:  python3 scripts/tune_block_switching.py      -> results/block_switching_tuning.csv
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
from codec.decoder import decode  # noqa: E402
from codec.encoder import get_pam, encode  # noqa: E402
from codec.wavio import read_wav  # noqa: E402
from evaluation.metrics import all_metrics  # noqa: E402
from scripts.run_peaq import peaq  # noqa: E402

RATIOS_DB = [6, 8, 10, 13, 16, 20]
DEV = ["castanets", "speech"]
MODELS = ["baseline", "improved"]
RATES = [64, 96, 128]


def to16(y):
    return np.clip(np.round(y * 32768), -32768, 32767) / 32768


def main():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in DEV:
            x, _ = read_wav(ROOT / "data" / "prepared" / f"{name}.wav")
            x = x[:, 0]
            for ratio in RATIOS_DB:
                seq = bs.sequence_from_short_frames(bs.short_frames_for(x, ratio))
                n_short = int(np.sum(seq == bs.EIGHT_SHORT))
                for model in MODELS:
                    pam_out = get_pam(model).analyze(x, seq=seq)
                    for rate in RATES:
                        data, _ = encode(x, rate * 1000, model, pam_out=pam_out,
                                         block_switching=True, seq=seq)
                        y = to16(decode(data)[0])
                        m = all_metrics(x, y)
                        odg = peaq(x, y, tmp)["ODG"]
                        rows.append(dict(excerpt=name, ratio_db=ratio, short_frames=n_short,
                                         model=model, rate=rate, odg=round(odg, 3),
                                         nmr_db=round(m["total_nmr_db"], 2),
                                         pre_echo_db=round(m["mean_pre_echo_to_signal_db"], 2)))
                        print(f"{name:10s} {ratio:3d} dB  short {n_short:3d}  {model:9s} {rate:3d}k  "
                              f"ODG {odg:6.3f}  NMR {m['total_nmr_db']:6.2f}  "
                              f"pre-echo {m['mean_pre_echo_to_signal_db']:6.2f}", flush=True)
    with open(ROOT / "results" / "block_switching_tuning.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\nmean over dev excerpts, PAMs and rates:")
    for ratio in RATIOS_DB:
        r = [row for row in rows if row["ratio_db"] == ratio]
        print(f"  {ratio:3d} dB: ODG {np.mean([q['odg'] for q in r]):6.3f}  "
              f"NMR {np.mean([q['nmr_db'] for q in r]):6.2f}  "
              f"pre-echo {np.mean([q['pre_echo_db'] for q in r]):6.2f}")


if __name__ == "__main__":
    main()
