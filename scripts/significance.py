"""Paired per-attack significance tests on pre-echo, from the decoded files.

Comparisons (A vs B; "A wins" = less pre-echo before that attack):
  improved vs baseline, control vs baseline, improved vs control,
  and with block switching: baseline_bs vs baseline, improved_bs vs baseline_bs
for every excerpt x rate. Writes results/significance.csv.

Usage:  python scripts/significance.py   (after scripts/run_all.py)
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from codec.wavio import read_wav  # noqa: E402
from evaluation.metrics import detect_attacks, pre_echo  # noqa: E402
from evaluation.significance import holm, sign_test, wilcoxon  # noqa: E402

EXCERPTS = ["castanets", "speech", "harpsichord"]
RATES = [64, 96, 128]
COMPARISONS = [("improved", "baseline"), ("control", "baseline"), ("improved", "control"),
               ("baseline_bs", "baseline"), ("improved_bs", "baseline_bs")]
MODELS = sorted({m for pair in COMPARISONS for m in pair})


def main():
    res = ROOT / "results"
    per = {}
    for n in EXCERPTS:
        x, _ = read_wav(res / n / f"{n}__original.wav")
        x = x[:, 0]
        att = detect_attacks(x)
        for r in RATES:
            for m in MODELS:
                p = next((res / n).glob(f"{n}__{m}__target{r:03d}k__actual*.wav"), None)
                if p is None:
                    continue
                y, _ = read_wav(p)
                per[(n, m, r)] = pre_echo(x, y[:, 0], att)["per_attack_db"]
    rows = []
    for a, b in COMPARISONS:
        block = []
        for n in EXCERPTS:
            for r in RATES:
                if (n, a, r) not in per or (n, b, r) not in per:
                    continue
                d = per[(n, a, r)] - per[(n, b, r)]
                s = sign_test(d)
                w = wilcoxon(d)
                block.append(dict(comparison=f"{a} vs {b}", excerpt=n, rate_kbps=r, attacks=len(d),
                                  a_wins=s["wins"], median_diff_db=round(float(np.median(d)), 2),
                                  mean_diff_db=round(float(np.mean(d)), 2),
                                  sign_p=s["p"], wilcoxon_p=w["p"]))
        if block:
            adj = holm([row["wilcoxon_p"] for row in block])
            for row, ap in zip(block, adj):
                row["wilcoxon_p_holm"] = ap
            rows += block
    with open(res / "significance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in row.items()})
    for row in rows:
        star = "*" if row["wilcoxon_p_holm"] < 0.05 else " "
        print(f"{row['comparison']:26s} {row['excerpt']:11s} {row['rate_kbps']:3d}k  "
              f"wins {row['a_wins']:2d}/{row['attacks']:2d}  median {row['median_diff_db']:+6.2f} dB  "
              f"sign p={row['sign_p']:.4f}  Wilcoxon p={row['wilcoxon_p']:.4f}  Holm p={row['wilcoxon_p_holm']:.4f} {star}")


if __name__ == "__main__":
    main()
