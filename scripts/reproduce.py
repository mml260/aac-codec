"""Reproduce every result from the three SQAM source files, in order:

  1. prepare inputs (48 kHz mono, first 10 s)       -> data/prepared/
  2. MDCT perfect-reconstruction tests               (no quantization)
  3. codec tests (bit accounting, fresh-process decode)
  4. encode + decode all conditions + metrics        -> results/
     (18 required: 3 excerpts x baseline/improved x 3 rates, plus 9 control runs
      and 18 block-switching runs)
  5. paired significance tests on per-attack pre-echo -> results/significance.csv
  6. PEAQ, only if the GstPEAQ core is present        -> results/peaq.csv
     (see scripts/run_peaq.py; skipped otherwise)
  7. figures                                         -> figures/

Usage (from the project folder):  python3 scripts/reproduce.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    ("Prepare inputs", ["scripts/prepare_inputs.py"]),
    ("MDCT reconstruction tests", ["tests/test_mdct.py"]),
    ("Block-switching transform tests", ["tests/test_blockswitch.py"]),
    ("Codec tests", ["tests/test_codec.py"]),
    ("Encode/decode 45 conditions", ["scripts/run_all.py", "--models",
                                     "baseline,improved,control,baseline_bs,improved_bs"]),
    ("Significance tests", ["scripts/significance.py"]),
    ("PEAQ", ["scripts/run_peaq.py"]),
    ("Figures", ["scripts/make_figures.py"]),
]
OPTIONAL = {"PEAQ": lambda: (ROOT / "tools" / "peaq" / "peaq_basic").exists()
            or (ROOT / "tools" / "peaq" / "gstpeaq" / "src").exists()}

if __name__ == "__main__":
    for title, args in STEPS:
        if title in OPTIONAL and not OPTIONAL[title]():
            print(f"\n=== {title}: skipped (optional tool not installed) ===", flush=True)
            continue
        t = time.time()
        print(f"\n=== {title} ===", flush=True)
        r = subprocess.run([sys.executable] + args, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"step '{title}' failed")
        print(f"--- {title}: {time.time() - t:.1f} s", flush=True)
    print("\nDone. See results/summary.csv, results/<excerpt>/ and figures/.")
