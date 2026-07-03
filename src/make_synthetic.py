"""
Generate a SYNTHETIC dataset shaped like the hackathon data, for smoke-testing
the pipeline end-to-end without the real DataSet.csv.

It deliberately includes:
  * ~0.9% positive prevalence (rare mules),
  * a planted leak column named F3912 (near-perfect correlation with target),
  * many redundant / mostly-empty columns (to exercise cleaning),
  * a real (but weak) signal in a handful of columns.

This is NOT the real data and produces NO submittable numbers — it only proves
the code runs and behaves sensibly. Delete data/DataSet.csv afterwards and drop
in the real file to get honest results.

Run:  python src/make_synthetic.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

RNG = np.random.default_rng(C.RANDOM_STATE)

N = 3000          # smaller than 9082 for a fast smoke test
N_FEAT = 300      # smaller than 3924 but same character
PREVALENCE = 0.01


def main() -> None:
    n_pos = max(int(N * PREVALENCE), 20)
    y = np.zeros(N, dtype=int)
    y[RNG.choice(N, size=n_pos, replace=False)] = 1

    cols = {}
    # Genuine (noisy) signal in ~6 columns.
    for i in range(6):
        base = RNG.normal(0, 1, N)
        base[y == 1] += RNG.normal(1.5, 0.8, n_pos)  # mules shifted
        cols[f"F{i+1}"] = base

    # Redundant rolling-window copies of the signal (collinear).
    for i in range(6):
        cols[f"F{i+1}_w7"] = cols[f"F{i+1}"] + RNG.normal(0, 0.001, N)

    # Noise columns.
    for i in range(N_FEAT):
        cols[f"Fn{i}"] = RNG.normal(0, 1, N)

    # Mostly-empty columns (to be dropped by cleaning).
    for i in range(40):
        c = RNG.normal(0, 1, N)
        mask = RNG.random(N) < 0.7   # 70% missing
        c[mask] = np.nan
        cols[f"Fsparse{i}"] = c

    # Planted LEAK column F3912 (near-perfect correlation with target).
    leak = y.astype(float) + RNG.normal(0, 0.02, N)
    cols[C.LEAK_COL] = leak

    df = pd.DataFrame(cols)
    df[C.TARGET_COL] = y

    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RAW_CSV, index=False)
    print(f"[synthetic] wrote {C.RAW_CSV}  shape={df.shape}  positives={n_pos}")
    print(f"[synthetic] planted leak col={C.LEAK_COL}  "
          f"corr={np.corrcoef(leak, y)[0,1]:.3f}")


if __name__ == "__main__":
    main()
