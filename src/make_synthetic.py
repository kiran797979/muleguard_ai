"""
Generate a SYNTHETIC dataset for smoke-testing the pipeline end to end.

Two modes, and the second is the point:

  --schema psb    (default)  Mimics the hackathon file: F-coded columns, target
                             F3924, a planted F3912 leak. Exercises the plumbing
                             on the shape the project was built for.

  --schema alien             A DELIBERATELY DIFFERENT dataset. Readable column
                             names, a target called `is_mule`, an account-id
                             column, a `data_month` partition column, a
                             post-outcome `case_resolution` leak, and NO data
                             dictionary at all.

The alien mode is the regression test for dataset-independence. Nothing in it
shares a single column name with the hackathon file, so if the pipeline finds
the target, strips the identifier, catches the partition column and removes the
post-outcome leak on that file, it is doing so from structure rather than from
memorised names.

Neither mode produces submittable numbers. They prove the code runs and behaves
sensibly, nothing more.

Run:  python src/make_synthetic.py --schema alien --out data/alien.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config as C

RNG = np.random.default_rng(C.RANDOM_STATE)

N = 3000          # smaller than 9082 for a fast smoke test
N_FEAT = 300      # smaller than 3924 but same character
PREVALENCE = 0.01


def _labels(n: int, prevalence: float) -> tuple[np.ndarray, int]:
    n_pos = max(int(n * prevalence), 20)
    y = np.zeros(n, dtype=int)
    y[RNG.choice(n, size=n_pos, replace=False)] = 1
    return y, n_pos


# --------------------------------------------------------------------------
# Mode 1 — mimic the hackathon schema
# --------------------------------------------------------------------------
def build_psb() -> pd.DataFrame:
    y, n_pos = _labels(N, PREVALENCE)
    cols: dict[str, np.ndarray] = {}

    # Genuine (noisy) signal in ~6 columns.
    for i in range(6):
        base = RNG.normal(0, 1, N)
        base[y == 1] += RNG.normal(1.5, 0.8, n_pos)
        cols[f"F{i+1}"] = base

    # Redundant rolling-window copies of the signal (collinear).
    for i in range(6):
        cols[f"F{i+1}_w7"] = cols[f"F{i+1}"] + RNG.normal(0, 0.001, N)

    for i in range(N_FEAT):
        cols[f"Fn{i}"] = RNG.normal(0, 1, N)

    for i in range(40):
        c = RNG.normal(0, 1, N)
        c[RNG.random(N) < 0.7] = np.nan
        cols[f"Fsparse{i}"] = c

    # Planted LEAK column (near-perfect correlation with target).
    cols[C.LEAK_COL] = y.astype(float) + RNG.normal(0, 0.02, N)

    df = pd.DataFrame(cols)
    df[C.TARGET_COL_HINT] = y
    return df


# --------------------------------------------------------------------------
# Mode 2 — a schema the pipeline has never seen
# --------------------------------------------------------------------------
def build_alien() -> pd.DataFrame:
    """A dataset sharing no column name with the hackathon file.

    Everything the pipeline must discover on its own is planted here exactly
    once, so a failure is unambiguous:

      target        `is_mule`          — not F3924, not last, name-matched
      identifier    `account_number`   — unique per row; must never be a feature
      partition     `data_month`       — classes in disjoint months (the MNTH trap)
      post-outcome  `case_resolution`  — written after an analyst closes the case
      behavioural   readable names in the project's own vocabulary, differently
                    punctuated, so fuzzy resolution has to do the work
    """
    y, n_pos = _labels(N, PREVALENCE)
    d: dict[str, object] = {}

    # --- identifier: unique, useless, tempting to memorise ------------------
    d["account_number"] = [f"AC{100000 + i}" for i in range(N)]

    # --- the partition trap: positives and negatives in disjoint months -----
    months = np.array(["2025-10"] * N, dtype=object)
    pos_idx = np.where(y == 1)[0]
    months[pos_idx] = RNG.choice(["2025-09", "2025-11", "2025-12"], size=len(pos_idx))
    d["data_month"] = months

    # --- post-outcome leak: only exists after the case is closed ------------
    res = np.array(["not_investigated"] * N, dtype=object)
    res[pos_idx] = RNG.choice(["confirmed_mule", "account_closed"], size=len(pos_idx))
    d["case_resolution"] = res

    # --- behavioural columns, in the project's vocabulary but differently
    #     punctuated: the fuzzy resolver has to match these to build features.
    credit_7d = np.abs(RNG.lognormal(9, 1.2, N))
    debit_7d = credit_7d * RNG.uniform(0.2, 0.9, N)
    # Mules push nearly everything straight back out.
    debit_7d[pos_idx] = credit_7d[pos_idx] * RNG.uniform(0.93, 1.0, len(pos_idx))
    balance = np.abs(RNG.lognormal(8, 1.0, N))
    balance[pos_idx] *= 0.05          # holds almost nothing

    d["Tot Txnamt Cr L7D"] = credit_7d
    d["Tot Txnamt Db L7D"] = debit_7d
    d["tot.txnamt.l7d"] = credit_7d + debit_7d
    d["avg-bal-7days"] = balance
    d["Tot Txns L7D"] = RNG.poisson(6, N) + 1
    d["Cash Amt Db L7D"] = debit_7d * RNG.uniform(0.0, 0.3, N)
    d["Upi Amt Cr L7D"] = credit_7d * RNG.uniform(0.3, 0.9, N)

    # --- a genuine categorical: BOTH classes appear in every value, so the
    #     partition detector must NOT flag it (the false-positive guard).
    d["customer_segment"] = RNG.choice(["retail", "sme", "corporate"], size=N)

    # --- ordinary numeric noise + a little real signal ----------------------
    for i in range(6):
        col = RNG.normal(0, 1, N)
        col[y == 1] += RNG.normal(1.2, 0.9, n_pos)
        d[f"risk_indicator_{i}"] = col
    for i in range(120):
        d[f"metric_{i:03d}"] = RNG.normal(0, 1, N)
    for i in range(20):
        c = RNG.normal(0, 1, N)
        c[RNG.random(N) < 0.7] = np.nan
        d[f"sparse_metric_{i:02d}"] = c

    df = pd.DataFrame(d)
    df["is_mule"] = y
    # Target deliberately NOT last, so "last column" cannot be the thing that
    # finds it — the name pattern has to.
    df["days_since_last_txn"] = RNG.integers(0, 90, N)
    return df


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schema", choices=("psb", "alien"), default="psb")
    ap.add_argument("--out", default=None,
                    help="output path (default: data/DataSet.csv for psb, "
                         "data/alien_dataset.csv for alien)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the output file if it exists")
    args = ap.parse_args()

    if args.schema == "alien":
        df = build_alien()
        default = C.DATA_DIR / "alien_dataset.csv"
        target = "is_mule"
    else:
        df = build_psb()
        default = C.DATA_DIR / "DataSet.csv"
        target = C.TARGET_COL_HINT

    out = C.DATA_DIR / args.out if args.out and "/" not in args.out and "\\" not in args.out \
        else (C.ROOT / args.out if args.out else default)

    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Never write to C.RAW_CSV implicitly: it resolves to whichever real dataset
    # is present, so a stray default would silently destroy the hackathon file.
    if out.exists() and not args.force:
        raise SystemExit(
            f"[synthetic] {out} already exists — refusing to overwrite it.\n"
            f"            Pass --force if you really want to replace it."
        )
    df.to_csv(out, index=False)
    print(f"[synthetic] schema={args.schema}  wrote {out}")
    print(f"[synthetic] shape={df.shape}  target={target}  "
          f"positives={int(df[target].sum())}")
    if args.schema == "alien":
        print("[synthetic] planted for the pipeline to DISCOVER unaided:")
        print("            target       is_mule          (not last, not F-coded)")
        print("            identifier   account_number   (unique per row)")
        print("            partition    data_month       (classes in disjoint months)")
        print("            post-outcome case_resolution  (written after closure)")
        print(f"[synthetic] run it with:  MULEGUARD_DATA={out} python src/pipeline.py")


if __name__ == "__main__":
    main()
