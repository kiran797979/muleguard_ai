"""
Does the temporal localiser actually find a window it was not told about?

Builds a synthetic ledger where each mule account has a *known* laundering
period embedded in an otherwise ordinary multi-year transaction history, runs
`temporal.detect_windows` with no knowledge of those periods, and scores the
result with the same temporal IoU the challenge uses.

Like the graph demonstration, this proves the code path works and says nothing
about the supplied dataset, which has no transactions in it at all. Outputs are
namespaced under reports/demo_temporal/.

Run:  python src/temporal_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import temporal as T
from utils import log, save_json

OUT_DIR = C.REPORTS_DIR / "demo_temporal"

N_NORMAL = 300
N_MULE = 120
HISTORY_DAYS = 900
WINDOW_DAYS = (25, 90)
SEED = C.RANDOM_STATE
START = pd.Timestamp("2022-01-01")


def _ordinary_days(rng, n_days, start_day):
    """A salary-and-bills rhythm: a monthly credit, scattered small debits."""
    rows = []
    for d in range(n_days):
        day = START + pd.Timedelta(days=start_day + d)
        if day.day in (1, 2):
            rows.append((day, rng.normal(45_000, 4_000), "C"))
        for _ in range(rng.poisson(1.2)):
            rows.append((day, abs(rng.normal(1_400, 900)) + 50, "D"))
    return rows


def _laundering_days(rng, n_days, start_day):
    """Pass-through: money lands and leaves the same day, often just under a line."""
    rows = []
    for d in range(n_days):
        day = START + pd.Timedelta(days=start_day + d)
        for _ in range(rng.integers(2, 6)):
            if rng.random() < 0.55:
                amt = rng.uniform(0.90, 0.995) * 50_000        # structuring
            elif rng.random() < 0.4:
                amt = float(rng.choice(T.ROUND_AMOUNTS))       # round amounts
            else:
                amt = abs(rng.normal(80_000, 25_000)) + 5_000
            rows.append((day, amt, "C"))
            # Forwarded within the same day, keeping almost nothing.
            rows.append((day, amt * rng.uniform(0.94, 0.995), "D"))
    return rows


def build_ledger(rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    txns, truth = [], []

    for i in range(N_NORMAL):
        acct = f"ACCT_N{i:05d}"
        for day, amt, dirn in _ordinary_days(rng, HISTORY_DAYS, 0):
            txns.append((acct, day, amt, dirn))
        truth.append({"account_id": acct, "is_mule": 0,
                      "suspicious_start": pd.NaT, "suspicious_end": pd.NaT})

    for i in range(N_MULE):
        acct = f"ACCT_M{i:05d}"
        wlen = int(rng.integers(*WINDOW_DAYS))
        wstart = int(rng.integers(120, HISTORY_DAYS - wlen - 60))

        # Ordinary life before and after the episode.
        for day, amt, dirn in _ordinary_days(rng, wstart, 0):
            txns.append((acct, day, amt, dirn))
        for day, amt, dirn in _laundering_days(rng, wlen, wstart):
            txns.append((acct, day, amt, dirn))
        after = HISTORY_DAYS - wstart - wlen
        for day, amt, dirn in _ordinary_days(rng, after, wstart + wlen):
            txns.append((acct, day, amt, dirn))

        truth.append({
            "account_id": acct, "is_mule": 1,
            "suspicious_start": START + pd.Timedelta(days=wstart),
            "suspicious_end": START + pd.Timedelta(days=wstart + wlen - 1),
        })

    tx = pd.DataFrame(txns, columns=["account_id", "timestamp", "amount", "direction"])
    return tx, pd.DataFrame(truth)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    log("=" * 68)
    log("TEMPORAL LOCALISATION DEMONSTRATION — SYNTHETIC DATA, NOT A RESULT")
    log("The supplied account-level file has no transactions, so this cannot")
    log("run on it. This shows the same code working when a ledger exists.")
    log("=" * 68)

    txns, truth = build_ledger(rng)
    log(f"Ledger: {txns['account_id'].nunique():,} accounts, {len(txns):,} transactions "
        f"over {HISTORY_DAYS} days ({N_MULE} with a planted window)")

    windows = T.detect_windows(txns)
    got = windows["suspicious_start"].notna().sum()
    log(f"Localised a window for {got:,} of {len(windows):,} accounts")

    mules = truth[truth.is_mule == 1]
    overall = T.score_windows(windows, mules)
    log("")
    log("  Scored against the planted windows (mule accounts only):")
    for k, v in overall.items():
        log(f"    {k:22s} {v}")

    # How does it do on accounts that never laundered? The correct answer there
    # is "no window", and predicting one anyway would be a false localisation.
    normals = truth[truth.is_mule == 0]
    n_pred = windows.set_index("account_id").reindex(normals.account_id)
    false_windows = int(n_pred["suspicious_start"].notna().sum())
    log("")
    log(f"  Windows predicted on non-mule accounts: {false_windows} of {len(normals)} "
        f"({false_windows / max(len(normals),1):.1%})")
    log("  (these are suppressed in the submission by the is_mule threshold)")

    # And the length error, which is what usually costs IoU.
    j = windows.merge(mules, on="account_id", suffixes=("_pred", "_true"))
    j = j.dropna(subset=["suspicious_start_pred"])
    pred_len = (j.suspicious_end_pred - j.suspicious_start_pred).dt.days + 1
    true_len = (j.suspicious_end_true - j.suspicious_start_true).dt.days + 1
    log("")
    log(f"  Median true window {int(true_len.median())} days, "
        f"median predicted {int(pred_len.median())} days")

    save_json({
        "WARNING": "SYNTHETIC CAPABILITY DEMONSTRATION. The planted windows were "
                   "created by this script and then found by this script. It says "
                   "nothing about the supplied dataset, which contains no "
                   "transactions to localise.",
        "ledger": {"accounts": int(txns["account_id"].nunique()),
                   "transactions": int(len(txns)),
                   "history_days": HISTORY_DAYS,
                   "mule_accounts": N_MULE},
        "scored_on_mules": overall,
        "false_windows_on_normals": false_windows,
        "normals": int(len(normals)),
        "median_true_window_days": int(true_len.median()),
        "median_pred_window_days": int(pred_len.median()),
    }, OUT_DIR / "temporal_demo_report.json")
    windows.to_csv(OUT_DIR / "windows.csv", index=False)
    log(f"Wrote {OUT_DIR / 'temporal_demo_report.json'}")


if __name__ == "__main__":
    main()
