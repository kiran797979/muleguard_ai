"""
MuleGuard AI — shared helpers used across pipeline stages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def log(msg: str) -> None:
    """Timestamp-free structured stdout line (deterministic for logs/tests)."""
    print(f"[MuleGuard] {msg}", flush=True)


def die(msg: str, code: int = 1) -> None:
    """Print an error to stderr and exit."""
    print(f"[MuleGuard][ERROR] {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Load the raw dataset, failing loudly with guidance if it's missing."""
    if not csv_path.exists():
        die(
            f"Dataset not found at {csv_path}\n"
            f"       Place the hackathon file there as 'DataSet.csv'.\n"
            f"       Expected shape ~ (9082 rows, 3924 feature cols + target)."
        )
    log(f"Loading {csv_path.name} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    log(f"Loaded shape: {df.shape[0]:,} rows x {df.shape[1]:,} cols")
    return df


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    log(f"Wrote {path}")


def save_frame(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as parquet, falling back to CSV if pyarrow is absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        log(f"Wrote {path}  ({df.shape[0]:,} x {df.shape[1]:,})")
    except Exception:  # noqa: BLE001 — pyarrow/fastparquet not installed
        alt = path.with_suffix(".csv")
        df.to_csv(alt, index=False)
        log(f"parquet unavailable; wrote {alt}  ({df.shape[0]:,} x {df.shape[1]:,})")


def load_frame(path: Path) -> pd.DataFrame:
    """Load a DataFrame from parquet, falling back to a sibling .csv."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            pass
    alt = path.with_suffix(".csv")
    if alt.exists():
        return pd.read_csv(alt, low_memory=False)
    die(f"Expected intermediate file not found: {path} (or {alt}). Run earlier stages first.")
    return pd.DataFrame()  # unreachable, keeps type-checkers happy
