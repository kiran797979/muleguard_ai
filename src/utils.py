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
    """Load the raw dataset in whatever format it arrived in.

    This used to be a bare `pd.read_csv`, which is fine for the hackathon file
    and wrong for a dataset handed over on the day. Two failures showed up in
    testing and both are the sort you cannot afford live:

      * an .xlsx crashed with a raw pandas traceback;
      * a semicolon-delimited CSV loaded as ONE column and then failed several
        stages later complaining it could not find a target, which sends you
        hunting in completely the wrong place.

    So: Excel and parquet are read natively, the delimiter is sniffed rather
    than assumed, and a frame that arrives with a single column is treated as a
    parsing failure and reported as one.
    """
    if not csv_path.exists():
        die(
            f"Dataset not found at {csv_path}\n"
            f"       Point at it with MULEGUARD_DATA=<path>, or place it in data/.\n"
            f"       Accepted: .csv .tsv .txt .xlsx .xls .parquet"
        )

    suffix = csv_path.suffix.lower()
    log(f"Loading {csv_path.name} ...")

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        try:
            df = pd.read_excel(csv_path, sheet_name=0)
        except Exception as exc:  # noqa: BLE001
            die(f"Could not read {csv_path.name} as Excel: {exc}\n"
                f"       If openpyxl is missing: pip install openpyxl")
            raise
    elif suffix == ".parquet":
        df = pd.read_parquet(csv_path)
    else:
        # Sniff the delimiter. engine="python" with sep=None does this properly;
        # if it fails we fall back to a plain comma read rather than dying.
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python",
                             nrows=5000)  # sniff on a sample, it is slow
            sep = ","
            if df.shape[1] > 1:
                # Re-read the whole file fast, using the delimiter that worked.
                for cand in (",", ";", "\t", "|"):
                    probe = pd.read_csv(csv_path, sep=cand, nrows=5)
                    if probe.shape[1] == df.shape[1]:
                        sep = cand
                        break
            df = pd.read_csv(csv_path, sep=sep, low_memory=False)
            if sep != ",":
                log(f"Detected '{sep}' as the column separator.")
        except Exception:  # noqa: BLE001
            df = pd.read_csv(csv_path, low_memory=False)

    if df.shape[1] <= 1:
        die(
            f"{csv_path.name} parsed into only {df.shape[1]} column(s), which "
            f"almost always means the delimiter was not recognised.\n"
            f"       First column name looks like: {list(df.columns)[:1]}\n"
            f"       Re-save it as a comma-separated CSV, or convert it to "
            f".xlsx and pass that instead."
        )

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
