"""
MuleGuard — fittable, persistable preprocessor (closes the no-inference-path gap).

Stage 1 (01_clean.py) originally derived every cleaning decision inline from
whatever dataframe it was handed — median values, one-hot vocabulary, the
date-vintage reference date, and the sparse/leak/collinear drop lists. None of
that state was saved, so a genuinely NEW account could not be scored: the transform
could not be replayed, and data-dependent thresholds meant the surviving feature
set silently changed batch to batch.

`Preprocessor` fixes that. `fit()` learns and stores ALL transform state; `transform()`
replays it deterministically on new raw rows (no target, no re-derivation); `save()`/
`load()` persist it next to the model. The fitted schema (ordered surviving columns),
per-column medians, one-hot vocabulary, date reference, parse convention, per-column
type decisions, and all drop lists travel with the model.

Design constraint: fit()+transform() on the training CSV must reproduce the EXISTING
cleaned matrix bit-for-bit (same columns, order, dtypes, values), so no measured
number moves. That equivalence is asserted by tests/test_preprocess.py.

The class is intentionally a bespoke transformer rather than a sklearn Pipeline:
the date-vintage-with-persisted-reference logic, the day/month parse choice, and the
supervised global drops (corr-with-target, collinear) do not fit the ColumnTransformer
paradigm, and pandas.get_dummies column naming/order must be preserved exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config as C


class Preprocessor:
    """Learns Stage-1 cleaning state on fit; replays it deterministically on transform."""

    def __init__(self) -> None:
        # Populated by fit(). All are fit-time-only and reused verbatim at transform.
        self.id_cols: list[str] = []
        self.col_types: dict[str, str] = {}          # raw obj col -> one of kept_numeric/one_hot/date/drop/leak
        self.one_hot_vocab: dict[str, list[str]] = {}  # base col -> ordered dummy column names
        self.date_cols: dict[str, dict] = {}         # col -> {"dayfirst": bool, "ref": iso-date str}
        self.sparse_dropped: list[str] = []
        self.leak_dropped: list[str] = []
        self.collinear_dropped: list[str] = []
        self.medians: dict[str, float] = {}          # feature col -> median (fit-time)
        self.final_columns: list[str] = []           # ordered surviving cols (incl. target at fit)
        self.feature_columns: list[str] = []         # final_columns minus target
        self.fitted: bool = False

    # ----------------------------------------------------------------------
    # Fit
    # ----------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Learn all state from a raw dataframe (target present) and return cleaned df + report.

        Mirrors 01_clean.py's main() step order exactly so the output is identical.
        """
        report: dict = {"input_shape": list(df.shape)}
        df = df.copy()
        target = C.TARGET_COL

        # 0. Drop identifier columns (R "Unnamed: 0" row-id).
        self.id_cols = [c for c in df.columns
                        if any(str(c).startswith(p) for p in C.ID_COL_PREFIXES)]
        if self.id_cols:
            df = df.drop(columns=self.id_cols)
        report["dropped_id_columns"] = self.id_cols

        y = df[target]
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        report["class_balance"] = {
            "positives(mules)": n_pos, "negatives(normal)": n_neg,
            "prevalence_pct": round(100 * n_pos / len(y), 4),
        }

        # 2. Encode categoricals (learn vocab / date refs / type decisions).
        df, cat_info = self._fit_encode_categoricals(df, protect=[target])
        report["categorical_encoding"] = cat_info
        df = self._coerce_numeric(df, protect=[target])

        # 3. Sparse drop (fit-time list).
        miss_frac = df.drop(columns=[target]).isna().mean()
        self.sparse_dropped = miss_frac[miss_frac > C.MISSING_DROP_FRAC].index.tolist()
        df = df.drop(columns=self.sparse_dropped)
        report["dropped_sparse_count"] = len(self.sparse_dropped)

        # 4. Correlation leak scan (fit-time list). Mirrors detect_leaks + forced LEAK_COL.
        self.leak_dropped, top_corr = self._detect_leaks(df, target)
        report["top_target_correlations"] = top_corr
        if C.LEAK_COL in df.columns and C.LEAK_COL not in self.leak_dropped:
            self.leak_dropped.append(C.LEAK_COL)
        self.leak_dropped = [c for c in self.leak_dropped if c in df.columns]
        df = df.drop(columns=self.leak_dropped)
        report["removed_leak_columns"] = self.leak_dropped

        # 5. Collinear / constant de-dup (fit-time list).
        df, self.collinear_dropped = self._drop_collinear(df, target)
        report["dropped_collinear_count"] = len(self.collinear_dropped)

        # 6. Median impute (learn medians).
        feat_cols = [c for c in df.columns if c != target]
        self.medians = df[feat_cols].median(numeric_only=True).to_dict()
        df[feat_cols] = df[feat_cols].fillna(self.medians)
        still_na = [c for c in feat_cols if df[c].isna().any()]
        if still_na:
            df[still_na] = df[still_na].fillna(0)
            for c in still_na:
                self.medians.setdefault(c, 0.0)

        self.final_columns = list(df.columns)
        self.feature_columns = feat_cols
        report["output_shape"] = list(df.shape)
        report["clean_feature_count"] = len(feat_cols)
        self.fitted = True
        return df, report

    # ----------------------------------------------------------------------
    # Transform (new data — no target required, nothing re-derived)
    # ----------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replay the fitted transform on new raw rows. Returns feature matrix (no target)."""
        if not self.fitted:
            raise RuntimeError("Preprocessor.transform called before fit/load.")
        df = df.copy()

        # 0. Drop id columns if present.
        drop_now = [c for c in self.id_cols if c in df.columns]
        if drop_now:
            df = df.drop(columns=drop_now)

        # 2. Replay categorical encoding using stored decisions.
        for col, kind in self.col_types.items():
            if col not in df.columns:
                continue
            s = df[col]
            if kind == "kept_numeric":
                df[col] = pd.to_numeric(s, errors="coerce")
            elif kind == "date":
                meta = self.date_cols[col]
                dt = pd.to_datetime(s, errors="coerce", dayfirst=meta["dayfirst"])
                ref = pd.Timestamp(meta["ref"])
                df[col] = (ref - dt).dt.days.astype(float)
            elif kind == "one_hot":
                dummies = pd.get_dummies(s, prefix=col, dummy_na=False, dtype=float)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            elif kind in ("drop", "leak"):
                df = df.drop(columns=[col])

        # Coerce any remaining objects (except nothing is protected at score time).
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Reindex to the fitted one-hot vocabulary + final feature schema. Unseen
        # categories vanish (their dummy isn't in the schema); absent categories get
        # an all-zero column. This is exactly the fit-time behaviour for new levels.
        out = df.reindex(columns=self.feature_columns)

        # Impute. A one-hot dummy that is ABSENT from a new batch means "this row is
        # not in that category" -> it must be 0.0, NOT a learned median. (A dummy for
        # a majority category has median 1.0; median-filling it would silently encode
        # a new account INTO a category it isn't in — and for single-row scoring could
        # emit two 1s in a mutually-exclusive one-hot group.) Only genuinely numeric
        # / date-vintage columns use the fit-time median.
        dummy_cols = {c for dummies in self.one_hot_vocab.values() for c in dummies}
        for col in self.feature_columns:
            med = 0.0 if col in dummy_cols else self.medians.get(col, 0.0)
            out[col] = out[col].fillna(med)
        out = out.fillna(0.0)
        return out[self.feature_columns]

    # ----------------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id_cols": self.id_cols,
            "col_types": self.col_types,
            "one_hot_vocab": self.one_hot_vocab,
            "date_cols": self.date_cols,
            "sparse_dropped": self.sparse_dropped,
            "leak_dropped": self.leak_dropped,
            "collinear_dropped": self.collinear_dropped,
            "medians": self.medians,
            "final_columns": self.final_columns,
            "feature_columns": self.feature_columns,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        p = cls()
        for k, v in d.items():
            setattr(p, k, v)
        return p

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(d)

    # ----------------------------------------------------------------------
    # Internals — mirror 01_clean.py exactly.
    # ----------------------------------------------------------------------
    @staticmethod
    def _coerce_numeric(df: pd.DataFrame, protect: list[str]) -> pd.DataFrame:
        for col in df.columns:
            if col in protect:
                continue
            if df[col].dtype == object:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def _is_categorical_leak(s: pd.Series, y: pd.Series) -> bool:
        base_err = min(y.mean(), 1 - y.mean())
        if base_err <= 0:
            return False
        g = pd.DataFrame({"cat": s.values, "y": y.values})
        cond_err = (g.groupby("cat")["y"]
                    .apply(lambda v: min(v.mean(), 1 - v.mean()) * len(v)).sum()) / len(g)
        return bool((base_err - cond_err) / base_err >= C.CATEGORICAL_LEAK_ERROR_REDUCTION)

    def _fit_encode_categoricals(self, df: pd.DataFrame, protect: list[str]) -> tuple[pd.DataFrame, dict]:
        """Same rules as 01_clean.encode_categoricals, but records decisions for replay."""
        info: dict = {"one_hot": {}, "date_vintage": [], "dropped_highcard": [],
                      "kept_numeric": [], "dropped_leak": []}
        obj_cols = [c for c in df.columns if c not in protect and df[c].dtype == object]
        y = df[C.TARGET_COL] if C.TARGET_COL in df.columns else None

        for col in obj_cols:
            s = df[col]
            as_num = pd.to_numeric(s, errors="coerce")
            if as_num.notna().mean() >= C.DATE_PARSE_MIN_FRAC:
                df[col] = as_num
                self.col_types[col] = "kept_numeric"
                info["kept_numeric"].append(col)
                continue

            nun = s.nunique(dropna=True)
            if nun <= C.CATEGORICAL_MAX_CARDINALITY:
                if y is not None and self._is_categorical_leak(s, y):
                    df = df.drop(columns=[col])
                    self.col_types[col] = "leak"
                    info["dropped_leak"].append({"col": col, "cardinality": int(nun)})
                    continue
                dummies = pd.get_dummies(s, prefix=col, dummy_na=False, dtype=float)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                self.col_types[col] = "one_hot"
                self.one_hot_vocab[col] = list(dummies.columns)
                info["one_hot"][col] = list(dummies.columns)
                continue

            dt_mdy = pd.to_datetime(s, errors="coerce", dayfirst=False)
            dt_dmy = pd.to_datetime(s, errors="coerce", dayfirst=True)
            use_dmy = dt_dmy.notna().mean() > dt_mdy.notna().mean()
            dt = dt_dmy if use_dmy else dt_mdy
            if dt.notna().mean() >= C.DATE_PARSE_MIN_FRAC:
                ref = dt.max()
                df[col] = (ref - dt).dt.days.astype(float)
                self.col_types[col] = "date"
                self.date_cols[col] = {"dayfirst": bool(use_dmy), "ref": ref.isoformat()}
                info["date_vintage"].append(col)
                continue

            df = df.drop(columns=[col])
            self.col_types[col] = "drop"
            info["dropped_highcard"].append({"col": col, "cardinality": int(nun)})

        return df, info

    @staticmethod
    def _detect_leaks(df: pd.DataFrame, target: str) -> tuple[list[str], dict]:
        feats = df.drop(columns=[target])
        y = df[target]
        with np.errstate(divide="ignore", invalid="ignore"):
            corrs = feats.corrwith(y, numeric_only=True).abs().dropna()
        leaks = corrs[corrs > C.LEAK_CORR_THRESHOLD].sort_values(ascending=False)
        top = corrs.sort_values(ascending=False).head(15).round(4).to_dict()
        return leaks.index.tolist(), top

    @staticmethod
    def _drop_collinear(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
        feats = df.drop(columns=[target])
        nunique = feats.nunique()
        constant = nunique[nunique <= 1].index.tolist()
        feats = feats.drop(columns=constant)
        corr = feats.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        dup = [c for c in upper.columns if (upper[c] > C.COLLINEAR_CORR_THRESHOLD).any()]
        to_drop = sorted(set(constant + dup))
        return df.drop(columns=to_drop), to_drop


# Default location for the persisted preprocessor (next to the model bundle).
PREPROCESSOR_PATH = C.MODELS_DIR / "preprocessor.json"
