"""The label-free path on a schema the model was never trained on.

A file from another bank shares no column name with the training data, so the
deployed ensemble cannot score it. These tests pin the fallback: typology
rebuilt by meaning, oriented by directions fixed in advance, and cut by Otsu.

The fixture is the 100-account test extract with 40 known mules. It is
synthetic and cleanly separated, so the perfect scores below are a statement
about the plumbing, not a claim about production performance.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import service  # noqa: E402

FIXTURE = ROOT / "runs" / "mule_account_test_data_12e88411ef3c" / "data" / "uploaded.csv"
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="labelled fixture absent")


@pytest.fixture(scope="module")
def data():
    df = pd.read_csv(FIXTURE)
    return df.drop(columns=["mule_label"]), df["mule_label"].astype(int).to_numpy()


def test_signals_are_rebuilt_from_an_unfamiliar_schema(data):
    df, _ = data
    sig, prov = service._typology_by_role(df)
    assert not sig.empty
    assert "passthrough" in sig.columns          # derived, not read
    assert len(sig.columns) >= 8
    for col in prov.values():                    # every signal traces to a real column
        assert col.split(" (")[0] in df.columns


def test_counterparty_counts_stay_out(data):
    """Their direction depends on the role the account plays, so it cannot be
    fixed in advance. Assuming 'more is worse' scored below random."""
    df, _ = data
    sig, _ = service._typology_by_role(df)
    assert not any("benefic" in c.lower() or "sender" in c.lower() for c in sig.columns)


def test_ranking_separates_mules(data):
    df, y = data
    score = service._rank_score(service._typology_by_role(df)[0])
    order = np.argsort(-score)
    assert y[order[:40]].sum() >= 36          # ~90% of the mules in the top 40


def test_otsu_flags_without_a_tuned_threshold(data):
    from temporal import otsu_threshold
    df, y = data
    score = service._rank_score(service._typology_by_role(df)[0])
    cut, _ = otsu_threshold(score)
    flag = score >= cut
    tp = int((flag & (y == 1)).sum())
    precision = tp / max(int(flag.sum()), 1)
    recall = tp / max(int(y.sum()), 1)
    assert precision >= 0.85 and recall >= 0.85
