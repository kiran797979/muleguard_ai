"""Routing by how much of its own schema the model actually has.

The deployed ensemble is fitted on 1,506 columns and needs most of them: masked
to 300 it scores at the random baseline. Scoring a partial extract with it
anyway produces confident noise, which is worse than refusing. These tests pin
the three tiers and the floor between them.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app import service  # noqa: E402


def test_floor_is_half_the_schema():
    """Measured, not chosen: 750 of 1506 scores 0.937, 300 scores 0.009."""
    assert service.COVERAGE_FLOOR == 0.50


def test_intersection_needs_enough_columns():
    """Fitting on one or two shared columns is not honest, so it refuses."""
    with pytest.raises(ValueError, match="too few"):
        service.fit_on_intersection({"a": "F1"}, pd.DataFrame({"a": [1, 2]}))


@pytest.mark.skipif(not (ROOT / "data" / "features.parquet").exists(),
                    reason="pipeline artefacts absent")
def test_partial_schema_fits_and_reports_its_own_ceiling():
    """A file below the floor is fitted on shared columns, and the fit reports
    its out-of-fold quality so the caller knows what the number is worth."""
    val = Path("D:/mule_validation_style_213_unlabeled.csv")
    if not val.exists():
        pytest.skip("validation-style extract not present")
    out = service.score_file(str(val), top=50)
    assert out["mode"] == "FITTED_ON_SHARED_COLUMNS"
    assert out["schema_coverage_pct"] < service.COVERAGE_FLOOR * 100
    assert 0.0 <= out["fit_quality"]["out_of_fold_auprc"] <= 1.0
    assert len(out["columns_used"]) >= service.MIN_INTERSECTION


def test_typology_output_does_not_claim_certainty():
    """That path asserts signal directions it cannot verify without labels, and
    on one extract it scored worse than chance. It must say so."""
    hard = Path("D:/mule_detection_213_hard_unlabeled.csv")
    if not hard.exists():
        pytest.skip("alien-schema extract not present")
    out = service.score_file(str(hard), top=20)
    assert out["mode"] == "TYPOLOGY_RANKING"
    assert "UNVALIDATED" in out["provenance"]
    assert "supply" in out["provenance"].lower()
