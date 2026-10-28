"""Data validation, loading and cleaning."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import DatasetNotFoundError, clean, load_processed, load_raw, prepare
from src.data.validate_data import DataValidationError, validate_dataframe


def test_valid_frame_passes(raw_df):
    report = validate_dataframe(raw_df)
    assert report.ok
    assert report.stats["duplicate_rows"] == 0
    assert report.stats["cities_in_multiple_states"] == 0


def test_empty_dataset_is_rejected(raw_df):
    with pytest.raises(DataValidationError, match="empty"):
        validate_dataframe(raw_df.iloc[0:0])


def test_missing_column_is_rejected(raw_df):
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataframe(raw_df.drop(columns=["City"]))


def test_non_numeric_column_is_rejected(raw_df):
    bad = raw_df.assign(Size_in_SqFt=raw_df["Size_in_SqFt"].astype(str))
    with pytest.raises(DataValidationError, match="must be numeric"):
        validate_dataframe(bad)


def test_non_positive_price_is_an_error(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Price_in_Lakhs"] = 0
    report = validate_dataframe(bad, raise_on_error=False)
    assert not report.ok


def test_unknown_category_and_range_are_warnings(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "Facing"] = "Northeast"
    bad.loc[1, "BHK"] = 9
    report = validate_dataframe(bad)
    assert report.ok
    assert report.stats["unknown_categories"] == {"Facing": ["Northeast"]}
    assert report.stats["out_of_documented_range"]["BHK"] == 1


def test_missing_values_are_counted(raw_df):
    bad = raw_df.copy()
    bad.loc[:4, "Nearby_Schools"] = np.nan
    report = validate_dataframe(bad)
    assert report.stats["missing_values"] == {"Nearby_Schools": 5}


def test_inconsistency_shares_are_reported(raw_df):
    report = validate_dataframe(raw_df)
    expected = (raw_df["Floor_No"] > raw_df["Total_Floors"]).mean()
    assert report.stats["inconsistency_shares"]["floor_above_total_floors"] == pytest.approx(expected, abs=1e-4)


def test_clean_recomputes_price_per_sqft_in_rupees(raw_df):
    out = clean(raw_df)
    expected = raw_df["Price_in_Lakhs"] * 1e5 / raw_df["Size_in_SqFt"]
    np.testing.assert_allclose(out["Price_per_SqFt_INR"], expected, rtol=1e-6, atol=0.01)


def test_clean_parses_amenities_and_flags(raw_df):
    out = clean(raw_df)
    assert set(out["amenity_count"].unique()) <= {1, 2, 3, 4, 5}
    assert out["flag_floor_above_total"].equals(raw_df["Floor_No"] > raw_df["Total_Floors"])
    assert len(out) == len(raw_df)  # inconsistent rows are kept, not dropped


def test_clean_drops_exact_duplicates(raw_df):
    doubled = pd.concat([raw_df, raw_df.iloc[:3]], ignore_index=True)
    assert len(clean(doubled)) == len(raw_df)


def test_missing_raw_file_gives_actionable_error(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="data/raw"):
        load_raw(tmp_path / "nope.csv")


def test_missing_processed_file_gives_actionable_error(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="prepare_data"):
        load_processed(tmp_path / "nope.parquet")


def test_prepare_writes_parquet(raw_df, tmp_path, monkeypatch):
    import src.data.load_data as ld
    csv = tmp_path / "raw.csv"
    raw_df.to_csv(csv, index=False)
    monkeypatch.setattr(ld, "QUALITY_REPORT_PATH", tmp_path / "quality.json")
    out = prepare(csv, tmp_path / "clean.parquet")
    assert (tmp_path / "clean.parquet").exists() and (tmp_path / "quality.json").exists()
    assert len(pd.read_parquet(tmp_path / "clean.parquet")) == len(out)
