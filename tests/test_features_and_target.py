"""Feature engineering, cohort statistics (leakage safety) and the label rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineering import (
    RAW_INPUT_COLUMNS, CohortStats, add_descriptive_features, build_model_frame, parse_amenities,
)
from src.models.classification import make_splits
from src.models.target import make_label


def test_parse_amenities_ignores_order_and_spacing():
    out = parse_amenities(pd.Series(["Pool, Gym", "Gym,Pool", " Garden ", ""]))
    assert out.loc[0, "has_pool"] == out.loc[1, "has_pool"] == 1
    assert out.loc[0, "amenity_count"] == out.loc[1, "amenity_count"] == 2
    assert out.loc[2, "has_garden"] == 1 and out.loc[2, "amenity_count"] == 1
    assert out.loc[3, "amenity_count"] == 0


def test_descriptive_features(raw_df):
    out = add_descriptive_features(raw_df)
    assert (out["Age_Years"] == raw_df["Age_of_Property"]).all()
    assert out["Size_Band"].notna().all()
    bad_floor = raw_df["Floor_No"] > raw_df["Total_Floors"]
    assert out.loc[bad_floor, "Floor_Position"].isna().all()
    assert out["Infrastructure_Index"].between(0, 1).all()


def test_build_model_frame_requires_all_fields(sample_record):
    rec = dict(sample_record)
    rec.pop("BHK")
    with pytest.raises(KeyError, match="BHK"):
        build_model_frame(pd.DataFrame([rec]))


def test_build_model_frame_is_identical_for_batch_and_single_row(raw_df):
    batch = build_model_frame(raw_df)
    single = build_model_frame(raw_df.iloc[[7]])
    pd.testing.assert_frame_equal(batch.iloc[[7]].reset_index(drop=True), single.reset_index(drop=True),
                                  check_dtype=False)


def test_cohort_stats_use_training_rows_only(raw_df):
    splits = make_splits(raw_df)
    stats = CohortStats(min_size=5).fit(splits.train)
    before = dict(stats.cohort_median)
    # Changing test prices must not change anything learned from training rows.
    test_changed = splits.test.assign(Price_in_Lakhs=splits.test["Price_in_Lakhs"] * 10)
    stats.transform(test_changed)
    assert stats.cohort_median == before
    key = next(iter(before))
    mask = np.logical_and.reduce([splits.train[k] == v for k, v in zip(stats.keys, key)])
    expected = (splits.train.loc[mask, "Price_in_Lakhs"] * 1e5 / splits.train.loc[mask, "Size_in_SqFt"]).median()
    assert before[key] == pytest.approx(expected)


def test_cohort_fallback_levels(raw_df, sample_record):
    stats = CohortStats(min_size=10_000).fit(raw_df)  # every cohort "too small"
    out = stats.transform(pd.DataFrame([sample_record]))
    assert out.loc[0, "benchmark_level"] == "city"
    unseen = dict(sample_record, City="Atlantis")
    assert stats.transform(pd.DataFrame([unseen])).loc[0, "benchmark_level"] == "global"


def test_cohort_stats_errors():
    with pytest.raises(RuntimeError):
        CohortStats().transform(pd.DataFrame({"City": ["Pune"], "Property_Type": ["Villa"], "BHK": [2]}))
    with pytest.raises(ValueError, match="empty"):
        CohortStats().fit(pd.DataFrame(columns=RAW_INPUT_COLUMNS))


def test_percentile(raw_df):
    stats = CohortStats(min_size=5).fit(raw_df)
    key = max(stats.cohort_count, key=stats.cohort_count.get)
    row = pd.Series(dict(zip(stats.keys, key)))
    vals = stats.cohort_values[key]
    assert stats.percentile(row, float(vals.min()) - 1) == 0
    assert stats.percentile(row, float(vals.max()) + 1) == 100


def test_label_matches_rule(raw_df):
    stats = CohortStats(min_size=5).fit(raw_df)
    y = make_label(raw_df, stats)
    bench = stats.transform(raw_df)["benchmark_pps"]
    pps = raw_df["Price_in_Lakhs"] * 1e5 / raw_df["Size_in_SqFt"]
    assert (y == (pps <= bench).astype(int)).all()
    assert 0.3 < y.mean() < 0.7


def test_splits_are_disjoint_and_cover_data(raw_df):
    s = make_splits(raw_df)
    ids = [set(x["ID"]) for x in (s.train, s.val, s.test)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(len(i) for i in ids) == len(raw_df)
