"""Feature engineering.

Two kinds of features live here:

1. **Row-level descriptive features** (size band, area per bedroom, amenity
   flags, an infrastructure index …). They use only the row itself, so they
   cannot leak information across the train/test boundary.
2. **Cohort statistics** (median ₹/sqft of comparable properties). These are
   aggregates over many rows, so they are *fitted on the training split only*
   via :class:`CohortStats` and then applied to validation, test and user
   inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.utils.config import (
    AMENITY_TOKENS, COHORT_KEYS, LAKH, MIN_COHORT_SIZE, REFERENCE_YEAR,
)

SIZE_BANDS = [0, 1000, 2000, 3000, 4000, np.inf]
SIZE_BAND_LABELS = ["<1,000", "1,000–2,000", "2,000–3,000", "3,000–4,000", "4,000+"]
TRANSPORT_SCORE = {"Low": 0.0, "Medium": 0.5, "High": 1.0}


# ---------------------------------------------------------------------------
# Row-level features
# ---------------------------------------------------------------------------
def parse_amenities(series: pd.Series) -> pd.DataFrame:
    """Turn 'Pool, Gym' strings into binary flags and a count.

    Order and spacing in the raw strings vary (325 raw strings encode only 31
    distinct sets), so tokens are matched individually.
    """
    tokens = series.fillna("").astype(str).str.split(",").apply(
        lambda parts: {p.strip() for p in parts if p.strip()}
    )
    flags = pd.DataFrame(
        {f"has_{t.lower()}": tokens.apply(lambda s, t=t: int(t in s)) for t in AMENITY_TOKENS},
        index=series.index,
    )
    flags["amenity_count"] = flags.sum(axis=1)
    return flags


def add_descriptive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level, leakage-free features used for EDA, the UI and the models."""
    out = df.copy()
    out["Age_Years"] = REFERENCE_YEAR - out["Year_Built"]
    out["Size_Band"] = pd.cut(
        out["Size_in_SqFt"], bins=SIZE_BANDS, labels=SIZE_BAND_LABELS, right=False
    ).astype(str)
    out["SqFt_per_BHK"] = (out["Size_in_SqFt"] / out["BHK"]).round(1)
    # Floor position is only meaningful when the floor is within the building.
    valid = out["Floor_No"] <= out["Total_Floors"]
    out["Floor_Position"] = np.where(valid, out["Floor_No"] / out["Total_Floors"].clip(lower=1), np.nan)
    out["Transport_Score"] = out["Public_Transport_Accessibility"].map(TRANSPORT_SCORE)
    # Transparent, equal-weight index of the three access fields, each scaled
    # to 0–1. Weights are an assumption, not a validated valuation model.
    out["Infrastructure_Index"] = (
        ((out["Nearby_Schools"] - 1) / 9 + (out["Nearby_Hospitals"] - 1) / 9 + out["Transport_Score"]) / 3
    ).round(3)
    return out


# ---------------------------------------------------------------------------
# Model features
# ---------------------------------------------------------------------------
LISTING_NUMERIC = [
    "Price_in_Lakhs", "Size_in_SqFt", "log_price", "log_size", "BHK", "SqFt_per_BHK",
    "Age_Years", "Floor_No", "Total_Floors", "Nearby_Schools", "Nearby_Hospitals",
    "amenity_count", "has_clubhouse", "has_garden", "has_gym", "has_playground", "has_pool",
]
ATTRIBUTE_NUMERIC = [c for c in LISTING_NUMERIC if c not in ("Price_in_Lakhs", "log_price")]
MODEL_CATEGORICAL = [
    "City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility",
    "Parking_Space", "Security", "Facing", "Owner_Type", "Availability_Status",
]
# Inputs a user (or a dataset row) must supply for inference.
RAW_INPUT_COLUMNS = [
    "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt", "Price_in_Lakhs",
    "Year_Built", "Furnished_Status", "Floor_No", "Total_Floors", "Nearby_Schools",
    "Nearby_Hospitals", "Public_Transport_Accessibility", "Parking_Space", "Security",
    "Amenities", "Facing", "Owner_Type", "Availability_Status",
]


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Single function used by BOTH training and inference.

    Accepts raw-schema rows (dataset rows or user input) and returns every
    column any model may consume. Keeping this in one place guarantees that
    training and inference transform data identically.
    """
    missing = [c for c in RAW_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Input is missing required fields: {missing}")
    out = df[RAW_INPUT_COLUMNS].copy()
    amen_cols = [f"has_{t.lower()}" for t in AMENITY_TOKENS] + ["amenity_count"]
    out = out.drop(columns=[c for c in amen_cols if c in out.columns])
    out = pd.concat([out, parse_amenities(out["Amenities"])], axis=1)
    out = add_descriptive_features(out)
    out["log_price"] = np.log(out["Price_in_Lakhs"])
    out["log_size"] = np.log(out["Size_in_SqFt"])
    return out


# ---------------------------------------------------------------------------
# Cohort statistics (fit on training data only)
# ---------------------------------------------------------------------------
@dataclass
class CohortStats:
    """Median ₹/sqft per comparable cohort, learned from training rows only.

    Look-up falls back from (City, Property_Type, BHK) → City → global when a
    cohort has fewer than ``min_size`` training rows or was never seen.
    """

    keys: tuple[str, ...] = COHORT_KEYS
    min_size: int = MIN_COHORT_SIZE
    cohort_median: dict = field(default_factory=dict)
    cohort_count: dict = field(default_factory=dict)
    cohort_values: dict = field(default_factory=dict)
    city_median: dict = field(default_factory=dict)
    global_median: float = float("nan")
    n_train: int = 0

    @staticmethod
    def pps(df: pd.DataFrame) -> pd.Series:
        return df["Price_in_Lakhs"] * LAKH / df["Size_in_SqFt"]

    def fit(self, train: pd.DataFrame) -> "CohortStats":
        if train.empty:
            raise ValueError("Cannot fit cohort statistics on an empty training set.")
        pps = self.pps(train)
        frame = train[list(self.keys)].assign(pps=pps.values)
        grouped = frame.groupby(list(self.keys))["pps"]
        self.cohort_median = {k: float(v) for k, v in grouped.median().items()}
        self.cohort_count = {k: int(v) for k, v in grouped.size().items()}
        self.cohort_values = {k: np.sort(v.to_numpy(dtype=np.float32)) for k, v in grouped}
        self.city_median = {k: float(v) for k, v in frame.groupby("City")["pps"].median().items()}
        self.global_median = float(pps.median())
        self.n_train = len(train)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return benchmark columns aligned with ``df``'s index."""
        if not self.cohort_median:
            raise RuntimeError("CohortStats must be fitted before transform().")
        keys = [tuple(r) for r in df[list(self.keys)].itertuples(index=False, name=None)]
        med = np.array([self.cohort_median.get(k, np.nan) for k in keys], dtype=float)
        cnt = np.array([self.cohort_count.get(k, 0) for k in keys], dtype=int)
        city = df["City"].map(self.city_median).to_numpy(dtype=float)
        use_cohort = (cnt >= self.min_size) & ~np.isnan(med)
        use_city = ~use_cohort & ~np.isnan(city)
        bench = np.where(use_cohort, med, np.where(use_city, city, self.global_median))
        level = np.where(use_cohort, "cohort", np.where(use_city, "city", "global"))
        return pd.DataFrame(
            {"benchmark_pps": bench, "cohort_n": cnt, "benchmark_level": level}, index=df.index
        )

    def percentile(self, row: pd.Series, pps_value: float) -> float | None:
        """Share of training cohort members priced at or below ``pps_value`` (0–100)."""
        key = tuple(row[k] for k in self.keys)
        values = self.cohort_values.get(key)
        if values is None or len(values) < self.min_size:
            return None
        return float(np.searchsorted(values, pps_value, side="right") / len(values) * 100)
