"""Shared test fixtures.

`raw_df` is a SMALL GENERATED TEST FIXTURE that follows the CSV schema. It
exists only so tests run without the 41 MB dataset. It is never used for
training the shipped models or for any reported result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.engineering import LISTING_NUMERIC, CohortStats
from src.models.classification import build_pipeline, raw_inputs
from src.models.inference import ModelBundle
from src.models.target import make_label
from src.utils import config as C

CITY_STATE = {"Pune": "Maharashtra", "Mumbai": "Maharashtra", "Chennai": "Tamil Nadu"}


def make_fixture_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cities = rng.choice(list(CITY_STATE), n)
    amen_pool = list(C.AMENITY_TOKENS)
    df = pd.DataFrame({
        "ID": np.arange(1, n + 1),
        "State": [CITY_STATE[c] for c in cities],
        "City": cities,
        "Locality": [f"Locality_{i}" for i in rng.integers(1, 20, n)],
        "Property_Type": rng.choice(C.CATEGORY_VALUES["Property_Type"], n),
        "BHK": rng.integers(1, 4, n),
        "Size_in_SqFt": rng.integers(500, 5001, n),
        "Price_in_Lakhs": rng.uniform(10, 500, n).round(2),
        "Year_Built": rng.integers(1990, 2024, n),
        "Furnished_Status": rng.choice(C.CATEGORY_VALUES["Furnished_Status"], n),
        "Floor_No": rng.integers(0, 31, n),
        "Total_Floors": rng.integers(1, 31, n),
        "Nearby_Schools": rng.integers(1, 11, n),
        "Nearby_Hospitals": rng.integers(1, 11, n),
        "Public_Transport_Accessibility": rng.choice(C.CATEGORY_VALUES["Public_Transport_Accessibility"], n),
        "Parking_Space": rng.choice(["No", "Yes"], n),
        "Security": rng.choice(["No", "Yes"], n),
        "Amenities": [", ".join(rng.choice(amen_pool, rng.integers(1, 6), replace=False)) for _ in range(n)],
        "Facing": rng.choice(C.CATEGORY_VALUES["Facing"], n),
        "Owner_Type": rng.choice(C.CATEGORY_VALUES["Owner_Type"], n),
        "Availability_Status": rng.choice(C.CATEGORY_VALUES["Availability_Status"], n),
    })
    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] / df["Size_in_SqFt"]).round(2)
    df["Age_of_Property"] = C.REFERENCE_YEAR - df["Year_Built"]
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_fixture_frame()


@pytest.fixture
def sample_record() -> dict:
    return {
        "State": "Maharashtra", "City": "Pune", "Locality": "Locality_3", "Property_Type": "Apartment",
        "BHK": 2, "Size_in_SqFt": 1200, "Price_in_Lakhs": 90.0, "Year_Built": 2015,
        "Furnished_Status": "Semi-furnished", "Floor_No": 4, "Total_Floors": 12, "Nearby_Schools": 5,
        "Nearby_Hospitals": 3, "Public_Transport_Accessibility": "High", "Parking_Space": "Yes",
        "Security": "Yes", "Amenities": "Gym, Pool", "Facing": "East", "Owner_Type": "Owner",
        "Availability_Status": "Ready_to_Move",
    }


@pytest.fixture
def tiny_bundle(raw_df) -> ModelBundle:
    """A real (small) trained bundle built from the fixture frame."""
    train = raw_df.iloc[:450]
    stats = CohortStats(min_size=5).fit(train)
    y = make_label(train, stats)
    pipe = build_pipeline(LogisticRegression(max_iter=2000), LISTING_NUMERIC, scale=True)
    pipe.fit(raw_inputs(train), y)
    metadata = {
        "classifier": {"show_probability": True, "test_ece": 0.01, "champion": "listing/logistic_regression"},
        "domain": {"city_state": CITY_STATE},
    }
    return ModelBundle(classifier=pipe, cohort_stats=stats, metadata=metadata, evaluation={})
