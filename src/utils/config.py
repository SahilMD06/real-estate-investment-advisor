"""Central configuration for PropWise AI.

Every path, constant and documented assumption lives here so that the
training scripts, the tests and the Streamlit app share one source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = Path(os.getenv("PROPWISE_RAW_DATA", DATA_DIR / "raw" / "india_housing_prices.csv"))
PROCESSED_DATA_PATH = Path(
    os.getenv("PROPWISE_PROCESSED_DATA", DATA_DIR / "processed" / "housing_clean.parquet")
)
QUALITY_REPORT_PATH = DATA_DIR / "processed" / "data_quality_report.json"

MODELS_DIR = Path(os.getenv("PROPWISE_MODELS_DIR", PROJECT_ROOT / "models"))
CLASSIFIER_PATH = MODELS_DIR / "classifier.joblib"
COHORT_STATS_PATH = MODELS_DIR / "cohort_stats.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
EVALUATION_PATH = MODELS_DIR / "evaluation.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# MLflow: a local SQLite backend by default. Override with MLFLOW_TRACKING_URI
# (e.g. a remote server) without touching code.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}")
MLFLOW_ARTIFACT_ROOT = (PROJECT_ROOT / "mlartifacts").as_uri()
CLASSIFICATION_EXPERIMENT = "propwise-investment-classification"
REGRESSION_EXPERIMENT = "propwise-valuation-signal-test"
REGISTERED_CLASSIFIER = "propwise-investment-classifier"

# --------------------------------------------------------------------------
# Data facts (verified against the CSV in Phase 1)
# --------------------------------------------------------------------------
# Age_of_Property == 2025 - Year_Built for every row, so the dataset's
# implicit reference year is 2025. We keep it for consistency.
REFERENCE_YEAR = 2025
LAKH = 100_000  # 1 lakh = ₹100,000

AMENITY_TOKENS = ("Clubhouse", "Garden", "Gym", "Playground", "Pool")

CATEGORY_VALUES: dict[str, tuple[str, ...]] = {
    "Property_Type": ("Apartment", "Independent House", "Villa"),
    "Furnished_Status": ("Furnished", "Semi-furnished", "Unfurnished"),
    "Public_Transport_Accessibility": ("Low", "Medium", "High"),
    "Parking_Space": ("No", "Yes"),
    "Security": ("No", "Yes"),
    "Facing": ("East", "North", "South", "West"),
    "Owner_Type": ("Broker", "Builder", "Owner"),
    "Availability_Status": ("Ready_to_Move", "Under_Construction"),
}

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "BHK": (1, 5),
    "Size_in_SqFt": (500, 5000),
    "Price_in_Lakhs": (10, 500),
    "Year_Built": (1990, 2023),
    "Floor_No": (0, 30),
    "Total_Floors": (1, 30),
    "Nearby_Schools": (1, 10),
    "Nearby_Hospitals": (1, 10),
}

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

# Comparable-property cohort used to define the "good value" label.
COHORT_KEYS = ("City", "Property_Type", "BHK")
# A property is labelled "good value" when its ₹/sqft is at or below
# GOOD_VALUE_RATIO × the training median ₹/sqft of its cohort.
GOOD_VALUE_RATIO = 1.0
MIN_COHORT_SIZE = 30  # smaller cohorts fall back to City, then global

# Expected calibration error threshold below which a probability is shown.
MAX_DISPLAY_ECE = 0.03

# --------------------------------------------------------------------------
# Five-year scenario assumptions (nominal, before costs)
# --------------------------------------------------------------------------
HORIZON_YEARS = 5
SCENARIO_RATES: dict[str, float] = {
    "Conservative": 0.02,
    "Baseline": 0.04,
    "Optimistic": 0.06,
}
MIN_RATE, MAX_RATE = -0.05, 0.15
RATE_SOURCES = [
    {
        "name": "RBI All-India House Price Index, Q1 2026-27 (released 24 Aug 2026)",
        "value": "3.6% year-on-year",
        "url": "https://rbi.org.in/SCRIPTS/BS_PressReleaseDisplay.aspx?prid=63436",
    },
    {
        "name": "NHB RESIDEX, Q3 FY2025-26 (Oct–Dec 2025), 50-city HPI @ assessment prices",
        "value": "5.0% year-on-year",
        "url": "https://www.nhb.org.in/wp-content/uploads/2026/03/National-Press-Release-RESIDEX-Dec25-DV002.pdf",
    },
]
RATE_RATIONALE = (
    "Baseline 4% sits between the latest RBI (3.6%) and NHB (5.0%) all-India annual "
    "readings. Conservative (2%) and Optimistic (6%) bracket it by ±2 points. These are "
    "single recent readings of national indices, not forecasts, and city-level growth "
    "varies widely (NHB Dec-2025 city readings ranged from −8.9% to +22.8%)."
)

APP_NAME = "PropWise AI"
APP_TAGLINE = "Evidence-first property value analytics for Indian residential real estate"
