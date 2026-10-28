"""Loading and cleaning the raw CSV into an analysis-ready frame."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.validate_data import validate_dataframe
from src.features.engineering import add_descriptive_features, parse_amenities
from src.utils.config import (
    LAKH, PROCESSED_DATA_PATH, QUALITY_REPORT_PATH, RAW_DATA_PATH,
)
from src.utils.logging import get_logger

log = get_logger(__name__)


class DatasetNotFoundError(FileNotFoundError):
    pass


def file_fingerprint(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file — used to version the dataset in MLflow."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Raw dataset not found at {path}. Place india_housing_prices.csv in data/raw/ "
            "(or set PROPWISE_RAW_DATA to its location)."
        )
    df = pd.read_csv(path)
    log.info("Loaded raw data %s rows × %s cols from %s", *df.shape, path.name)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy.

    * Recomputes price per sqft in rupees (the raw column is lakhs/sqft rounded
      to 2 dp, which collapses ~1% of rows to 0.00).
    * Normalises the Amenities string into five binary flags + a count.
    * Strips whitespace from text columns.
    * Adds consistency flags. Inconsistent rows are **kept and flagged**, not
      silently repaired, because we cannot know which field is wrong.
    """
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].astype(str).str.strip()
    out = out.drop_duplicates()

    out["Price_INR"] = out["Price_in_Lakhs"] * LAKH
    out["Price_per_SqFt_INR"] = (out["Price_INR"] / out["Size_in_SqFt"]).round(2)

    amen = parse_amenities(out["Amenities"])
    out = pd.concat([out, amen], axis=1)

    out["flag_floor_above_total"] = out["Floor_No"] > out["Total_Floors"]
    out["flag_uc_old_build"] = (out["Availability_Status"] == "Under_Construction") & (
        out["Year_Built"] <= 2020
    )
    out["flag_house_many_floors"] = out["Property_Type"].isin(["Villa", "Independent House"]) & (
        out["Total_Floors"] > 4
    )
    out["n_quality_flags"] = out[
        ["flag_floor_above_total", "flag_uc_old_build", "flag_house_many_floors"]
    ].sum(axis=1)

    out = add_descriptive_features(out)
    return out.reset_index(drop=True)


def prepare(raw_path: Path | str = RAW_DATA_PATH, out_path: Path | str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    raw = load_raw(raw_path)
    report = validate_dataframe(raw)
    for w in report.warnings:
        log.warning("Data quality: %s", w)
    cleaned = clean(raw)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out_path, index=False)
    payload = report.to_dict()
    payload["raw_sha256"] = file_fingerprint(Path(raw_path))
    payload["processed_rows"] = len(cleaned)
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s (%s rows) and %s", out_path.name, len(cleaned), QUALITY_REPORT_PATH.name)
    return cleaned


def load_processed(path: Path | str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Processed dataset not found at {path}. Run `python -m scripts.prepare_data` first."
        )
    return pd.read_parquet(path)
