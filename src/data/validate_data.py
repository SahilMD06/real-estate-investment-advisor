"""Schema and quality validation for the housing dataset.

Validation is split into *errors* (the data cannot be used: missing columns,
empty frame, wrong types) and *warnings* (the data is usable but contains
records that are internally inconsistent and must be documented).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import CATEGORY_VALUES, NUMERIC_RANGES

EXPECTED_COLUMNS: tuple[str, ...] = (
    "ID", "State", "City", "Locality", "Property_Type", "BHK", "Size_in_SqFt",
    "Price_in_Lakhs", "Price_per_SqFt", "Year_Built", "Furnished_Status", "Floor_No",
    "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
    "Public_Transport_Accessibility", "Parking_Space", "Security", "Amenities",
    "Facing", "Owner_Type", "Availability_Status",
)
NUMERIC_COLUMNS: tuple[str, ...] = (
    "ID", "BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt", "Year_Built",
    "Floor_No", "Total_Floors", "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
)


class DataValidationError(ValueError):
    """Raised when the dataset cannot be used at all."""


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def _share(mask: pd.Series) -> float:
    return round(float(mask.mean()), 4) if len(mask) else 0.0


def validate_dataframe(df: pd.DataFrame, raise_on_error: bool = True) -> ValidationReport:
    """Validate schema, types, ranges, categories and internal consistency."""
    report = ValidationReport(n_rows=len(df), n_columns=df.shape[1])

    if df.empty:
        report.errors.append("Dataset is empty (0 rows).")
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        report.errors.append(f"Missing required columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        report.warnings.append(f"Unexpected extra columns ignored: {extra}")

    if report.errors:
        if raise_on_error:
            raise DataValidationError("; ".join(report.errors))
        return report

    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            report.errors.append(f"Column {col!r} must be numeric, found {df[col].dtype}.")
    if report.errors:
        if raise_on_error:
            raise DataValidationError("; ".join(report.errors))
        return report

    # Missing values and duplicates --------------------------------------
    nulls = df[list(EXPECTED_COLUMNS)].isna().sum()
    report.stats["missing_values"] = {k: int(v) for k, v in nulls.items() if v}
    if nulls.any():
        report.warnings.append(f"{int(nulls.sum())} missing values across {int((nulls > 0).sum())} columns.")
    report.stats["duplicate_rows"] = int(df.duplicated().sum())
    report.stats["duplicate_rows_excluding_id"] = int(df.drop(columns="ID").duplicated().sum())
    report.stats["duplicate_ids"] = int(df["ID"].duplicated().sum())
    if report.stats["duplicate_ids"]:
        report.warnings.append(f"{report.stats['duplicate_ids']} duplicated IDs.")

    # Ranges ----------------------------------------------------------------
    out_of_range = {}
    for col, (lo, hi) in NUMERIC_RANGES.items():
        n = int(((df[col] < lo) | (df[col] > hi)).sum())
        if n:
            out_of_range[col] = n
    report.stats["out_of_documented_range"] = out_of_range
    if out_of_range:
        report.warnings.append(f"Values outside documented ranges: {out_of_range}")
    non_positive = int(((df["Size_in_SqFt"] <= 0) | (df["Price_in_Lakhs"] <= 0)).sum())
    if non_positive:
        report.errors.append(f"{non_positive} rows have non-positive size or price.")

    # Categories ------------------------------------------------------------
    unknown = {}
    for col, allowed in CATEGORY_VALUES.items():
        bad = sorted(set(df[col].dropna().unique()) - set(allowed))
        if bad:
            unknown[col] = bad
    report.stats["unknown_categories"] = unknown
    if unknown:
        report.warnings.append(f"Unrecognised category values: {unknown}")
    report.stats["cardinality"] = {
        c: int(df[c].nunique()) for c in ("State", "City", "Locality", "Amenities")
    }
    city_states = df.groupby("City")["State"].nunique()
    report.stats["cities_in_multiple_states"] = int((city_states > 1).sum())
    report.stats["city_locality_pairs"] = int(df.groupby(["City", "Locality"]).ngroups)

    # Derived-field consistency --------------------------------------------
    pps_lakh = df["Price_in_Lakhs"] / df["Size_in_SqFt"]
    report.stats["price_per_sqft_matches_price_over_size"] = _share(
        (pps_lakh - df["Price_per_SqFt"]).abs() < 0.006
    )
    report.stats["price_per_sqft_rounded_to_zero"] = int((df["Price_per_SqFt"] == 0).sum())
    ages = df["Age_of_Property"] + df["Year_Built"]
    report.stats["age_reference_years"] = {int(k): int(v) for k, v in ages.value_counts().items()}

    # Internal inconsistencies (kept, flagged, documented) ---------------
    inconsistencies = {
        "floor_above_total_floors": _share(df["Floor_No"] > df["Total_Floors"]),
        "under_construction_built_2020_or_earlier": _share(
            (df["Availability_Status"] == "Under_Construction") & (df["Year_Built"] <= 2020)
        ),
        "house_or_villa_over_4_floors": _share(
            df["Property_Type"].isin(["Villa", "Independent House"]) & (df["Total_Floors"] > 4)
        ),
        # conditional shares: within 1-BHK and within 5-BHK listings
        "one_bhk_listings_over_3000_sqft": _share(df.loc[df["BHK"] == 1, "Size_in_SqFt"] > 3000),
        "five_bhk_listings_under_1000_sqft": _share(df.loc[df["BHK"] == 5, "Size_in_SqFt"] < 1000),
    }
    report.stats["inconsistency_shares"] = inconsistencies
    labels = {
        "floor_above_total_floors": "of all rows have floor above building height",
        "under_construction_built_2020_or_earlier": "of all rows are 'under construction' but built <= 2020",
        "house_or_villa_over_4_floors": "of all rows are villas/houses in buildings over 4 floors",
        "one_bhk_listings_over_3000_sqft": "of 1-BHK listings exceed 3,000 sqft",
        "five_bhk_listings_under_1000_sqft": "of 5-BHK listings are under 1,000 sqft",
    }
    for name, share in inconsistencies.items():
        if share > 0.01:
            report.warnings.append(f"{share:.1%} {labels[name]}")

    # Outliers (IQR rule) ----------------------------------------------
    pps_inr = pps_lakh * 1e5
    q1, q3 = np.percentile(pps_inr, [25, 75])
    upper = q3 + 1.5 * (q3 - q1)
    report.stats["pps_iqr_upper_fence_inr"] = round(float(upper), 1)
    report.stats["pps_iqr_outliers"] = int((pps_inr > upper).sum())

    if report.errors and raise_on_error:
        raise DataValidationError("; ".join(report.errors))
    return report
