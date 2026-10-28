"""Inference: load the approved model bundle and analyse a property.

The Streamlit app only ever calls :func:`load_bundle` and
:func:`analyze_property`. Nothing here retrains models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.engineering import RAW_INPUT_COLUMNS, CohortStats
from src.models.persistence import load_classifier
from src.models.scenario import run_scenarios, validate_rate
from src.models.target import LABEL_TEXT
from src.utils import config as C


class ModelArtifactsMissing(FileNotFoundError):
    """Raised when the trained bundle is not present."""


class InvalidPropertyInput(ValueError):
    """Raised when user input cannot be analysed."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ModelBundle:
    classifier: Any
    cohort_stats: CohortStats
    metadata: dict
    evaluation: dict

    @property
    def city_state(self) -> dict[str, str]:
        return self.metadata["domain"]["city_state"]


def load_bundle(models_dir: Path | str = C.MODELS_DIR) -> ModelBundle:
    models_dir = Path(models_dir)
    paths = {
        "classifier": models_dir / C.CLASSIFIER_PATH.name,
        "cohort_stats": models_dir / C.COHORT_STATS_PATH.name,
        "metadata": models_dir / C.METADATA_PATH.name,
        "evaluation": models_dir / C.EVALUATION_PATH.name,
    }
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        raise ModelArtifactsMissing(
            f"Model artefacts missing from {models_dir}: {missing}. "
            "Run `python -m scripts.prepare_data` then `python -m scripts.train_models`."
        )
    return ModelBundle(
        classifier=load_classifier(paths["classifier"]),
        cohort_stats=joblib.load(paths["cohort_stats"]),
        metadata=json.loads(paths["metadata"].read_text()),
        evaluation=json.loads(paths["evaluation"].read_text()),
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
INTEGER_FIELDS = ("BHK", "Size_in_SqFt", "Year_Built", "Floor_No", "Total_Floors",
                  "Nearby_Schools", "Nearby_Hospitals")


def validate_property(record: dict, city_state: dict[str, str]) -> tuple[dict, list[str]]:
    """Return a cleaned record and a list of non-blocking warnings.

    Raises :class:`InvalidPropertyInput` with every blocking problem at once.
    """
    errors: list[str] = []
    warnings: list[str] = []
    rec = dict(record)
    rec.setdefault("Locality", "Unspecified")
    rec.setdefault("Amenities", "")

    missing = [c for c in RAW_INPUT_COLUMNS if c not in rec or rec[c] is None]
    if missing:
        raise InvalidPropertyInput([f"Missing required field(s): {', '.join(missing)}"])

    # Types --------------------------------------------------------------
    for col in (*INTEGER_FIELDS, "Price_in_Lakhs"):
        try:
            val = float(rec[col])
        except (TypeError, ValueError):
            errors.append(f"{col.replace('_', ' ')} must be a number.")
            continue
        if not np.isfinite(val):
            errors.append(f"{col.replace('_', ' ')} must be a finite number.")
            continue
        rec[col] = int(round(val)) if col in INTEGER_FIELDS else val
    if errors:
        raise InvalidPropertyInput(errors)

    if rec["Price_in_Lakhs"] <= 0:
        errors.append("Price must be greater than zero.")
    if rec["Size_in_SqFt"] <= 0:
        errors.append("Size must be greater than zero.")
    if rec["BHK"] < 1:
        errors.append("BHK must be at least 1.")
    if rec["Total_Floors"] < 1:
        errors.append("Total floors must be at least 1.")
    if rec["Floor_No"] < 0:
        errors.append("Floor number cannot be negative (use 0 for ground floor).")
    if rec["Year_Built"] > C.REFERENCE_YEAR + 5:
        errors.append(f"Year built {rec['Year_Built']} is too far in the future.")

    # Categories ---------------------------------------------------------
    if rec["City"] not in city_state:
        errors.append(f"City '{rec['City']}' is not covered by the training data.")
    elif city_state[rec["City"]] != rec["State"]:
        errors.append(f"{rec['City']} is in {city_state[rec['City']]}, not {rec['State']}.")
    for col, allowed in C.CATEGORY_VALUES.items():
        if rec[col] not in allowed:
            errors.append(f"{col.replace('_', ' ')} '{rec[col]}' is not supported. Choose one of: {', '.join(allowed)}.")
    if errors:
        raise InvalidPropertyInput(errors)

    # Out-of-distribution warnings -------------------------------------
    for col, (lo, hi) in C.NUMERIC_RANGES.items():
        if not lo <= rec[col] <= hi:
            warnings.append(
                f"{col.replace('_', ' ')} = {rec[col]:,} is outside the training range "
                f"({lo:,}–{hi:,}). Results are extrapolations and less reliable."
            )
    if rec["Floor_No"] > rec["Total_Floors"]:
        warnings.append("Floor number is above the building's total floors — please double-check.")
    if rec["Availability_Status"] == "Under_Construction" and rec["Year_Built"] < C.REFERENCE_YEAR - 3:
        warnings.append("Marked under construction but built several years ago — please double-check.")
    return rec, warnings


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    record: dict
    pps: float
    benchmark_pps: float
    benchmark_level: str
    cohort_n: int
    gap_pct: float
    cohort_percentile: float | None
    city_median_pps: float | None
    rule_label: int
    model_probability: float
    model_label: int
    show_probability: bool
    benchmark_value_lakhs: float
    scenarios: list
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return LABEL_TEXT[self.rule_label]

    @property
    def models_agree(self) -> bool:
        return self.rule_label == self.model_label

    @property
    def borderline(self) -> bool:
        return abs(self.gap_pct) < 5


def analyze_property(record: dict, bundle: ModelBundle, rates: dict[str, float] | None = None) -> AnalysisResult:
    rec, warnings = validate_property(record, bundle.city_state)
    rates = rates or C.SCENARIO_RATES
    for r in rates.values():
        validate_rate(r)

    frame = pd.DataFrame([rec])
    stats = bundle.cohort_stats
    bench = stats.transform(frame).iloc[0]
    pps = float(rec["Price_in_Lakhs"] * C.LAKH / rec["Size_in_SqFt"])
    bpps = float(bench["benchmark_pps"])
    gap = (pps / bpps - 1) * 100
    rule = int(pps <= C.GOOD_VALUE_RATIO * bpps)
    prob = float(bundle.classifier.predict_proba(frame[RAW_INPUT_COLUMNS])[:, 1][0])
    pct = stats.percentile(frame.iloc[0], pps)
    city_med = stats.city_median.get(rec["City"])

    if bench["benchmark_level"] != "cohort":
        warnings.append(
            f"Fewer than {stats.min_size} comparable training properties for this "
            f"{rec['City']} / {rec['Property_Type']} / {rec['BHK']} BHK combination; "
            f"benchmark falls back to the {bench['benchmark_level']} median."
        )

    positives, negatives, context = [], [], []
    if gap <= 0:
        positives.append(f"₹/sqft is {abs(gap):.1f}% below the median of comparable properties.")
    else:
        negatives.append(f"₹/sqft is {gap:.1f}% above the median of comparable properties.")
    if city_med:
        cg = (pps / city_med - 1) * 100
        (positives if cg <= 0 else negatives).append(
            f"{abs(cg):.1f}% {'below' if cg <= 0 else 'above'} the {rec['City']} city-wide median ₹/sqft."
        )
    if pct is not None:
        (positives if pct <= 50 else negatives).append(
            f"Cheaper per sqft than {100 - pct:.0f}% of comparable training listings."
            if pct <= 50 else f"More expensive per sqft than {pct:.0f}% of comparable training listings."
        )
    # Context factors: domain considerations NOT used in the verdict.
    if rec["Availability_Status"] == "Under_Construction":
        context.append("Under construction: delivery and completion risk is not modelled.")
    if C.REFERENCE_YEAR - rec["Year_Built"] >= 25:
        context.append("Building is 25+ years old: maintenance and redevelopment factors are not modelled.")
    if rec["Public_Transport_Accessibility"] == "High":
        context.append("High public-transport access (in this dataset it shows no measurable price effect).")
    n_amen = len([a for a in str(rec["Amenities"]).split(",") if a.strip()])
    context.append(f"{n_amen} of 5 listed amenities (no measurable price premium in this dataset).")

    return AnalysisResult(
        record=rec, pps=pps, benchmark_pps=bpps, benchmark_level=str(bench["benchmark_level"]),
        cohort_n=int(bench["cohort_n"]), gap_pct=gap, cohort_percentile=pct,
        city_median_pps=city_med, rule_label=rule, model_probability=prob,
        model_label=int(prob >= 0.5),
        show_probability=bool(bundle.metadata["classifier"]["show_probability"]),
        benchmark_value_lakhs=bpps * rec["Size_in_SqFt"] / C.LAKH,
        scenarios=run_scenarios(rec["Price_in_Lakhs"], rates),
        positives=positives, negatives=negatives, context=context, warnings=warnings,
    )
