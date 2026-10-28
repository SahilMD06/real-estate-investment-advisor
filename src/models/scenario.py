"""Transparent five-year scenario valuation.

The dataset is a single cross-section with no repeated sales, so a
supervised five-year forecast cannot be trained. Instead we compound the
current price at user-visible annual rates:

    future_price = current_price × (1 + r) ** years

Excluded on purpose (stated in the UI): stamp duty and registration,
brokerage, maintenance, property tax, rental income, financing costs,
inflation adjustment and capital-gains tax.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.config import HORIZON_YEARS, MAX_RATE, MIN_RATE, SCENARIO_RATES


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    annual_rate: float
    years: int
    current_value: float
    future_value: float

    @property
    def gain(self) -> float:
        return self.future_value - self.current_value

    @property
    def gain_pct(self) -> float:
        return self.gain / self.current_value * 100


def validate_rate(rate: float) -> float:
    if rate is None or not isinstance(rate, (int, float)):
        raise TypeError("Annual rate must be a number, e.g. 0.04 for 4%.")
    if not (MIN_RATE <= rate <= MAX_RATE):
        raise ValueError(
            f"Annual rate {rate:.2%} is outside the supported range "
            f"{MIN_RATE:.0%} to {MAX_RATE:.0%}. Enter rates as decimals (4% → 0.04)."
        )
    return float(rate)


def project_value(current_value: float, annual_rate: float, years: int = HORIZON_YEARS) -> float:
    if current_value is None or current_value <= 0:
        raise ValueError("Current value must be a positive number.")
    if years < 0:
        raise ValueError("Years must be non-negative.")
    validate_rate(annual_rate)
    return float(current_value * (1 + annual_rate) ** years)


def run_scenarios(
    current_value: float,
    rates: dict[str, float] | None = None,
    years: int = HORIZON_YEARS,
) -> list[ScenarioResult]:
    rates = rates or SCENARIO_RATES
    return [
        ScenarioResult(name, rate, years, float(current_value), project_value(current_value, rate, years))
        for name, rate in rates.items()
    ]


def trajectory(current_value: float, rates: dict[str, float], years: int = HORIZON_YEARS) -> pd.DataFrame:
    """Year-by-year values for charting."""
    rows = [
        {"Year": y, "Scenario": name, "Value": project_value(current_value, r, y)}
        for name, r in rates.items()
        for y in range(years + 1)
    ]
    return pd.DataFrame(rows)
