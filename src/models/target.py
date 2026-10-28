"""Definition of the rule-derived "good value" label.

The dataset has no observed investment outcomes (no resale prices, rents or
returns). The label therefore encodes a *documented rule*:

    good_value = 1  if  ₹/sqft  <=  GOOD_VALUE_RATIO × median ₹/sqft of the
                        property's comparable cohort (City × Property_Type × BHK),
                        where the median is computed on the TRAINING split only.

A model trained on this label learns to reproduce the rule. It does not
predict verified future returns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.engineering import CohortStats
from src.utils.config import GOOD_VALUE_RATIO

LABEL_NAME = "good_value"
LABEL_TEXT = {1: "Potentially good value", 0: "Priced above comparables"}


def make_label(df: pd.DataFrame, stats: CohortStats, ratio: float = GOOD_VALUE_RATIO) -> pd.Series:
    bench = stats.transform(df)["benchmark_pps"]
    pps = CohortStats.pps(df)
    return pd.Series(np.where(pps <= ratio * bench, 1, 0), index=df.index, name=LABEL_NAME)
