# Data Dictionary

Source: `india_housing_prices.csv`: 250,000 rows, 23 columns, no missing values, no duplicate rows.
Observed values come from profiling (`data/processed/data_quality_report.json`, `reports/eda_stats.json`).

## Raw fields

| Field | Type | Meaning | Observed values | Notes |
|---|---|---|---|---|
| ID | int | Unique listing identifier | 1 – 250,000 | Unique |
| State | text | Indian state / UT | 20 values | Each city maps to exactly one state |
| City | text | City | 42 values, ≈6,000 rows each | Model input; cohort key |
| Locality | text | Anonymised neighbourhood ID | `Locality_1` … `Locality_500` | The same ID appears in every city. The real unit is City + Locality (20,999 pairs, ≈12 rows each). Not a model input. |
| Property_Type | text | Dwelling type | Apartment, Independent House, Villa (≈⅓ each) | Cohort key |
| BHK | int | Bedrooms (bedroom-hall-kitchen count) | 1 – 5, uniform | Cohort key. Not linked to size (44.5% of 1-BHK listings exceed 3,000 sqft). |
| Size_in_SqFt | int | Built-up area | 500 – 5,000, uniform | |
| Price_in_Lakhs | float | Asking price in lakhs (1 lakh = ₹100,000) | 10 – 500, uniform | Asking, not transaction, price |
| Price_per_SqFt | float | Price ÷ area in **lakhs/sqft**, rounded to 2 dp | 0.00 – 0.99 | 2,534 rows round to 0.00, so it is recomputed in ₹ |
| Year_Built | int | Construction year | 1990 – 2023 | |
| Furnished_Status | text | Furnishing level | Furnished, Semi-furnished, Unfurnished | |
| Floor_No | int | Floor of the unit (0 = ground) | 0 – 30 | Above Total_Floors in 46.5% of rows |
| Total_Floors | int | Floors in the building | 1 – 30 | Villas/houses with > 4 floors: 57.9% |
| Age_of_Property | int | Years since construction | 2 – 35 | Exactly 2025 − Year_Built, so the reference year is 2025 |
| Nearby_Schools | int | Nearby schools (count or score, not specified) | 1 – 10 | |
| Nearby_Hospitals | int | Nearby hospitals (count or score) | 1 – 10 | |
| Public_Transport_Accessibility | text | Transport access | Low, Medium, High | |
| Parking_Space | text | Parking available | Yes, No | Number of spaces not recorded |
| Security | text | Security features present | Yes, No | |
| Amenities | text | Comma-separated amenity list | 325 strings = 31 sets of {Clubhouse, Garden, Gym, Playground, Pool} | Order varies, so it is parsed into flags |
| Facing | text | Direction the unit faces | East, North, South, West | |
| Owner_Type | text | Who lists the property | Owner, Builder, Broker | |
| Availability_Status | text | Construction status | Ready_to_Move, Under_Construction | 45.6% of rows are "under construction" yet built ≤ 2020 |

## Engineered fields (`clean()` / `build_model_frame()`)

| Field | Definition | Used by |
|---|---|---|
| Price_INR | Price_in_Lakhs × 100,000 | EDA, UI |
| Price_per_SqFt_INR | Price_INR ÷ Size_in_SqFt | Label, EDA, UI |
| has_clubhouse … has_pool | 1 if the amenity is listed | Models, EDA |
| amenity_count | Number of amenities (0–5) | Models, UI |
| Age_Years | 2025 − Year_Built | Models |
| Size_Band | <1,000 / 1,000–2,000 / … / 4,000+ sqft | EDA, UI |
| SqFt_per_BHK | Size ÷ BHK | Models |
| Floor_Position | Floor_No ÷ Total_Floors (NaN when floor > total) | Descriptive |
| Transport_Score | Low 0 · Medium 0.5 · High 1 | Descriptive |
| Infrastructure_Index | Mean of scaled schools, hospitals, transport (0–1). Equal weights are an assumption. | Descriptive only |
| log_price, log_size | Natural logs (make the ₹/sqft ratio linear for Logistic Regression) | Listing-track models |
| flag_floor_above_total, flag_uc_old_build, flag_house_many_floors, n_quality_flags | Consistency flags | UI listings table |

## Benchmark & label (fitted on training split only)

| Field | Definition |
|---|---|
| benchmark_pps | Training median ₹/sqft of the City × Property_Type × BHK cohort (≥ 30 rows; else City, then global median) |
| cohort_n | Training rows in the cohort (min 165 in the full data) |
| good_value | 1 if Price_per_SqFt_INR ≤ benchmark_pps, else 0 |

## Fields mentioned in the brief but **not** in the data
Crime rate, RERA status, infrastructure score and historical appreciation are absent and were not invented.
