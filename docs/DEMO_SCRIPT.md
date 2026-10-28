# Demo Script (6–8 minutes)

**Setup:** app running (`streamlit run app.py`), MLflow UI in a second tab
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`), `reports/eda_report.html` open in a third.

---

### 1. Hook — 30 s · *Executive Dashboard*
> "PropWise AI answers two questions an investor asks about a listing: is it priced well against similar homes,
> and what might it be worth in five years? The rule I followed was that every number has to be traceable,
> either to the data or to a cited source."

Point at the KPI row: 250,000 listings, 42 cities, 630 comparable cohorts.

### 2. The finding that shaped everything — 75 s · *Dashboard → EDA report*
> "Before modelling, I profiled the data. It's clean, with no missing values or duplicates, but price is unrelated
> to size, location, amenities or transport. Every correlation is below 0.01."

Show EDA §4 (size vs price, r ≈ 0) and §10 (top localities match a random shuffle).
> "I also found internal contradictions: 46% of rows have a floor number above the building height. So the
> data looks synthetic. Instead of hiding that, I designed the product around it."

### 3. Analyze a property — 90 s · *Investment Analyzer*
Use the defaults (3 BHK apartment, Bangalore, 1,800 sqft, ₹140 L). Click **Analyze property**.
> "The verdict is relative value: ₹7,778 per sqft against a comparable median of ₹9,537, which is 18% below.
> Comparables are the same city, type and BHK, using medians from the training split only."

- **Value vs comparables:** the listing's position in the cohort distribution. The "implied price" is labelled
  as *not a valuation* because it scores R² −0.74.
- **Five-year scenarios:** 2 / 4 / 6% bracket the latest RBI (3.6%) and NHB (5.0%) indices. Open the
  assumptions panel to show they are editable. Point out the list of excluded costs.
- **Factors and Limitations:** rule-based factors, context items kept out of the verdict, and plain-language caveats.

Then change the size to 9,000 sqft and show the **extrapolation warning**.

### 4. Compare — 45 s · *Property Comparison*
Add IDs 1, 2, 3. Show the highlighted table and the shared growth slider.
> "There's deliberately no 'winner'. With identical growth rates, projected gains just scale with price. The real
> difference is each listing against its own comparables."

### 5. Model honesty — 90 s · *Model Insights*
> "I trained Logistic Regression, Random Forest and XGBoost, tracked in MLflow. The champion hits 0.999 AUC,
> and I state openly that this is expected: the label is a rule on price and area, so the model is reproducing
> it. To show what is actually learnable, the attributes-only track removes price and lands at 0.78, almost
> entirely from area."

Show calibration (ECE 0.003, which is why a probability is displayed), the confusion matrix and permutation importance.
Switch to the **Valuation signal test**: four regressors at R² ≈ 0.
> "That's why there is no fake 'predicted price' and no fabricated five-year labels."

Switch to the MLflow UI tab: two experiments, 11 runs, registered model `@champion`.

### 6. Engineering — 45 s
- One `build_model_frame()` inside the saved pipeline, so training and inference match (a test enforces it).
- 65 pytest tests, including headless runs of every page.
- `scripts.run_pipeline` reproduces everything from the raw CSV. The deployed app needs only a ≈2.9 MB model bundle, stored in portable formats.

### 7. Close — 20 s
> "With real transaction histories, the same architecture becomes a genuine forecaster: swap the scenario
> engine for a time-validated model and the cohort rule for realised returns. Until then, PropWise is
> honest about what the data can and can't say."

---

**Likely questions**
- *Why not predict the five-year price with XGBoost?* There are no future prices to learn from or validate against.
- *Isn't 0.999 AUC suspicious?* Yes, which is why it's labelled as rule reproduction and paired with the attributes-only track.
- *Why the median as the threshold?* It's robust to the ₹/sqft right tail and yields balanced classes.
- *Why 4% baseline?* It sits between RBI 3.6% and NHB 5.0%, and users can change it.
