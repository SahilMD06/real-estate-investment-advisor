# Deployment Guide

## 1. What the deployed app needs

| Needed at runtime | Size | In git? |
|---|---|---|
| `app.py`, `views/`, `src/`, `assets/`, `.streamlit/config.toml` | small | yes |
| `data/processed/housing_clean.parquet` | ≈9 MB | **yes** (the app reads this, not the CSV) |
| `data/processed/data_quality_report.json` | 2 KB | yes |
| `models/` (classifier.joblib, classifier_xgb.json, cohort_stats.joblib, metadata.json, evaluation.json) | ≈2.9 MB | **yes** |
| `requirements.txt` | — | yes |

**Not committed** (see `.gitignore`): `data/raw/*` (the 41 MB CSV and any archive of it; regenerable input),
`mlflow.db` and `mlartifacts/` (≈45 MB local tracking store), `.venv/`, `.streamlit/secrets.toml`.
Everything committed is far below GitHub's 100 MB per-file limit, so Git LFS is not needed.

If you later need to share the raw CSV or MLflow artefacts, use a GitHub Release asset, Google Drive or an object
store (S3, GCS, Azure Blob), not the repository.

## 2. GitHub setup (Windows PowerShell)

```powershell
cd C:\Labmentix\real-estate-investment-advisor
git init
git add .
git status            # confirm: no .venv, no data/raw/*.csv, no mlflow.db
git commit -m "PropWise AI: real estate investment advisor"
git branch -M main
# Create an empty repo on github.com (no README), then:
git remote add origin https://github.com/<your-username>/real-estate-investment-advisor.git
git push -u origin main
```

## 3. Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. **Create app** → choose the repository, branch `main`, main file `app.py`.
3. **Advanced settings → Python version: 3.13** (the version the bundle was trained with; pickled
   scikit-learn/XGBoost models are safest on the same Python and library versions).
4. Deploy. The first build installs `requirements.txt` (a few minutes).

No secrets are required. If you add any (e.g. a remote MLflow URI), put them in the app's **Secrets** panel,
never in git:

```toml
# .streamlit/secrets.toml  (local only, git-ignored)
MLFLOW_TRACKING_URI = "https://your-mlflow-server"
```

`src/utils/config.py` reads paths from environment variables (`PROPWISE_MODELS_DIR`, `PROPWISE_PROCESSED_DATA`,
`PROPWISE_RAW_DATA`, `MLFLOW_TRACKING_URI`), so the code never hard-codes machine-specific paths.

### Other hosts
Any host that can run `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` works (Render,
Railway, Hugging Face Spaces with the Streamlit SDK, a VM). Install with `pip install -r requirements.txt`.

## 4. MLflow

- **Local (default):** training writes to `sqlite:///mlflow.db` with artefacts in `mlartifacts/`.
  Browse with:
  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
  ```
  Experiments: `propwise-investment-classification`, `propwise-valuation-signal-test`.
  Registered model: `propwise-investment-classifier` with alias `@champion`.
- **Load the registered model elsewhere:**
  ```python
  import mlflow
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  model = mlflow.pyfunc.load_model("models:/propwise-investment-classifier@champion")
  ```
- **Remote server:** set `MLFLOW_TRACKING_URI` before `python -m scripts.train_models`.
- **In the cloud app:** MLflow is not installed or needed. The app reads `models/`, and Model Insights
  shows the run IDs and registry version recorded at training time.

## 5. Updating the model
```bash
python -m scripts.run_pipeline   # retrain, re-register (new version → @champion), rewrite models/
pytest
git add models data/processed reports
git commit -m "Retrain models" && git push   # Streamlit Cloud redeploys automatically
```

## 6. Common failures

| Symptom | Cause | Fix |
|---|---|---|
| "Dataset unavailable … housing_clean.parquet" | Parquet not committed or path changed | Run `python -m scripts.prepare_data` and commit `data/processed/` |
| "Models unavailable … Run scripts.train_models" | `models/` missing | Train locally and commit `models/` |
| `ModuleNotFoundError: src` | App started from another folder | Run `streamlit run app.py` from the project root (Cloud does this automatically) |
| Unpickling / `AttributeError` loading `classifier.joblib` | Different scikit-learn/XGBoost version | Keep `requirements.txt` pins; retrain after upgrading libraries |
| `XGBoostError: input stream corrupted` | An XGBoost model pickled on another OS (old bundle format) | Pull the current `models/` folder (XGBoost stored as `classifier_xgb.json`) or run `python -m scripts.train_models` |
| Build fails on `pandas==3.1.0` | Python < 3.11 selected | Choose Python 3.13 in Advanced settings |
| `Activate.ps1 cannot be loaded` (Windows) | Execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `streamlit` not recognised | venv not active | Activate `.venv` or run `python -m streamlit run app.py` |
| MLflow "untrusted types" error when logging | skops safety check | The trusted list lives in `scripts/train_models.py` (`SKOPS_TRUSTED`); only add types you created |
| Port 8501 busy | Another Streamlit running | `streamlit run app.py --server.port 8502` |
| App slow on first load | Parquet + bundle loading | Cached after the first load (`st.cache_data` / `st.cache_resource`) |
