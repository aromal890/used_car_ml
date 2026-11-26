# AutoValuate AI — Used Car Price MLOps Pipeline

End-to-end used car valuation application built with Scikit-Learn, Streamlit, MLflow, Pytest, and Docker Compose.

## Features
- **Multi-Model Valuation (`app.py`):** Compares Random Forest, Decision Tree, and Ridge Regression with smoothed depreciation simulators and feature importance charts.
- **Versioned Preprocessing (`src/preprocess.py`):** Deterministic SHA-256 schema hashing and safe inference fallbacks for unseen brands, models, fuel types, and transmissions.
- **Experiment Tracking & Registry (`src/train.py`):** Logs hyperparameters and metrics to MLflow and registers the `@champion` model version.
- **Automated Gatekeeping (`tests/`):** Pytest suite validates representative vehicle profiles and unseen-category edge cases during Docker builds.
- **Prediction Drift Monitoring (`src/monitor.py`):** Logs live inference distributions and cross-model disagreement to `logs/prediction_distribution.jsonl`.

## Quickstart
```bash
docker compose up --build -d