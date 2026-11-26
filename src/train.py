import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.preprocess import (
    PREPROC_VERSION,
    clean_and_engineer_features,
    compute_schema_hash,
    fit_transform_training,
)

REGISTERED_MODEL_NAME = "UsedCarPriceChampion"

def run_training_pipeline(
    data_path: str = "data/used_car_dataset.csv",
    selected_model_key: str = "Random Forest Regression",
    tracking_uri: str = "file:./mlruns",
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Used_Car_Price_Regression")

    raw_df = pd.read_csv(data_path)
    raw_df["AskPrice"] = pd.to_numeric(
        raw_df["AskPrice"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce",
    )
    raw_df = raw_df.dropna(subset=["AskPrice", "Brand", "model", "Year", "FuelType", "Transmission"])
    df = clean_and_engineer_features(raw_df)

    feature_cols_raw = [
        "Year", "Age", "kmDriven", "km_per_year", "Owner_encoded",
        "is_top_model", "has_insurance", "single_owner_note",
        "Brand", "model", "FuelType", "Transmission",
    ]
    X = df[feature_cols_raw]
    y = df["AskPrice"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_train_log = np.log1p(y_train)

    X_train, te, feature_columns = fit_transform_training(X_train_raw, y_train_log)
    schema_hash = compute_schema_hash(feature_columns)

    X_test = X_test_raw.copy()
    X_test["model_encoded"] = te.transform(X_test[["model"]])["model"]
    X_test = pd.get_dummies(X_test.drop(columns=["model"]), columns=["Brand", "FuelType", "Transmission"])
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    candidates = {
        "Linear (Ridge) Regression": (
            make_pipeline(StandardScaler(), Ridge(alpha=15.0)),
            {"alpha": 15.0, "scaler": "StandardScaler"},
        ),
        "Decision Tree Regression": (
            DecisionTreeRegressor(max_depth=14, min_samples_split=8, min_samples_leaf=5, random_state=42),
            {"max_depth": 14, "min_samples_split": 8, "min_samples_leaf": 5},
        ),
        "Random Forest Regression": (
            RandomForestRegressor(n_estimators=200, max_depth=22, min_samples_split=5, min_samples_leaf=4, random_state=42, n_jobs=1),
            {"n_estimators": 200, "max_depth": 22, "min_samples_split": 5, "min_samples_leaf": 4},
        ),
    }

    trained_models = {}
    metrics_summary = {}
    run_ids = {}

    for model_name, (estimator, params) in candidates.items():
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.set_tag("preproc_version", PREPROC_VERSION)
            mlflow.set_tag("schema_hash", schema_hash)
            mlflow.set_tag("algorithm", model_name)
            mlflow.log_params(params)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("num_features", len(feature_columns))

            estimator.fit(X_train, y_train_log)
            preds = np.clip(np.expm1(estimator.predict(X_test)), y_train.min(), y_train.max())

            r2 = float(r2_score(y_test, preds))
            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

            mlflow.log_metrics({"r2_score": r2, "mae_inr": mae, "rmse_inr": rmse})
            mlflow.sklearn.log_model(estimator, artifact_path="model")

            trained_models[model_name] = estimator
            metrics_summary[model_name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
            run_ids[model_name] = run.info.run_id

    selected_run_id = run_ids[selected_model_key]
    model_uri = f"runs:/{selected_run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered.version,
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=registered.version,
        key="preproc_version",
        value=PREPROC_VERSION,
    )

    brand_models_map = (
        df.groupby("Brand")["model"].unique().apply(lambda x: sorted(list(x))).to_dict()
    )

    artifacts = {
        "lr_model": trained_models["Linear (Ridge) Regression"],
        "dt_model": trained_models["Decision Tree Regression"],
        "rf_model": trained_models["Random Forest Regression"],
        "target_encoder": te,
        "feature_columns": feature_columns,
        "brand_models_map": brand_models_map,
        "fuel_types": sorted(df["FuelType"].unique().tolist()),
        "transmissions": sorted(df["Transmission"].unique().tolist()),
        "min_price": float(y_train.min()),
        "max_price": float(y_train.max()),
        "metrics": metrics_summary,
        "metadata": {
            "preproc_version": PREPROC_VERSION,
            "schema_hash": schema_hash,
            "selected_model": selected_model_key,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "model_version": f"v{registered.version}",
            "mlflow_run_id": selected_run_id,
        },
    }

    joblib.dump(artifacts, "car_price_artifacts.joblib", compress=3)
    print(f"Exported artifacts | Model: {REGISTERED_MODEL_NAME} v{registered.version} | Preproc: {PREPROC_VERSION} ({schema_hash})")

if __name__ == "__main__":
    run_training_pipeline()