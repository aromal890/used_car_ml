import joblib
import numpy as np
import pytest
from src.preprocess import compute_schema_hash, transform_inference_row

@pytest.fixture(scope="module")
def artifacts():
    data = joblib.load("car_price_artifacts.joblib")
    if hasattr(data.get("rf_model"), "n_jobs"):
        data["rf_model"].n_jobs = 1
    return data

def predict_for_profile(artifacts, profile: dict) -> float:
    X_row = transform_inference_row(
        profile,
        artifacts["target_encoder"],
        artifacts["feature_columns"],
    )
    assert list(X_row.columns) == artifacts["feature_columns"]
    assert not X_row.isna().any().any(), "NaN values detected in transformed features"

    pred_log = artifacts["rf_model"].predict(X_row)[0]
    return float(np.clip(np.expm1(pred_log), artifacts["min_price"], artifacts["max_price"]))

@pytest.mark.parametrize(
    "profile_name, profile",
    [
        (
            "recent_budget_hatchback",
            {
                "Brand": "Maruti Suzuki", "model": "Swift", "Year": 2021,
                "kmDriven": 28000, "FuelType": "Petrol", "Transmission": "Manual",
                "Owner": "First", "AdditionInfo": "single owner insured",
            },
        ),
        (
            "high_mileage_diesel_suv",
            {
                "Brand": "Toyota", "model": "Fortuner", "Year": 2018,
                "kmDriven": 125000, "FuelType": "Diesel", "Transmission": "Automatic",
                "Owner": "Second", "AdditionInfo": "top model sunroof",
            },
        ),
        (
            "older_multi_owner_sedan",
            {
                "Brand": "Honda", "model": "City", "Year": 2012,
                "kmDriven": 110000, "FuelType": "Petrol", "Transmission": "Manual",
                "Owner": "Third", "AdditionInfo": "",
            },
        ),
        (
            "unseen_brand_and_model",
            {
                "Brand": "Rivian_UnseenBrand", "model": "R1S_UnseenModel", "Year": 2023,
                "kmDriven": 12000, "FuelType": "Petrol", "Transmission": "Automatic",
                "Owner": "First", "AdditionInfo": "top trim",
            },
        ),
        (
            "unseen_fuel_and_transmission",
            {
                "Brand": "Hyundai", "model": "Creta", "Year": 2022,
                "kmDriven": 20000, "FuelType": "Hydrogen_UnseenFuel",
                "Transmission": "QuadClutch_Unseen", "Owner": "First", "AdditionInfo": "",
            },
        ),
    ],
)
def test_representative_and_unseen_profiles(artifacts, profile_name, profile):
    price = predict_for_profile(artifacts, profile)
    assert np.isfinite(price), f"{profile_name} produced non-finite prediction"
    assert artifacts["min_price"] <= price <= artifacts["max_price"]

def test_schema_hash_consistency(artifacts):
    meta = artifacts.get("metadata", {})
    if "schema_hash" in meta:
        assert meta["schema_hash"] == compute_schema_hash(artifacts["feature_columns"])