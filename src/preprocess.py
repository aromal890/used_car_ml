import hashlib
import json
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder

PREPROC_VERSION = "v1.2.0"

def compute_schema_hash(feature_columns: list[str]) -> str:
    payload = json.dumps({"version": PREPROC_VERSION, "columns": feature_columns}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["kmDriven"] = pd.to_numeric(
        df["kmDriven"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce"
    )
    df["kmDriven"] = df["kmDriven"].fillna(df["kmDriven"].median())
    df["Age"] = (2024 - df["Year"]).clip(lower=0)

    # Fix shorthand odometer entries (e.g., 65 -> 65,000 km) on used cars
    shorthand_mask = (df["kmDriven"] > 0) & (df["kmDriven"] < 500) & (df["Age"] >= 1)
    df.loc[shorthand_mask, "kmDriven"] = df.loc[shorthand_mask, "kmDriven"] * 1000
    df = df[(df["kmDriven"] >= 500) & (df["kmDriven"] <= 350000)].copy()

    df["km_per_year"] = df["kmDriven"] / (df["Age"] + 1)

    owner_map = {"test drive car": 0, "first": 1, "second": 2, "third": 3, "fourth": 4}
    df["Owner_encoded"] = df["Owner"].astype(str).str.lower().map(owner_map).fillna(1)

    info_col = df.get("AdditionInfo", pd.Series("", index=df.index)).fillna("").str.lower()
    df["is_top_model"] = info_col.str.contains("top|zxi|vx|sx|titanium|sunroof|leather").astype(int)
    df["has_insurance"] = info_col.str.contains("insurance|insured").astype(int)
    df["single_owner_note"] = (
        (df["Owner_encoded"] == 1) | info_col.str.contains("single|1st|first")
    ).astype(int)

    return df

def fit_transform_training(X_train: pd.DataFrame, y_train_log: pd.Series):
    te = TargetEncoder(cols=["model"], handle_unknown="value", handle_missing="value")
    X_train = X_train.copy()
    X_train["model_encoded"] = np.asarray(
        te.fit_transform(X_train[["model"]], y_train_log)
    ).ravel()
    X_train = X_train.drop(columns=["model"])
    X_train_enc = pd.get_dummies(X_train, columns=["Brand", "FuelType", "Transmission"])
    feature_columns = list(X_train_enc.columns)
    return X_train_enc, te, feature_columns

def transform_inference_row(raw_row: dict, te: TargetEncoder, feature_columns: list[str]) -> pd.DataFrame:
    """Transforms a single vehicle dict safely—even when Brand, Model, or FuelType are unseen."""
    age = max(0, 2024 - int(raw_row["Year"]))
    km_driven = float(raw_row["kmDriven"])
    owner_map = {"test drive car": 0, "first": 1, "second": 2, "third": 3, "fourth": 4}
    owner_encoded = owner_map.get(str(raw_row.get("Owner", "First")).lower(), 1)
    info_lower = str(raw_row.get("AdditionInfo", "")).lower()

    row_dict = {
        "Year": int(raw_row["Year"]),
        "Age": age,
        "kmDriven": km_driven,
        "km_per_year": km_driven / (age + 1),
        "Owner_encoded": owner_encoded,
        "is_top_model": int(raw_row.get("is_top_model", False) or any(k in info_lower for k in ["top", "zxi", "vx", "sx", "titanium", "sunroof", "leather"])),
        "has_insurance": int(raw_row.get("has_insurance", False) or any(k in info_lower for k in ["insurance", "insured"])),
        "single_owner_note": int(owner_encoded == 1 or any(k in info_lower for k in ["single", "1st", "first"])),
        "Brand": str(raw_row["Brand"]),
        "FuelType": str(raw_row["FuelType"]),
        "Transmission": str(raw_row["Transmission"]),
    }
    if "log_kmDriven" in feature_columns:
        row_dict["log_kmDriven"] = float(np.log1p(km_driven))

    df_row = pd.DataFrame([row_dict])
    df_row["model_encoded"] = np.asarray(
        te.transform(pd.DataFrame({"model": [str(raw_row["model"])]}))
    ).ravel()
    encoded = pd.get_dummies(df_row, columns=["Brand", "FuelType", "Transmission"])
    return encoded.reindex(columns=feature_columns, fill_value=0)