from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd

LOG_PATH = Path("logs/prediction_distribution.jsonl")

def log_prediction_event(inputs: dict, all_preds: dict, selected_model: str, artifacts: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    meta = artifacts.get("metadata", {})
    known_brands = set(artifacts.get("brand_models_map", {}).keys())
    known_models = set(artifacts.get("brand_models_map", {}).get(inputs["Brand"], []))

    pred_values = list(all_preds.values())
    spread_pct = (
        (max(pred_values) - min(pred_values)) / max(all_preds[selected_model], 1.0)
    ) * 100.0

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": meta.get("model_version", "v1.0-colab"),
        "preproc_version": meta.get("preproc_version", "v1.2.0"),
        "schema_hash": meta.get("schema_hash", "legacy"),
        "selected_model": selected_model,
        "brand": inputs["Brand"],
        "model": inputs["model"],
        "year": int(inputs["Year"]),
        "km_driven": float(inputs["kmDriven"]),
        "fuel_type": inputs["FuelType"],
        "transmission": inputs["Transmission"],
        "unseen_brand": inputs["Brand"] not in known_brands,
        "unseen_model": inputs["model"] not in known_models,
        "predicted_price_inr": round(float(all_preds[selected_model]), 2),
        "rf_price_inr": round(float(all_preds["Random Forest Regression"]), 2),
        "dt_price_inr": round(float(all_preds["Decision Tree Regression"]), 2),
        "ridge_price_inr": round(float(all_preds["Linear (Ridge) Regression"]), 2),
        "model_disagreement_pct": round(float(spread_pct), 2),
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

def generate_periodic_report(last_n_days: int = 7):
    if not LOG_PATH.exists():
        print("No prediction logs found yet.")
        return

    records = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(records)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    cutoff = pd.Timestamp.now(timezone.utc) - pd.Timedelta(days=last_n_days)
    recent = df[df["timestamp_utc"] >= cutoff]

    print(f"=== Prediction Distribution Report (Last {last_n_days} Days | n={len(recent)}) ===")
    if recent.empty:
        print("No predictions recorded in the selected window.")
        return

    prices_lakhs = recent["predicted_price_inr"] / 100000.0
    quantiles = np.percentile(prices_lakhs, [10, 25, 50, 75, 90])
    print(f"Mean Price      : ₹{prices_lakhs.mean():.2f}L (Std: ₹{prices_lakhs.std():.2f}L)")
    print(f"Quantiles (L)   : P10={quantiles[0]:.2f} | P25={quantiles[1]:.2f} | Median={quantiles[2]:.2f} | P75={quantiles[3]:.2f} | P90={quantiles[4]:.2f}")
    print(f"Avg Disagreement: {recent['model_disagreement_pct'].mean():.1f}% across RF / DT / Ridge")
    print(f"Unseen Brand %  : {recent['unseen_brand'].mean()*100:.2f}% | Unseen Model %: {recent['unseen_model'].mean()*100:.2f}%")
    print("\nTop Queried Brands:")
    print(recent["brand"].value_counts().head(5).to_string())

if __name__ == "__main__":
    generate_periodic_report(last_n_days=7)