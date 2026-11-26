from src.monitor import log_prediction_event
import os
from datetime import datetime
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM STYLING
# ---------------------------------------------------------
st.set_page_config(
    page_title="AutoValuate AI | Used Car Price Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .price-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        padding: 1.8rem;
        border-radius: 14px;
        color: white;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.2);
    }
    .price-title {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.85;
    }
    .price-value {
        font-size: 2.7rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
    .price-sub {
        font-size: 1.05rem;
        opacity: 0.92;
    }
</style>
""", unsafe_allow_html=True)

HISTORY_FILE = "prediction_history.csv"

# ---------------------------------------------------------
# 2. LOAD ARTIFACTS & INITIALIZE HISTORY
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    data = joblib.load('car_price_artifacts.joblib')
    # Single-thread inference avoids Windows joblib warnings and speeds up single-row predictions
    if hasattr(data.get('rf_model'), 'n_jobs'):
        data['rf_model'].n_jobs = 1
    return data

try:
    artifacts = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ `car_price_artifacts.joblib` not found! Place it in the same folder as `app.py`.")
    st.stop()

if "history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        try:
            st.session_state.history = pd.read_csv(HISTORY_FILE).to_dict("records")
        except Exception:
            st.session_state.history = []
    else:
        st.session_state.history = []

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def build_feature_row(brand, car_model, year, km_driven, fuel_type, transmission,
                      owner_label, is_top_model, has_insurance, addition_info):
    age = max(0, 2024 - year)
    km_per_year = km_driven / (age + 1)

    owner_map = {'test drive car': 0, 'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
    owner_encoded = owner_map.get(owner_label.lower(), 1)

    info_lower = addition_info.lower()
    top_flag = int(is_top_model or any(k in info_lower for k in ['top', 'zxi', 'vx', 'sx', 'titanium', 'creta', 'sunroof', 'leather']))
    ins_flag = int(has_insurance or any(k in info_lower for k in ['insurance', 'insured']))
    single_flag = int(owner_label == "First" or any(k in info_lower for k in ['single', '1st', 'first']))

    row_dict = {
        'Year': year,
        'Age': age,
        'kmDriven': float(km_driven),
        'km_per_year': km_per_year,
        'Owner_encoded': owner_encoded,
        'is_top_model': top_flag,
        'has_insurance': ins_flag,
        'single_owner_note': single_flag,
        'Brand': brand,
        'FuelType': fuel_type,
        'Transmission': transmission
    }

    # Include log_kmDriven automatically if the loaded model was trained with it
    if 'log_kmDriven' in artifacts['feature_columns']:
        row_dict['log_kmDriven'] = np.log1p(float(km_driven))

    raw_df = pd.DataFrame([row_dict])

    te = artifacts['target_encoder']
    raw_df['model_encoded'] = te.transform(pd.DataFrame({'model': [car_model]}))

    encoded_df = pd.get_dummies(raw_df, columns=['Brand', 'FuelType', 'Transmission'])
    final_X = encoded_df.reindex(columns=artifacts['feature_columns'], fill_value=0)
    return final_X, age, km_per_year

def predict_price(model_obj, input_df):
    pred_log = model_obj.predict(input_df)[0]
    pred_price = np.expm1(pred_log)
    return float(np.clip(pred_price, artifacts['min_price'], artifacts['max_price']))

def get_feature_importance_dfs(model_obj, feature_cols):
    """Extracts raw and grouped feature importances from tree-based models."""
    if not hasattr(model_obj, "feature_importances_"):
        return None, None

    importances = model_obj.feature_importances_
    raw_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance (%)": importances * 100
    }).sort_values("Importance (%)", ascending=False)

    # Group one-hot encoded columns into readable categories
    grouped = {}
    label_map = {
        "model_encoded": "Car Model Tier",
        "Year": "Manufacturing Year",
        "Age": "Vehicle Age",
        "kmDriven": "Odometer (km)",
        "log_kmDriven": "Odometer (Log Scale)",
        "km_per_year": "Annual Usage (km/yr)",
        "Owner_encoded": "Ownership History",
        "is_top_model": "Top Variant / Sunroof",
        "has_insurance": "Active Insurance",
        "single_owner_note": "Single-Owner Note"
    }

    for feat, imp in zip(feature_cols, importances):
        if feat.startswith("Brand_"):
            cat = "Brand (All Makes)"
        elif feat.startswith("FuelType_"):
            cat = "Fuel Type"
        elif feat.startswith("Transmission_"):
            cat = "Transmission"
        else:
            cat = label_map.get(feat, feat)
        grouped[cat] = grouped.get(cat, 0.0) + (imp * 100)

    grouped_df = pd.DataFrame({
        "Feature Category": list(grouped.keys()),
        "Importance (%)": list(grouped.values())
    }).sort_values("Importance (%)", ascending=False)

    return raw_imp, grouped_df

def save_prediction_record(record):
    st.session_state.history.insert(0, record)
    pd.DataFrame(st.session_state.history).to_csv(HISTORY_FILE, index=False)

# ---------------------------------------------------------
# 4. SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3202/3202926.png", width=68)
    st.title("AutoValuate AI")
    st.caption("Machine Learning Valuation Engine")
    meta = artifacts.get("metadata", {})
    st.caption(f"Model: `{meta.get('model_version', 'v1.0')}` | Preproc: `{meta.get('preproc_version', 'v1.2.0')}`")
    st.markdown("---")

    primary_model_name = st.selectbox(
        "🧠 Primary Prediction Model",
        ("Random Forest Regression", "Decision Tree Regression", "Linear (Ridge) Regression"),
        help="Random Forest provides the best overall accuracy on unseen test data."
    )

    m_stats = artifacts['metrics'][primary_model_name]
    st.markdown("### 📈 Active Model Accuracy")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("R² Score", f"{m_stats['R2']*100:.1f}%")
    col_s2.metric("Avg Error", f"₹{m_stats['MAE']/100000:.2f}L")

    st.markdown("---")
    st.metric("Saved Valuations", len(st.session_state.history))
    st.info("💡 **Tip:** Click **Save Valuation to History** after configuring a car to compare multiple vehicles side by side.")

# ---------------------------------------------------------
# 5. MAIN VEHICLE CONFIGURATION PANEL
# ---------------------------------------------------------
st.markdown('<div class="main-header">🚘 Smart Used Car Valuation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Configure vehicle details for real-time valuation, cross-model benchmarking, feature drivers, and saved history.</div>', unsafe_allow_html=True)

with st.container(border=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### 1️⃣ Make & Model")
        brands = sorted(list(artifacts['brand_models_map'].keys()))
        default_brand_idx = brands.index("Maruti Suzuki") if "Maruti Suzuki" in brands else 0
        brand = st.selectbox("Brand", brands, index=default_brand_idx)

        available_models = artifacts['brand_models_map'][brand]
        car_model = st.selectbox("Model", available_models)
        year = st.slider("Manufacturing Year", min_value=1995, max_value=2024, value=2019)

    with c2:
        st.markdown("#### 2️⃣ Usage & Powertrain")
        km_driven = st.number_input("Odometer (km driven)", min_value=500, max_value=350000, value=42000, step=2500)
        fuel_type = st.selectbox("Fuel Type", artifacts['fuel_types'])
        transmission = st.radio("Transmission", artifacts['transmissions'], horizontal=True)

    with c3:
        st.markdown("#### 3️⃣ Ownership & Condition")
        owner_label = st.selectbox("Ownership History", ["First", "Second", "Third", "Fourth", "Test Drive Car"])
        is_top_model = st.toggle("Top Trim / Premium Features (Sunroof, ZXi, SX, etc.)", value=False)
        has_insurance = st.toggle("Active Comprehensive Insurance", value=True)
        addition_info = st.text_input("Seller Notes (Optional)", placeholder="e.g. single owner, sunroof, new tires")

# Run live inference across all 3 models
input_X, car_age, km_per_yr = build_feature_row(
    brand, car_model, year, km_driven, fuel_type, transmission,
    owner_label, is_top_model, has_insurance, addition_info
)

model_map = {
    "Random Forest Regression": artifacts['rf_model'],
    "Decision Tree Regression": artifacts['dt_model'],
    "Linear (Ridge) Regression": artifacts['lr_model']
}

all_preds = {name: predict_price(m, input_X) for name, m in model_map.items()}
primary_price = all_preds[primary_model_name]
mae_margin = artifacts['metrics'][primary_model_name]['MAE']
low_est = max(artifacts['min_price'], primary_price - (mae_margin * 0.5))
high_est = primary_price + (mae_margin * 0.5)

st.markdown("<br>", unsafe_allow_html=True)
current_sig = (brand, car_model, year, km_driven, fuel_type, transmission, owner_label, primary_model_name)
if st.session_state.get("last_logged_sig") != current_sig:
    log_prediction_event(
        inputs={
            "Brand": brand, "model": car_model, "Year": year,
            "kmDriven": km_driven, "FuelType": fuel_type, "Transmission": transmission,
        },
        all_preds=all_preds,
        selected_model=primary_model_name,
        artifacts=artifacts,
    )
    st.session_state["last_logged_sig"] = current_sig
# ---------------------------------------------------------
# 6. FIVE-TAB ANALYTICS & HISTORY DASHBOARD
# ---------------------------------------------------------
history_count = len(st.session_state.history)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Valuation & Consensus",
    "📉 Depreciation Simulator",
    "🔍 Feature Importance",
    "🏆 Model Comparison",
    f"📜 Prediction History ({history_count})"
])

# --- TAB 1: VALUATION & SAVE TO HISTORY ---
with tab1:
    res_col1, res_col2 = st.columns([1.2, 1])

    with res_col1:
        st.markdown(f"""
        <div class="price-card">
            <div class="price-title">Estimated Market Value ({primary_model_name})</div>
            <div class="price-value">₹{primary_price:,.0f}</div>
            <div class="price-sub">≈ <b>₹{primary_price/100000:.2f} Lakhs</b> &nbsp;|&nbsp; Fair Range: ₹{low_est/100000:.2f}L – ₹{high_est/100000:.2f}L</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Vehicle Age", f"{car_age} yrs")
        m2.metric("Annual Usage", f"{km_per_yr:,.0f} km/yr",
                  delta="Low usage" if km_per_yr < 10000 else "Heavy usage",
                  delta_color="normal" if km_per_yr < 10000 else "inverse")
        m3.metric("Owner Tier", owner_label)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Valuation to History", type="primary", use_container_width=True):
            record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Brand": brand,
                "Model": car_model,
                "Year": year,
                "Odometer (km)": km_driven,
                "Fuel": fuel_type,
                "Transmission": transmission,
                "Owner": owner_label,
                "Selected Model": primary_model_name,
                "Estimated Price (₹)": round(primary_price, 0),
                "Price (Lakhs)": round(primary_price / 100000, 2),
                "Random Forest (₹)": round(all_preds["Random Forest Regression"], 0),
                "Decision Tree (₹)": round(all_preds["Decision Tree Regression"], 0),
                "Ridge Regression (₹)": round(all_preds["Linear (Ridge) Regression"], 0),
            }
            save_prediction_record(record)
            st.toast(f"Saved {brand} {car_model} (₹{primary_price/100000:.2f}L) to history!", icon="✅")
            st.rerun()

    with res_col2:
        st.markdown("#### Live Cross-Model Consensus")
        comp_df = pd.DataFrame({
            "Model": list(all_preds.keys()),
            "Estimated Price (₹)": list(all_preds.values())
        }).set_index("Model")

        st.bar_chart(comp_df, color="#2563EB", height=240)

        for m_name, val in all_preds.items():
            st.write(f"**{m_name}:** ₹{val:,.0f} (`₹{val/100000:.2f} L`)")

# --- TAB 2: DEPRECIATION SIMULATOR ---
with tab2:
    st.subheader(f"📉 How Usage & Age Impact Your {brand} {car_model}")
    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown("**Price vs. Odometer Reading (km)**")
        km_range = list(range(10000, 210000, 20000))
        km_prices = []
        for k in km_range:
            sim_X, _, _ = build_feature_row(
                brand, car_model, year, k, fuel_type, transmission,
                owner_label, is_top_model, has_insurance, addition_info
            )
            km_prices.append(predict_price(model_map[primary_model_name], sim_X) / 100000)

        km_df = pd.DataFrame({"Kilometers Driven": km_range, "Est. Price (Lakhs ₹)": km_prices}).set_index("Kilometers Driven")
        km_df["Est. Price (Lakhs ₹)"] = km_df["Est. Price (Lakhs ₹)"].cummin()
        st.line_chart(km_df, color="#10B981")

    with sim_col2:
        st.markdown("**Price by Manufacturing Year**")
        yr_range = list(range(max(2005, year - 5), min(2025, year + 6)))
        yr_prices = []
        for y_sim in yr_range:
            sim_X, _, _ = build_feature_row(
                brand, car_model, y_sim, km_driven, fuel_type, transmission,
                owner_label, is_top_model, has_insurance, addition_info
            )
            yr_prices.append(predict_price(model_map[primary_model_name], sim_X) / 100000)

        yr_df = pd.DataFrame({"Model Year": [str(y) for y in yr_range], "Est. Price (Lakhs ₹)": yr_prices}).set_index("Model Year")
        yr_df["Est. Price (Lakhs ₹)"] = yr_df["Est. Price (Lakhs ₹)"].cummax()
        st.line_chart(yr_df, color="#F59E0B")

# --- TAB 3: FEATURE IMPORTANCE ---
with tab3:
    st.subheader("🔍 What Drives Used Car Prices?")
    st.write("Relative importance of vehicle attributes learned during training:")

    imp_model_choice = st.radio(
        "Inspect Feature Importance From:",
        ["Random Forest Regression", "Decision Tree Regression"],
        horizontal=True
    )

    raw_imp_df, grouped_imp_df = get_feature_importance_dfs(
        model_map[imp_model_choice],
        artifacts['feature_columns']
    )

    if grouped_imp_df is not None:
        fi_col1, fi_col2 = st.columns(2)

        with fi_col1:
            st.markdown("#### Grouped Feature Impact (%)")
            st.bar_chart(
                grouped_imp_df.set_index("Feature Category")["Importance (%)"],
                color="#2563EB",
                height=360
            )

        with fi_col2:
            st.markdown("#### Top 12 Individual Model Features")
            top12 = raw_imp_df.head(12).copy()
            top12["Importance (%)"] = top12["Importance (%)"].round(2)
            st.dataframe(top12, use_container_width=True, hide_index=True, height=360)

# --- TAB 4: COMPREHENSIVE MODEL COMPARISON ---
with tab4:
    st.subheader("🏆 Side-by-Side Model Comparison")
    st.write("Comparing accuracy, error margins, and current vehicle predictions across all three trained algorithms:")

    bench_rows = []
    for name, m in artifacts['metrics'].items():
        bench_rows.append({
            "Model": name,
            "R² Accuracy Score": round(m['R2'], 4),
            "R² (%)": f"{m['R2']*100:.2f}%",
            "MAE (₹)": round(m['MAE'], 2),
            "RMSE (₹)": round(m['RMSE'], 2),
            "Current Car Estimate (₹)": round(all_preds[name], 0),
            "Current Car (Lakhs)": f"₹{all_preds[name]/100000:.2f} L"
        })
    bench_df = pd.DataFrame(bench_rows)

    st.dataframe(
        bench_df[["Model", "R² (%)", "MAE (₹)", "RMSE (₹)", "Current Car Estimate (₹)", "Current Car (Lakhs)"]],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    mc_col1, mc_col2 = st.columns(2)

    with mc_col1:
        st.markdown("#### R² Accuracy Comparison (Higher is Better)")
        r2_chart_df = bench_df.set_index("Model")[["R² Accuracy Score"]]
        st.bar_chart(r2_chart_df, color="#10B981", height=280)

    with mc_col2:
        st.markdown("#### Error Comparison: MAE vs. RMSE in ₹ (Lower is Better)")
        err_chart_df = bench_df.set_index("Model")[["MAE (₹)", "RMSE (₹)"]]
        st.bar_chart(err_chart_df, height=280)

# --- TAB 5: PREDICTION HISTORY ---
with tab5:
    st.subheader("📜 Saved Valuation History")

    if len(st.session_state.history) == 0:
        st.info("No predictions saved yet. Go to **💰 Valuation & Consensus** and click **💾 Save Valuation to History** to log vehicles here.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        # Summary KPIs across saved cars
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Saved Cars", len(hist_df))
        h2.metric("Average Valuation", f"₹{hist_df['Estimated Price (₹)'].mean()/100000:.2f} Lakhs")
        h3.metric("Highest Valued Car", f"₹{hist_df['Estimated Price (₹)'].max()/100000:.2f} Lakhs")

        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # Export & Clear Controls
        btn_col1, btn_col2, _ = st.columns([1.2, 1.2, 3])
        with btn_col1:
            csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download History (CSV)",
                data=csv_bytes,
                file_name=f"car_valuations_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with btn_col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                if os.path.exists(HISTORY_FILE):
                    os.remove(HISTORY_FILE)
                st.rerun()