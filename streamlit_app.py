import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import math

# ------------- CONFIG -------------
import os
API_URL = os.environ.get("API_URL", "https://5aif3j5udk.execute-api.ap-southeast-2.amazonaws.com/predict")
DATA_FILE  = "ev_test_canvas.csv"
TIMEOUT_SEC = 10
TITLE = "EV Ownership Probability (API Demo)"
# ----------------------------------

st.set_page_config(page_title="EV Prediction Demo", layout="wide")

# ---------- DATA LOADING ----------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    expected = ['household_id'] + [str(i) for i in range(48)] + ['weekday','rolling_std7']
    if list(df.columns) != expected:
        st.error(f"Column order mismatch.\nExpected:\n{expected}\nGot:\n{list(df.columns)}")
        st.stop()
    # Ensure native Python numeric types on extraction (we still cast individually later)
    return df

df = load_data(DATA_FILE)

# ---------- HELPERS ----------
def build_feature_vector(row):
    # Convert all numpy scalar types to native Python int/float
    half_hours = [float(row[str(i)]) for i in range(48)]
    weekday    = int(row['weekday'])
    rolling    = float(row['rolling_std7'])
    features   = [row['household_id']] + half_hours + [weekday, rolling]

    # Basic sanity checks
    if len(features) != 51:
        raise ValueError(f"Feature length {len(features)} != 51")

    # NaN check
    for v in half_hours + [rolling]:
        if isinstance(v, float) and math.isnan(v):
            raise ValueError("NaN detected in feature vector")

    return features

def call_api(feature_vector):
    payload = {"features": feature_vector}
    r = requests.post(API_URL, json=payload, timeout=TIMEOUT_SEC)
    r.raise_for_status()
    return r.json()

def extract_probability(pred_dict):
    candidates = [
        pred_dict.get("probability"),
        pred_dict.get("prob_1"),
        pred_dict.get("raw_prob"),
        pred_dict.get("prob_EV")
    ]
    for p in candidates:
        if p is None:
            continue
        if isinstance(p, str):
            try:
                p = float(p)
            except ValueError:
                continue
        if isinstance(p, (int, float)):
            return p
    return None

# ---------- UI LAYOUT ----------
st.title(TITLE)
st.caption("Local Streamlit → API Gateway → Lambda → SageMaker → DynamoDB")

house_ids = df['household_id'].unique()
with st.sidebar:
    st.header("Selection")
    selected = st.selectbox("Household ID", house_ids)
    subset = df[df.household_id == selected].reset_index(drop=True)
    row_idx = st.slider("Row index (day variant)", 0, len(subset)-1, 0)
    st.write(f"Total rows for {selected}: {len(subset)}")
    st.divider()
    st.markdown("**API URL**")
    st.code(API_URL, language="text")
    st.markdown("**Legend:** 0..47 = half-hour kWh, weekday=0(Mon)..6(Sun)")
    st.divider()
    if st.button("Predict EV Probability", type="primary"):
        row = subset.iloc[row_idx]
        try:
            features = build_feature_vector(row)
            result = call_api(features)
            pred = result.get("prediction", {})
            prob = extract_probability(pred)

            st.success("Prediction received")
            with st.expander("Raw prediction JSON"):
                st.json(pred)

            # Show probability
            if prob is not None:
                st.metric("P(EV)", f"{prob:.2%}")
            else:
                st.warning("Probability not found in response.")

            # Optional: feature vector preview
            with st.expander("Sent feature vector (first 10 elements)"):
                st.write(features[:10], "... total length:", len(features))

        except Exception as e:
            st.error(f"Request failed: {e}")

# ---------- MAIN PANEL ----------
row = subset.iloc[row_idx]

# Load shape plot
vals = [row[str(i)] for i in range(48)]
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(range(48), vals, marker='o', linewidth=1)
ax.set_xlabel("Half-hour slot (0 = 00:00–00:30)")
ax.set_ylabel("kWh")
ax.set_title(f"Daily Load Shape – {selected} (row {row_idx})")
ax.grid(alpha=0.3)
st.pyplot(fig)

col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.metric("Weekday", int(row['weekday']))
with col2:
    st.metric("7-day σ", f"{float(row['rolling_std7']):.2f}")
with col3:
    st.write("**Household ID:**", selected)
    st.write("Rows loaded:", len(df))

# Optional data preview
with st.expander("Data sample (first 5 rows)"):
    st.dataframe(df.head())