"""
California Housing Price Predictor
Streamlit app serving an XGBoost regressor trained on the California Housing dataset.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Load the model bundle (cached so it loads once, not on every interaction)
# ----------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "xgb_california_housing.joblib"


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


try:
    bundle = load_bundle()
    model = bundle["model"]
    FEATURES = bundle["feature_names"]
    METRICS = bundle["metrics"]
except FileNotFoundError:
    st.error(
        f"Model file not found at `{MODEL_PATH}`.\n\n"
        "Make sure `xgb_california_housing.joblib` sits next to `app.py` in the repo."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------------
CITIES = {
    "Los Angeles":      (34.05, -118.24),
    "San Francisco":    (37.77, -122.42),
    "San Diego":        (32.72, -117.16),
    "San Jose":         (37.34, -121.89),
    "Sacramento":       (38.58, -121.49),
    "Fresno":           (36.75, -119.77),
    "Oakland":          (37.80, -122.27),
    "Bakersfield":      (35.37, -119.02),
    "Santa Barbara":    (34.42, -119.70),
    "Palo Alto":        (37.44, -122.14),
    "Enter manually…":  (None, None),
}

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🏠 California Housing Price Predictor")
st.markdown(
    "Predicts the **median house value** of a California census block group using an "
    "XGBoost regression model trained on the 1990 California Housing dataset."
)

c1, c2, c3 = st.columns(3)
c1.metric("Model", "XGBoost Regressor")
c2.metric("R² Score", f"{METRICS['R2']:.3f}")
c3.metric("RMSE", f"${METRICS['RMSE'] * 100_000:,.0f}")

st.divider()

# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
st.subheader("Enter block group details")

left, right = st.columns(2)

with left:
    st.markdown("**Location**")
    city = st.selectbox(
        "City", list(CITIES.keys()), index=0,
        help="Sets latitude and longitude. Location is one of the strongest predictors.",
    )
    if CITIES[city][0] is None:
        latitude = st.number_input("Latitude", 32.54, 41.95, 34.26, 0.01)
        longitude = st.number_input("Longitude", -124.35, -114.31, -118.49, 0.01)
    else:
        latitude, longitude = CITIES[city]
        st.caption(f"Coordinates: {latitude}, {longitude}")

    st.markdown("**Income & age**")
    med_inc = st.slider(
        "Median income (tens of thousands $)", 0.50, 15.00, 3.54, 0.05,
        help="Median household income in the block group. 3.54 means about $35,400.",
    )
    house_age = st.slider("Median house age (years)", 1, 52, 29, 1)

with right:
    st.markdown("**Household composition**")
    ave_rooms = st.slider("Average rooms per household", 1.0, 12.0, 5.23, 0.05)
    ave_bedrms = st.slider("Average bedrooms per household", 0.5, 3.0, 1.05, 0.01)
    ave_occup = st.slider("Average occupants per household", 1.0, 6.0, 2.82, 0.05)

    st.markdown("**Block group size**")
    population = st.number_input("Population", 3, 6000, 1166, 10)

# ----------------------------------------------------------------------------
# Predict
# ----------------------------------------------------------------------------
st.divider()

if st.button("Predict house value", type="primary", use_container_width=True):

    # Build a DataFrame with columns in the model's exact expected order.
    # Passing a raw array instead would let a wrong column order pass silently.
    input_df = pd.DataFrame([{
        "MedInc":     med_inc,
        "HouseAge":   house_age,
        "AveRooms":   ave_rooms,
        "AveBedrms":  ave_bedrms,
        "Population": population,
        "AveOccup":   ave_occup,
        "Latitude":   latitude,
        "Longitude":  longitude,
    }])[FEATURES]

    prediction = float(model.predict(input_df)[0])
    dollars = prediction * 100_000

    st.success(f"### Predicted median house value: ${dollars:,.0f}")

    # The training target was capped at $500,001, so the model cannot exceed it.
    if prediction >= 4.9:
        st.warning(
            "This prediction is at the model's ceiling. The original 1990 survey capped "
            "median house value at $500,001, so the model was never shown anything higher "
            "and cannot predict above it. Read this as **$500,000 or more**."
        )

    with st.expander("See the values sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}), use_container_width=True)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
with st.expander("About this model"):
    st.markdown(
        f"""
**Model:** XGBoost Regressor (100 estimators), trained on the California Housing dataset
(20,640 census block groups, 8 features).

**Performance on a held-out 20% test set**

| Metric | Value |
|---|---|
| RMSE | {METRICS['RMSE']:.4f} (${METRICS['RMSE'] * 100_000:,.0f}) |
| MAE | {METRICS['MAE']:.4f} (${METRICS['MAE'] * 100_000:,.0f}) |
| R² | {METRICS['R2']:.4f} |

Selected after comparing three models — Linear Regression (R² 0.446),
Random Forest (R² 0.804), and XGBoost (R² 0.835).

**Important limitations**

- The data is from the **1990 US Census**. These are 1990 prices and 1990 relationships
  between income and house value. This is a demonstration of an ML workflow, not a
  current property valuation tool.
- A **block group** is a small census area of roughly 600–3,000 people. The model predicts
  the median value across that whole area, not the price of any individual house.
- About 4.8% of the training data sits at the $500,001 ceiling described above.
        """
    )
