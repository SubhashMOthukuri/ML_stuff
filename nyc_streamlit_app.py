"""
NYC Airbnb Price Predictor — Streamlit UI
Connects to the NYC Airbnb FastAPI at port 8001.
"""

import os
import pickle
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

API_URL  = os.getenv("NYC_API_URL", "http://localhost:8001")
MODEL_DIR = Path(__file__).resolve().parent / "models" / "nyc"

# ── borough centroids (lat/lon) ───────────────────────────────────────────────
BOROUGH_COORDS = {
    "Manhattan":     (40.7831, -73.9712),
    "Brooklyn":      (40.6782, -73.9442),
    "Queens":        (40.7282, -73.7949),
    "Bronx":         (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
}


@st.cache_resource
def _load_neighbourhoods() -> list[str]:
    try:
        with open(MODEL_DIR / "nyc_neighbourhood_means.pkl", "rb") as f:
            d = pickle.load(f)
        return ["(use borough average)"] + sorted(d["means"].keys())
    except Exception:
        return ["(use borough average)"]


@st.cache_data(ttl=60)
def _fetch_model_info() -> dict:
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def _fetch_metrics() -> dict:
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NYC Airbnb Price Predictor",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main { padding: 0 1rem; }
    h1 { font-size: 2.4rem; font-weight: 700; color: #c0392b; margin-bottom: 0.3rem; }
    h2 { font-size: 1.4rem; font-weight: 600; color: #2c3e50; margin-top: 1.5rem; }
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #c0392b;
        margin-bottom: 1rem;
    }
    .result-section {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        margin-top: 1.5rem;
    }
    .result-price { font-size: 3.5rem; font-weight: 800; margin: 0.5rem 0; }
    .result-sub   { font-size: 1rem; opacity: 0.85; }
    .stButton > button {
        width: 100%; padding: 1rem;
        font-size: 1.1rem; font-weight: 600;
        border-radius: 8px; border: none;
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color: white;
    }
    .footer {
        text-align: center; padding: 2rem 0;
        color: #888; font-size: 0.85rem;
        border-top: 1px solid #e0e0e0; margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────

col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("# 🗽 NYC Airbnb Price Predictor")
    st.markdown("**XGBoost model trained on 20,485 NYC listings — April 2026 snapshot**")

with col_badge:
    info = _fetch_model_info()
    r2   = info.get("r2_test", "—")
    mae  = info.get("mae_dollar", "—")
    st.markdown(f"""
    <div style="text-align:right; padding:1rem; background:#f0f0f0; border-radius:8px;">
        <div style="font-size:0.8rem; color:#888;">Model Performance</div>
        <div style="font-size:1.4rem; font-weight:700; color:#c0392b;">R² = {r2}</div>
        <div style="font-size:0.85rem; color:#666;">MAE ≈ ${mae}/night</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── NEIGHBOURHOODS ────────────────────────────────────────────────────────────

NEIGHBOURHOODS = _load_neighbourhoods()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LISTING BASICS
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("## 🏠 Listing Details")
st.markdown('<div class="input-section">', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Type & Location**")
    room_type = st.selectbox(
        "Room Type",
        ["Entire home/apt", "Private room", "Hotel room", "Shared room"],
        help="Entire home/apt is typically the highest priced",
    )
    borough = st.selectbox(
        "Borough",
        list(BOROUGH_COORDS.keys()),
        help="Manhattan is the most expensive borough",
    )
    neighbourhood = st.selectbox(
        "Neighbourhood (optional)",
        NEIGHBOURHOODS,
        help="Improves accuracy — leave as 'use borough average' if unsure",
    )

with c2:
    st.markdown("**Capacity**")
    accommodates = st.slider("Guests", 1, 16, 2)
    bedrooms     = st.slider("Bedrooms", 0, 10, 1)
    bathrooms    = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    is_private_bath = st.checkbox("Private bathroom", value=True,
                                   help="Shared bathrooms lower the predicted price")

with c3:
    st.markdown("**Host & Stay Rules**")
    host_is_superhost    = st.checkbox("Superhost", value=False)
    host_listings_count  = st.slider("Host's total listings", 1, 50, 1)
    minimum_nights       = st.slider("Minimum nights", 1, 30, 2)

st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REVIEWS
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("## ⭐ Reviews")
st.markdown('<div class="input-section">', unsafe_allow_html=True)

rc1, rc2 = st.columns(2)

with rc1:
    number_of_reviews = st.slider("Total reviews", 0, 500, 30)
    reviews_per_month = st.slider("Reviews per month", 0.0, 15.0, 1.2, step=0.1)
    review_scores_rating = st.slider("Overall rating (0–5)", 0.0, 5.0, 4.7, step=0.1)

with rc2:
    with st.expander("Detailed review scores (optional — defaults to overall rating)"):
        review_scores_accuracy      = st.slider("Accuracy",      0.0, 5.0, review_scores_rating, step=0.1, key="acc")
        review_scores_cleanliness   = st.slider("Cleanliness",   0.0, 5.0, review_scores_rating, step=0.1, key="cln")
        review_scores_checkin       = st.slider("Check-in",      0.0, 5.0, review_scores_rating, step=0.1, key="chk")
        review_scores_communication = st.slider("Communication", 0.0, 5.0, review_scores_rating, step=0.1, key="com")
        review_scores_location      = st.slider("Location",      0.0, 5.0, review_scores_rating, step=0.1, key="loc")
        review_scores_value         = st.slider("Value",         0.0, 5.0, review_scores_rating, step=0.1, key="val")

st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AMENITIES
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("## 🛎 Amenities")
st.markdown('<div class="input-section">', unsafe_allow_html=True)

ac1, ac2 = st.columns([1, 2])

with ac1:
    amenity_count = st.slider("Total amenities listed", 0, 100, 20,
                               help="More amenities correlates with higher price tier")

with ac2:
    st.markdown("**Key amenities**")
    am1, am2, am3 = st.columns(3)
    has_gym              = am1.checkbox("Gym")
    has_elevator         = am1.checkbox("Elevator")
    has_air_conditioning = am2.checkbox("Air conditioning")
    has_washer           = am2.checkbox("Washer")
    has_dryer            = am3.checkbox("Dryer")
    has_pool             = am3.checkbox("Pool")

st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ════════════════════════════════════════════════════════════════════════════════

st.divider()

_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🔮 Predict Nightly Price", use_container_width=True, type="primary")

if predict_clicked:
    lat, lon = BOROUGH_COORDS[borough]
    neigh_val = "" if neighbourhood == "(use borough average)" else neighbourhood

    payload = {
        "accommodates":              accommodates,
        "bedrooms":                  float(bedrooms),
        "bathrooms":                 float(bathrooms),
        "is_private_bath":           is_private_bath,
        "room_type":                 room_type,
        "borough":                   borough,
        "neighbourhood":             neigh_val,
        "latitude":                  lat,
        "longitude":                 lon,
        "minimum_nights":            minimum_nights,
        "host_is_superhost":         host_is_superhost,
        "host_listings_count":       host_listings_count,
        "number_of_reviews":         number_of_reviews,
        "number_of_reviews_ltm":     min(number_of_reviews, 12),
        "reviews_per_month":         reviews_per_month,
        "review_scores_rating":      review_scores_rating,
        "review_scores_accuracy":    review_scores_accuracy,
        "review_scores_cleanliness": review_scores_cleanliness,
        "review_scores_checkin":     review_scores_checkin,
        "review_scores_communication": review_scores_communication,
        "review_scores_location":    review_scores_location,
        "review_scores_value":       review_scores_value,
        "amenity_count":             amenity_count,
        "has_gym":                   has_gym,
        "has_elevator":              has_elevator,
        "has_dryer":                 has_dryer,
        "has_air_conditioning":      has_air_conditioning,
        "has_washer":                has_washer,
        "has_pool":                  has_pool,
    }

    with st.spinner("Predicting..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)

            if resp.status_code == 200:
                result = resp.json()

                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                st.markdown("### Predicted Nightly Price")

                pr1, pr2 = st.columns(2)
                with pr1:
                    st.markdown(
                        f'<div class="result-price">{result["price_str"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="result-sub">Model: {result["model"]} &nbsp;|&nbsp; R² = {result["r2_test"]}</div>',
                        unsafe_allow_html=True,
                    )
                with pr2:
                    mae_d = info.get("mae_dollar", 58)
                    lo = max(0, result["price_usd"] - mae_d)
                    hi = result["price_usd"] + mae_d
                    st.markdown(
                        f'<div style="padding-top:1rem;">'
                        f'<div style="font-size:1.1rem; font-weight:600;">Confidence range</div>'
                        f'<div style="font-size:1.8rem; font-weight:700;">${lo:,.0f} – ${hi:,.0f}</div>'
                        f'<div class="result-sub">±MAE of ${mae_d}/night</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

                # ── summary cards ─────────────────────────────────────────────
                st.markdown("### Listing Summary")
                s1, s2, s3, s4, s5 = st.columns(5)
                cards = [
                    (s1, "Borough",    borough),
                    (s2, "Room Type",  room_type.replace("/", "/​")),
                    (s3, "Guests",     accommodates),
                    (s4, "Bedrooms",   bedrooms),
                    (s5, "Rating",     f"{review_scores_rating}/5"),
                ]
                for col, label, val in cards:
                    col.markdown(
                        f'<div style="text-align:center;padding:1rem;background:#f8f9fa;border-radius:8px;">'
                        f'<div style="font-size:0.85rem;color:#888;">{label}</div>'
                        f'<div style="font-size:1.5rem;font-weight:700;color:#c0392b;">{val}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.session_state["last_price"] = result["price_usd"]

            elif resp.status_code == 422:
                detail = resp.json().get("detail", "Validation error")
                st.error(f"Invalid input: {detail}")
            else:
                st.error(f"API error {resp.status_code}: {resp.text[:200]}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Make sure it is running: `uvicorn src.new_york_workflow.nyc_api:app --port 8001`")
        except requests.exceptions.Timeout:
            st.error("Request timed out.")
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

# ════════════════════════════════════════════════════════════════════════════════
# MODEL INFO + LIVE METRICS
# ════════════════════════════════════════════════════════════════════════════════

st.divider()

info_col, how_col = st.columns(2)

with info_col:
    st.markdown("### Model Information")
    if info:
        st.markdown(f"""
**Algorithm:** {info.get("model_type", "XGBRegressor")}
**Features:** {info.get("n_features", 58)} engineered features
**Neighbourhoods:** {info.get("n_neighbourhoods", 219)} NYC areas with target encoding

**Test-set performance:**
- R² = {info.get("r2_test", "—")}
- MAE = ${info.get("mae_dollar", "—")}/night
- MAPE = {info.get("mape_pct", "—")}%

**Training data:** {info.get("training_rows", "—"):,} listings &nbsp;|&nbsp; {info.get("price_cap_used", "")}
        """)
    else:
        st.warning("Could not load model info — is the API running?")

with how_col:
    st.markdown("### How the model works")
    st.markdown("""
The model applies **10 feature engineering steps** to your inputs before predicting:

1. **Log transforms** on skewed fields (reviews, minimum nights, accommodates)
2. **Borough one-hot encoding** — Bronx as reference category
3. **Neighbourhood target encoding** — mean log-price for 219 NYC areas
4. **Private bath flag** — the #1 most important feature (importance=0.26)
5. **Polynomial features** — accommodates², bedrooms²
6. **Interaction features** — accommodates × rating, bedrooms × rating
7. **Ratio features** — reviews/bedroom, distance from Midtown
8. **StandardScaler** — normalises all 58 features
9. **XGBoost predict** — gradient-boosted trees, 800 estimators
10. **Inverse log1p** — converts log-price back to dollar amount
    """)

st.divider()

mc1, mc2 = st.columns([3, 1])
mc1.markdown("### Live API Metrics")
if mc2.button("Refresh", use_container_width=True):
    st.rerun()

m = _fetch_metrics()
if m:
    t   = m.get("totals", {})
    p   = m.get("predictions", {})
    lat = m.get("endpoints", {}).get("/predict", {}).get("latency_ms", {})

    cols = st.columns(5)
    cols[0].metric("Uptime",           m.get("uptime_human", "—"))
    cols[1].metric("Total Requests",   t.get("requests", 0))
    cols[2].metric("Predictions Made", p.get("total", 0))
    cols[3].metric("Error Rate",       t.get("error_rate", "—"))
    cols[4].metric("Avg Latency",      f"{lat['avg']} ms" if lat.get("avg") else "—")

    if p.get("total", 0) > 0:
        pc = st.columns(4)
        pc[0].metric("Avg Predicted",  f"${p['mean']:,.0f}/night"  if p.get("mean") else "—")
        pc[1].metric("Min Prediction", f"${p['min']:,.0f}/night"   if p.get("min")  else "—")
        pc[2].metric("Max Prediction", f"${p['max']:,.0f}/night"   if p.get("max")  else "—")
        pc[3].metric("p95 Latency",    f"{lat['p95']} ms"          if lat.get("p95") else "—")
else:
    st.info("Metrics unavailable — start the API with: `uvicorn src.new_york_workflow.nyc_api:app --port 8001`")

st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
---
**NYC Airbnb Price Prediction** &nbsp;|&nbsp; XGBoost · R²=0.82 · 58 features &nbsp;|&nbsp; v1.0

*Estimates based on April 2026 InsideAirbnb snapshot. Actual prices vary with demand, seasonality, and listing photos.*
""")
st.markdown('</div>', unsafe_allow_html=True)
