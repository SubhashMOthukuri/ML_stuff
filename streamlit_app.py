"""
House Price Predictor - Production Grade UI
Professional, Clean, Enterprise-Ready
"""

import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_data(ttl=60)
def _fetch_model_info() -> dict:
    """Fetch live model metrics from /model-info. Cached for 60s."""
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def _fetch_metrics() -> dict:
    """Fetch runtime observability data from /metrics. No cache — always live."""
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS - PROFESSIONAL STYLING
# ============================================================================

st.markdown("""
<style>
    /* Remove default Streamlit styling */
    .main {
        padding: 0 1rem;
    }
    
    /* Header styling */
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f3a93;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Input section */
    .input-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
    }
    
    /* Result section */
    .result-section {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    .result-price {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .result-confidence {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 1rem;
    }
    
    /* Slider styling */
    .stSlider {
        padding-top: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Feature card */
    .feature-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .feature-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("# 🏠 House Price Predictor")
    st.markdown("**Advanced ML Model for Real Estate Valuation**")

with col2:
    st.markdown("")
    st.markdown("")
    _info = _fetch_model_info()
    _r2   = _info.get("R²", "—")
    _err  = _info.get("Error%", "—")
    model_badge = f"""
    <div style="text-align: right; padding: 1rem; background: #f0f0f0; border-radius: 8px;">
        <div style="font-size: 0.85rem; color: #666;">Model Performance</div>
        <div style="font-size: 1.3rem; font-weight: 700; color: #667eea;">R² = {_r2}</div>
        <div style="font-size: 0.8rem; color: #999;">{_err} Avg Error</div>
    </div>
    """
    st.markdown(model_badge, unsafe_allow_html=True)

st.divider()

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Input Section
st.markdown("## 📊 Property Features")

st.markdown('<div class="input-section">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Location & Accessibility**")
    crim = st.slider(
        "Crime Rate (per capita)",
        min_value=0.0,
        max_value=30.0,
        value=0.5,
        step=0.1,
        help="Lower is better - safer neighborhoods"
    )
    
    zn = st.slider(
        "Zoning - Large Lots %",
        min_value=0.0,
        max_value=100.0,
        value=18.0,
        step=1.0,
        help="Percentage zoned for lots >25K sq ft"
    )
    
    dis = st.slider(
        "Distance to Boston Center (miles)",
        min_value=1.0,
        max_value=15.0,
        value=4.1,
        step=0.1
    )

with col2:
    st.markdown("**Property Characteristics**")
    rm = st.slider(
        "Average Rooms per Dwelling",
        min_value=3.0,
        max_value=9.0,
        value=6.5,
        step=0.1,
        help="Number of rooms - major price driver"
    )
    
    age = st.slider(
        "Building Age (years)",
        min_value=0,
        max_value=150,
        value=65,
        step=1,
        help="Newer buildings typically more valuable"
    )
    
    indus = st.slider(
        "Industrial Area %",
        min_value=0.0,
        max_value=30.0,
        value=2.3,
        step=0.1
    )

with col3:
    st.markdown("**Environment & Community**")
    
    nox = st.slider(
        "NOx Concentration (ppb)",
        min_value=0.0,
        max_value=1.0,
        value=0.54,
        step=0.01,
        help="Air quality indicator - lower is better"
    )
    
    ptratio = st.slider(
        "Pupil-Teacher Ratio",
        min_value=12.0,
        max_value=22.0,
        value=15.3,
        step=0.1,
        help="School quality indicator"
    )
    
    lstat = st.slider(
        "Lower Status Population %",
        min_value=1.0,
        max_value=40.0,
        value=5.0,
        step=0.1,
        help="Socioeconomic indicator"
    )

# Additional features in expander
with st.expander("🔧 Advanced Features"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chas = st.selectbox(
            "Charles River Boundary",
            ["No (0)", "Yes (1)"],
            help="Proximity to Charles River"
        )
        rad = st.slider(
            "Highway Access Rating",
            min_value=1,
            max_value=25,
            value=1
        )
    
    with col2:
        tax = st.slider(
            "Property Tax Rate ($/$10K)",
            min_value=200,
            max_value=700,
            value=296,
            step=1
        )
    
    with col3:
        b = st.slider(
            "Black Population Index",
            min_value=0.0,
            max_value=400.0,
            value=400.0,
            step=1.0
        )

st.markdown('</div>', unsafe_allow_html=True)

chas_value = 0 if chas == "No (0)" else 1

# ============================================================================
# PREDICTION SECTION
# ============================================================================

st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        "🔮 Predict House Price",
        use_container_width=True,
        type="primary"
    )

if predict_button:
    # Prepare features
    features = {
        'crim': crim,
        'zn': zn,
        'indus': indus,
        'chas': chas_value,
        'nox': nox,
        'rm': rm,
        'age': age,
        'dis': dis,
        'rad': rad,
        'tax': tax,
        'ptratio': ptratio,
        'b': b,
        'lstat': lstat
    }
    
    with st.spinner("🔄 Analyzing property..."):
        try:
            response = requests.post(
                f"{API_URL}/predict-raw",
                json=features,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                price = result['predicted_price']
                price_usd = result['price_usd']
                
                # Display result
                st.markdown('<div class="result-section">', unsafe_allow_html=True)
                
                st.markdown("### 🎯 Predicted Value")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f'<div class="result-price">{price_usd}</div>', 
                               unsafe_allow_html=True)
                    st.markdown("**Estimated Market Value (USD)**")
                
                with col2:
                    st.markdown(f'<div class="result-price">${price:.1f}K</div>', 
                               unsafe_allow_html=True)
                    st.markdown("**Value in $10K Units**")
                
                st.divider()
                
                # Model confidence
                st.markdown(f'<div class="result-confidence">{result["confidence"]}</div>', 
                           unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Property summary
                st.markdown("### 📋 Property Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <div class="feature-label">Rooms</div>
                        <div class="feature-value">{rm:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <div class="feature-label">Age (years)</div>
                        <div class="feature-value">{age}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <div class="feature-label">Crime Rate</div>
                        <div class="feature-value">{crim:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
                        <div class="feature-label">Distance (mi)</div>
                        <div class="feature-value">{dis:.1f}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Store result so metrics panel re-renders with fresh data
                st.session_state["last_prediction"] = price

            else:
                st.error(f"❌ API Error: {response.status_code}")
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. Make sure the API server is running.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to API. Ensure the server is available.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# ============================================================================
# INFORMATION SECTION
# ============================================================================

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Model Information")
    _info = _fetch_model_info()
    if _info:
        st.markdown(f"""
**Algorithm:** {_info.get("Model_Type", "Random Forest Regressor")}

**Performance:**
- R² Score: {_info.get("R²", "—")} (live from model)
- RMSE: {_info.get("RMSE", "—")}
- Avg Error: {_info.get("Error%", "—")} of predicted value

**Features Used:** {_info.get("Features", 18)} (13 original + 5 engineered)

**Training Data:** Boston Housing Dataset (484 properties)
        """)
    else:
        st.warning("API unavailable — model info could not be loaded.")

with col2:
    st.markdown("### ℹ️ How It Works")
    st.markdown("""
    This model uses advanced machine learning to predict house prices based on:
    
    1. **Property Features** - Room count, age, condition
    2. **Location Factors** - Crime rate, distance, accessibility
    3. **Community Metrics** - Schools, demographics, air quality
    
    The algorithm learns patterns from historical Boston housing data and applies them to your input.
    """)

# ============================================================================
# LIVE METRICS PANEL
# ============================================================================

st.divider()

_mcol1, _mcol2 = st.columns([3, 1])
_mcol1.markdown("### Live API Metrics")
if _mcol2.button("🔄 Refresh", use_container_width=True):
    st.rerun()

_m = _fetch_metrics()
if _m:
    _t   = _m.get("totals", {})
    _p   = _m.get("predictions", {})
    _lat = _m.get("endpoints", {}).get("/predict-raw", {}).get("latency_ms", {})

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Uptime",           _m.get("uptime_human", "—"))
    mc2.metric("Total Requests",   _t.get("requests", 0))
    mc3.metric("Predictions Made", _p.get("total", 0))
    mc4.metric("Error Rate",       _t.get("error_rate", "—"))
    mc5.metric("Avg Latency",
               f"{_lat.get('avg')} ms" if _lat.get("avg") else "—")

    if _p.get("total", 0) > 0:
        lc1, lc2, lc3 = st.columns(3)
        lc1.metric("Avg Predicted Price", f"${(_p['mean'] or 0) * 1000:,.0f}")
        lc2.metric("Min Prediction",      f"${(_p['min']  or 0) * 1000:,.0f}")
        lc3.metric("Max Prediction",      f"${(_p['max']  or 0) * 1000:,.0f}")
else:
    st.info("Metrics unavailable — start the API server to see live observability data.")

# Footer
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("""
---

**House Price Prediction System** | Built with Random Forest ML | v1.0

*Disclaimer: This model provides estimates based on historical data. Actual property values depend on many factors including market conditions, property condition, and local regulations.*
""")
st.markdown('</div>', unsafe_allow_html=True)