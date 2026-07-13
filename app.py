import streamlit as st

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------

st.set_page_config(
    page_title="RetailPulse: Forecasting & Demand Analytics",
    page_icon="assets/retailpulse-logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------
# Load CSS
# -----------------------------------------------------

with open("assets/css.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True,
    )

# =====================================================
# Sidebar Branding
# =====================================================

brand1, brand2 = st.sidebar.columns([1, 4], gap="small")

with brand1:
    st.image(
        "assets/retailpulse-logo.png",
        width=48
    )

with brand2:
    st.markdown(
        """
        <div style="padding-top:10px;">
            <h2 style="margin:0;font-size:24px;font-weight:700;color:white;">
                Retail Pulse
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
# -----------------------------------------------------
# Header
# -----------------------------------------------------

col1, col2, col3 = st.columns([1,8,1])

with col2:
    st.title("RetailPulse: Forecasting & Demand Analytics")
    st.caption(
        "Retail Intelligence Dashboard"
    )

with col2:
    st.image(
        "assets/hero.svg",
        width=950,
    )

st.markdown("---")

# -----------------------------------------------------
# Project Overview
# -----------------------------------------------------

st.subheader("Project Overview")

st.write(
    """
RetailPulse is an end-to-end AI-powered retail analytics dashboard built
using **Python, Streamlit, Scikit-learn and Machine Learning**.

The dashboard enables business users to analyze historical sales,
forecast future demand, detect anomalies and identify demand segments
through interactive visualizations.
"""
)

st.markdown("---")

# -----------------------------------------------------
# Features
# -----------------------------------------------------

st.subheader("Dashboard Features")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.image(
        "assets/icons/icon-dashboard.svg",
        width=45,
    )
    st.markdown("### Sales Overview")

with c2:
    st.image(
        "assets/icons/icon-forecast.svg",
        width=45,
    )
    st.markdown("### Forecast Explorer")

with c3:
    st.image(
        "assets/icons/icon-warning.svg",
        width=45,
    )
    st.markdown("### Anomaly Report")

with c4:
    st.image(
        "assets/icons/icon-demand.svg",
        width=45,
    )
    st.markdown("### Demand Segments")

st.markdown("---")

# -----------------------------------------------------
# Models & Dataset
# -----------------------------------------------------

left, right = st.columns(2)

with left:

    st.image(
        "assets/icons/icon-ai.svg",
        width=45,
    )

    st.subheader("Models Used")

    st.markdown("""
- Gradient Boosting Regressor
- SARIMA
- Isolation Forest
- K-Means Clustering
""")

with right:

    st.image(
        "assets/icons/icon-database.svg",
        width=45,
    )

    st.subheader("Dataset")

    st.write(
        "Superstore Sales Dataset"
    )

st.markdown("---")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------

st.success(
    "Open the pages from the left sidebar to explore the complete dashboard."
)