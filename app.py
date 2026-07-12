import streamlit as st

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------
# Load CSS
# -----------------------------------------------------

with open("assets/css.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# -----------------------------------------------------
# Hero
# -----------------------------------------------------

st.title("📊 RetailPulse")

st.subheader(
    "AI-Powered Retail Sales Forecasting & Demand Analytics Dashboard"
)

st.markdown("---")

st.markdown(
"""
RetailPulse is an interactive dashboard developed for retail sales analytics.

### Features

- 📈 Sales Overview Dashboard
- 🔮 Sales Forecast Explorer
- 🚨 Anomaly Detection
- 📦 Product Demand Segmentation

---

### Models Used

- Gradient Boosting Regressor
- SARIMA
- Isolation Forest
- KMeans Clustering
---

### Dataset

Superstore Sales Dataset

---

👈 Use the left sidebar to open each dashboard page.
"""
)

st.info(
    "👈 Open the pages from the left sidebar to explore the dashboard."
)