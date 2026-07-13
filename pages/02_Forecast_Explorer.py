 
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import (
    load_data,
    clean_data,
    create_features,
    monthly_sales,
)

from utils.forecasting import (
    generate_forecast,
    compare_models,
)

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="RetailPulse | Forecast Explorer",
    page_icon="assets/retailpulse-logo.png",
    layout="wide",
)

col1, col2 = st.columns([1, 8])

with col1:
    st.image(
        "assets/icons/icon-forecast.svg",
        width=45,
    )

with col2:
    st.markdown(
    "<h1 style='margin-top:8px;'>Forecast Explorer</h1>",
    unsafe_allow_html=True,
    )

    st.caption(
        "Predict future retail sales using Machine Learning models."
    )

st.markdown("---")

# =====================================================
# Load Dataset
# =====================================================
df = load_data("data/train.csv")
df = clean_data(df)
df = create_features(df)

# =====================================================
# Sidebar Filters
# =====================================================

st.sidebar.header("Forecast Settings")

forecast_months = st.sidebar.slider(
    "Forecast Horizon (Months)",
    min_value=1,
    max_value=3,
    value=3,
)

filter_type = st.sidebar.radio(
    "Forecast By",
    [
        "Overall",
        "Category",
        "Region",
    ],
)

filtered_df = df.copy()

# =====================================================
# Category Filter
# =====================================================

if filter_type == "Category":

    category = st.sidebar.selectbox(
        "Select Category",
        sorted(df["Category"].unique())
    )

    filtered_df = df[
        df["Category"] == category
    ]

# =====================================================
# Region Filter
# =====================================================

elif filter_type == "Region":

    region = st.sidebar.selectbox(
        "Select Region",
        sorted(df["Region"].unique())
    )

    filtered_df = df[
        df["Region"] == region
    ]

# =====================================================
# Prepare Monthly Sales
# =====================================================

monthly_df = monthly_sales(filtered_df)
monthly_df.columns = ["Date", "Sales"]

if len(monthly_df) < 12:

    st.warning(
        "Not enough monthly data available for forecasting."
    )

    st.stop()

# =====================================================
# Run Forecast
# =====================================================

result = generate_forecast(
    monthly_df,
    forecast_periods=forecast_months,
)

forecast_df = result["forecast"]

metrics = result["metrics"]

comparison = compare_models(
    monthly_df
)


# =====================================================
# KPI Metrics
# =====================================================

st.subheader("Forecast Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "MAE",
        f"{metrics['MAE']:.2f}"
    )

with col2:
    st.metric(
        "RMSE",
        f"{metrics['RMSE']:.2f}"
    )

with col3:
    st.metric(
        "MAPE",
        f"{metrics['MAPE']:.2f}%"
    )

st.divider()

# =====================================================
# Forecast Chart
# =====================================================

st.subheader("Sales Forecast")

history = monthly_df.copy()
history["Type"] = "Historical"

forecast = forecast_df.copy()
forecast = forecast.rename(
    columns={"Forecast": "Sales"}
)
forecast["Type"] = "Forecast"

plot_df = pd.concat(
    [
        history[["Date", "Sales", "Type"]],
        forecast[["Date", "Sales", "Type"]],
    ],
    ignore_index=True,
)

fig = px.line(
    plot_df,
    x="Date",
    y="Sales",
    color="Type",
    markers=True,
    title="Historical vs Forecast Sales",
)

fig.update_traces(
    line=dict(width=4),
    marker=dict(size=8)
)

for trace in fig.data:

    if trace.name == "Forecast":
        trace.line.dash = "dash"

fig.update_layout(
    template="plotly_dark",
    height=500,
    title_x=0.02,
    xaxis_title="Month",
    yaxis_title="Sales",
    hovermode="x unified",
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20
    )
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.divider()

# =====================================================
# Forecast Table
# =====================================================

st.subheader("Forecast Results")

display_df = forecast_df.copy()

display_df["Date"] = (
    pd.to_datetime(display_df["Date"])
    .dt.strftime("%b %Y")
)

display_df["Forecast"] = (
    display_df["Forecast"]
    .round(2)
)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
)

# =====================================================
# Download Forecast
# =====================================================

csv = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast Results",
    data=csv,
    file_name="forecast_results.csv",
    mime="text/csv",
)

st.divider()

# =====================================================
# Model Comparison
# =====================================================

st.subheader("Model Comparison")

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True,
)

best_model = comparison.iloc[0]["Model"]

st.markdown("---")

st.info(
    f"Best Performing Model: {best_model}"
)