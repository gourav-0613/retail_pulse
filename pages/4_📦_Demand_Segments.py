
import streamlit as st
import pandas as pd
import plotly.express as px

from utils.preprocessing import (
    load_data,
    clean_data,
    create_features,
)

from utils.clustering import (
    demand_segmentation,
    cluster_summary,
)

# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="Demand Segments",
    page_icon="📦",
    layout="wide",
)

st.title("📦 Product Demand Segmentation")

st.markdown(
    """
Group products into demand segments using
K-Means Clustering.
"""
)

# =====================================================
# Load Dataset
# =====================================================

df = load_data("data/train.csv")
df = clean_data(df)
df = create_features(df)

# =====================================================
# Sidebar
# =====================================================

st.sidebar.header("Segmentation Settings")

cluster_count = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=6,
    value=3,
)

# =====================================================
# Prepare Data
# =====================================================

cluster_df = demand_segmentation(df)

summary = cluster_summary(df)

cluster_df["Cluster"] = cluster_df["Cluster"].astype(str)

# =====================================================
# KPI Cards
# =====================================================

total_products = len(cluster_df)

avg_sales = cluster_df["Total_Sales"].mean()

avg_profit = cluster_df["Average_Sales"].mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Products",
        total_products,
    )

with col2:
    st.metric(
        "Average Sales",
        f"{avg_sales:,.0f}",
    )

with col3:
    st.metric(
        "Average Profit",
        f"{avg_profit:,.0f}",
    )

st.divider()

# =====================================================
# Scatter Plot
# =====================================================

st.subheader("📊 Demand Clusters")

fig = px.scatter(
    cluster_df,
    x="Average_Sales",
    y="Total_Sales",
    color="Cluster",
    hover_name="Sub-Category",
    size="Orders",
)

fig.update_layout(
    xaxis_title="Average_Sales",
    yaxis_title="Total_Sales",
)

st.plotly_chart(
    fig,
    use_container_width=True,
)

st.divider()

# =====================================================
# Cluster Summary
# =====================================================

st.subheader("📋 Product Segments")

st.dataframe(
    cluster_df.sort_values(
        [
            "Cluster",
            "Total_Sales"
        ],
        ascending=[
            True,
            False
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# =====================================================
# Cluster Statistics
# =====================================================

st.subheader("📈 Demand Segment Summary")

st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
)

csv = cluster_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇ Download Demand Segments",
    csv,
    "demand_segments.csv",
    "text/csv",
)