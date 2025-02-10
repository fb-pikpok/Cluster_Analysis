import streamlit as st
import pandas as pd
import os
from st_source.visuals import *
from st_source.filter_functions import *
from st_source.keywordSearch import initialize_miniLM, index_embedding, get_top_keyword_result
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json


# Set page layout
st.set_page_config(layout="wide")



# region 1) Page Setup & Load Data

# Define path to precomputed JSON file
s_root = r'S:\SID\Analytics\Working Files\Individual\Florian\Projects\DataScience\cluster_analysis\Data/'           # Root
s_project = r'HRC\Cluster_tests/'                                                      # Project
#s_project = r'HRC/'                                                                     # Project
s_db_table_preprocessed_json = os.path.join(s_root, s_project, 'db_final.json')          # Input data

@st.cache_data(show_spinner=False)
def load_data(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # region Data Preprocessing
    try:
        df['prestige_rank'] = pd.to_numeric(df['prestige_rank'].replace("", 0), errors='coerce').fillna(0).astype(int)
    except:
        pass
    try:
        df['ever_been_subscriber'] = pd.to_numeric(df['ever_been_subscriber'].replace("", 0), errors='coerce').fillna(0).astype(int)
    except:
        pass
    try:
        df['is_current_subscriber'] = pd.to_numeric(df['is_current_subscriber'].replace("", 0), errors='coerce').fillna(0).astype(int)
    except:
        pass
    try:
        df['spending'] = pd.to_numeric(df['spending'].replace("", 0), errors='coerce').fillna(0).astype(int)
    except:
        pass
    try:
        df['playtime_at_review_minutes'] = df['author'].apply(lambda x: x.get('playtime_at_review_minutes', 0))
    except:
        pass
    try:
        df['weighted_vote_score'] = pd.to_numeric(df['weighted_vote_score'], errors='coerce').fillna(0)
    except:
        pass
    try:
        # Note: Adjust 'unit' if needed. Some data uses seconds, others milliseconds.
        df['timestamp_updated'] = pd.to_datetime(df['timestamp_updated'], unit='s')
        df['month'] = df['timestamp_updated'].dt.to_period('M').astype(str)
    except:
        pass
    # endregion

    return df

df_total = load_data(s_db_table_preprocessed_json)

# endregion


# region 2) Sidebar Filters

st.sidebar.header("Filter Options")


display_mode = st.sidebar.selectbox("Display Mode", ["ID", "Name"])

dimensionality_options = ["UMAP", "PCA", "tSNE"]
clustering_options = ["hdbscan", "kmeans"]
selected_dimensionality = st.sidebar.selectbox("Dimensionality Reduction", dimensionality_options)
selected_clustering = st.sidebar.selectbox("Clustering Algorithm", clustering_options)

hide_noise = st.sidebar.checkbox("Hide Noise", value=False)
view_options = ["2D", "3D"]
selected_view = st.sidebar.radio("Select View", view_options)

# K-Means specific cluster-size selector
# Since we don't know what the available cluster sizes are, this function checks them and returns them
# so we can display them in the selector on the sidebar
if selected_clustering == "kmeans":
    kmeans_columns = [col for col in df_total.columns if col.startswith("kmeans_")]
    if kmeans_columns:
        available_kmeans_sizes = sorted({int(col.split("_")[1]) for col in kmeans_columns if col.split("_")[1].isdigit()})
        selected_kmeans_size = st.sidebar.selectbox("Cluster size", options=available_kmeans_sizes)
    else:
        selected_kmeans_size = st.sidebar.number_input(
            "Enter Number of KMeans Clusters",
            min_value=1, max_value=100, value=15, step=1
        )


if selected_clustering == "kmeans":
    clustering_column = f"{selected_clustering}_{selected_kmeans_size}_id"
    clustering_name_column = f"{clustering_column}_name"
else:
    clustering_column = "hdbscan_id"
    clustering_name_column = "hdbscan_id_name"

# Display by name or by ID (Name is initially selected)
if display_mode == "Name":
    all_clusters_list = sorted(df_total[clustering_name_column].dropna().unique().tolist())
else:
    all_clusters_list = sorted(df_total[clustering_column].dropna().unique().tolist())


cluster_options = ["All Clusters"] + all_clusters_list
selected_clusters = st.sidebar.multiselect(
    "Select Clusters (Multi-Select)",
    cluster_options,
    default=["All Clusters"]
)

# endregion


# region 3) Apply Additional Filters

filtered_df = df_total.copy()

# Apply optional sidebar filters (like sentiment, prestige, etc.)
# (Assuming you have an apply_optional_filters function)
filtered_df = apply_optional_filters(filtered_df)

# Hide Noise
if hide_noise:
    if display_mode == "ID":
        filtered_df = filtered_df[filtered_df[clustering_column] != -1]
    else:
        filtered_df = filtered_df[filtered_df[clustering_name_column] != "Noise"]

# Filter by selected cluster(s)
# NEW OR CHANGED
if "All Clusters" not in selected_clusters:
    if display_mode == "ID":
        filtered_df = filtered_df[filtered_df[clustering_column].isin(selected_clusters)]
    else:
        filtered_df = filtered_df[filtered_df[clustering_name_column].isin(selected_clusters)]

# endregion



# region 4) Cluster Visualization

color_map = generate_color_map(df_total, clustering_column, clustering_name_column, display_mode)  # color map from full dataset

st.subheader("Cluster Visualization")
if not filtered_df.empty:
    x_col = f"{selected_clustering}_{selected_dimensionality}_2D_x"
    y_col = f"{selected_clustering}_{selected_dimensionality}_2D_y"
    z_col = f"{selected_clustering}_{selected_dimensionality}_3D_z" if selected_view == "3D" else None

    missing_cols = [col for col in [x_col, y_col, z_col] if col and col not in df_total.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    else:
        fig = visualize_embeddings(
            filtered_df,
            x_col,
            y_col,
            z_col,
            'sentence',  # hover info
            clustering_name_column if display_mode == "Name" else clustering_column,
            color_map
        )
        st.plotly_chart(fig)
else:
    st.warning("No data available for visualization.")

# endregion


# region 5) Data Table
mandatory_columns = ['topic', 'sentence', 'category', 'sentiment']
optional_columns = [
    col for col in df_total.columns
    if col not in mandatory_columns
    and col not in [clustering_column, clustering_name_column]
]

selected_columns = st.sidebar.multiselect(
    "Select Additional Columns to Display",
    optional_columns,
    default=[]
)
columns_to_display = mandatory_columns + selected_columns

st.subheader("Filtered Data Table")
if not filtered_df.empty:
    st.dataframe(filtered_df[columns_to_display])
else:
    st.warning("No data available for the selected filters (table).")

# endregion


# region 6) Additional Charts

# Sentiment per Cluster
st.subheader("Sentiment per Cluster")
if not filtered_df.empty:
    fig_sentiment = plot_sentiments(
        filtered_df,
        sentiment_col='sentiment',
        cluster_name_col=clustering_name_column if display_mode == "Name" else clustering_column
    )
    st.plotly_chart(fig_sentiment)
else:
    st.warning("No sentiment data available for the selected filters.")

# Cluster sizes
st.subheader("Requests per Cluster")
data_type = st.radio(
    "Select data to display:",
    options=["requests", "facts", "both"],
    index=0,
    horizontal=True
)
if not filtered_df.empty:
    if 'category' not in filtered_df.columns:
        st.warning("The 'category' column is not available in the dataset.")
    else:
        fig_request_count = plot_request_count_by_cluster(
            filtered_df,
            cluster_name_col=clustering_name_column if display_mode == "Name" else clustering_column,
            data_type=data_type
        )
        st.plotly_chart(fig_request_count)
else:
    st.warning("No request count data available for the selected filters.")


# Bar Chart Sentiment Over Time
st.subheader("Sentiment Over Time")
try:
    # Here we assume plot_sentiments_over_time handles any data subset
    # If multiple clusters are selected, it aggregates them
    fig_sentiment_time = plot_sentiments_over_time(
        df=filtered_df,
        sentiment_col='sentiment',
        month_col='month'
    )
    st.plotly_chart(fig_sentiment_time)
except:
    st.warning("No data available for sentiment over time (bar).")


# Line Chart Sentiment Over Time
cluster_col = None
if "All Clusters" not in selected_clusters:
    # We have a cluster column
    cluster_col = clustering_name_column if display_mode == "Name" else clustering_column

aggregated_df = compute_sentiment_over_time(
    df=filtered_df,
    time_col="timestamp_updated",
    cluster_col=cluster_col,           # None means single aggregated, otherwise multi-cluster
    granularity="M"                    # or "W", "D", "H" etc.
)

fig_line = plot_sentiments_over_time_line(
    aggregated_df=aggregated_df,
    cluster_col=cluster_col,
    color_map=color_map,
    title="Individual Line Chart"
)

if fig_line is not None:
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.warning("No data available for time-based sentiment analysis.")


