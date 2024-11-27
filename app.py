import streamlit as st
import pandas as pd
from st_source.visuals import *
from st_source.keywordSearch import initialize_miniLM, index_embedding, get_top_keyword_result
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json


# Set page layout
st.set_page_config(layout="wide")

# Define path to precomputed JSON file
s_root = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis/'           # Root
s_db_table_preprocessed_json = 'Data/db_final.json'                                 # Input data

# Load precomputed data
@st.cache_data(show_spinner=False)
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    #TODO: This should be handled in preprocessing
    # region Handle empty cells and non String values for the filters

    # Handle 'prestige_rank' column: replace empty strings with 0 and convert to integers
    try:
        df['prestige_rank'] = pd.to_numeric(df['prestige_rank'].replace("", 0), errors='coerce').fillna(0).astype(int)
    except:
        pass

    # Handle 'ever_been_subscriber': replace empty strings with 0 and convert to integers
    try:
        df['ever_been_subscriber'] = pd.to_numeric(df['ever_been_subscriber'].replace("", 0), errors='coerce').fillna(0).astype(int)
    except:
        pass

    # Handle 'is_current_subscriber': replace empty strings with 0 and convert to integers
    try:
        df['is_current_subscriber'] = pd.to_numeric(df['is_current_subscriber'].replace("", 0), errors='coerce').fillna(0).astype(int)
    except:
        pass

    # Handle 'spending': replace empty strings with 0 and convert to integers
    try:
        df['spending'] = pd.to_numeric(df['spending'].replace("", 0), errors='coerce').fillna(0).astype(int)
    except:
        pass
    # endregion
    return df

df_total = load_data(s_root + s_db_table_preprocessed_json)


# Sidebar filters
st.sidebar.header("Filter Options")
# Display mode
display_mode = st.sidebar.selectbox("Display Mode", ["ID", "Name"])
dimensionality_options = ["UMAP","PCA", "tSNE"]
clustering_options = ["hdbscan", "kmeans"]
selected_dimensionality = st.sidebar.selectbox("Dimensionality Reduction", dimensionality_options)
selected_clustering = st.sidebar.selectbox("Clustering Algorithm", clustering_options)


# Kmeans selected -> cluster size selector enabled
if selected_clustering == "kmeans":
    # Check for existing KMeans clusters in the input data
    kmeans_columns = [col for col in df_total.columns if col.startswith("kmeans_")]

    if kmeans_columns:
        # Extract available cluster sizes from existing KMeans columns
        available_kmeans_sizes = sorted(
            {int(col.split("_")[1]) for col in kmeans_columns if col.split("_")[1].isdigit()}
        )
        selected_kmeans_size = st.sidebar.selectbox(
            "Cluster size",
            options=available_kmeans_sizes,
            help="Select from available cluster sizes in the input data."
        )
    else:
        # Allow manual input if no existing KMeans clusters are found
        selected_kmeans_size = st.sidebar.number_input(
            "Enter Number of KMeans Clusters",
            min_value=1,
            max_value=100,
            value=15,
            step=1,
            help="Manually specify the number of KMeans clusters."
        )

# Hide Noise Checkbox
hide_noise = st.sidebar.checkbox("Hide Noise", value=False)

# View selection (2D or 3D)
view_options = ["2D", "3D"]
selected_view = st.sidebar.radio("Select View", view_options)

# region Cluster Selection
if selected_clustering == "kmeans":
    clustering_column = f"{selected_clustering}_{selected_kmeans_size}_{selected_dimensionality}_{selected_view}"
else:
    clustering_column = f"{selected_clustering}_{selected_dimensionality}_{selected_view}"

clustering_name_column = f"{clustering_column}_name"

# Add "All Clusters" option to the cluster selection
if display_mode == "ID":
    cluster_options = ["All Clusters"] + sorted(df_total[clustering_column].dropna().unique().tolist())
else:
    cluster_options = ["All Clusters"] + sorted(df_total[clustering_name_column].dropna().unique().tolist())

selected_cluster_value = st.sidebar.selectbox("Select Cluster", cluster_options)
# endregion


# Sentiment filter
selected_sentiment = st.sidebar.multiselect(
    "Select Sentiment",
    df_total['sentiment'].unique(),
    default=df_total['sentiment'].unique()
)

# Prestige Rank Slider
if 'prestige_rank' in df_total.columns:
    min_value = int(df_total['prestige_rank'].min())
    max_value = int(df_total['prestige_rank'].max())
    prestige_rank_range = st.sidebar.slider(
        "Prestige Rank Slider",
        min_value, max_value, (min_value, max_value),
    )

    # Extract the minimum and maximum from the slider's output
    prestige_rank_min, prestige_rank_max = prestige_rank_range
else:
    prestige_rank_min = None
    prestige_rank_max = None


# Spenders Checkbox
if 'spending' in df_total.columns:
    filter_spenders = st.sidebar.checkbox("Only Show Spenders", value=False)

# Current_Subscriber Checkbox
if 'is_current_subscriber' in df_total.columns:
    filter_current_subscriber = st.sidebar.checkbox("Only Show Current Subscribers", value=False)

# Prevoius_Subscriber Checkbox
if 'ever_been_subscriber' in df_total.columns:
    filter_previous_subscriber = st.sidebar.checkbox("Only Show Previous Subscribers", value=False)


# region Apply filters to the DataFrame

# Define filtered DataFrame
filtered_df = df_total[
    (df_total['sentiment'].isin(selected_sentiment))
]

# filter by Prestige Rank range
if prestige_rank_min is not None and prestige_rank_max is not None:
    filtered_df = filtered_df[
        (df_total['prestige_rank'] >= prestige_rank_min) &
        (df_total['prestige_rank'] <= prestige_rank_max)
    ]

# Hide noise if selected
if hide_noise:
    if display_mode == "ID":
        filtered_df = filtered_df[filtered_df[clustering_column] != -1]
    else:
        filtered_df = filtered_df[filtered_df[clustering_name_column] != "Unknown"]

# Filter by Spenders
if filter_spenders:
    filtered_df = filtered_df[df_total['spending'] > 0]

# Filter by Current Subscribers
if filter_current_subscriber:
    filtered_df = filtered_df[df_total['is_current_subscriber'] == 1]

# Filter by Previous Subscribers
if filter_previous_subscriber:
    filtered_df = filtered_df[df_total['ever_been_subscriber'] == 1]

# Select individual clusters
if selected_cluster_value != "All Clusters":
    if display_mode == "ID":
        filtered_df = filtered_df[filtered_df[clustering_column] == selected_cluster_value]
    else:
        filtered_df = filtered_df[filtered_df[clustering_name_column] == selected_cluster_value]
# endregion


# region Cluster Visualization

# Map colors to clusters to ensure the same cluster has the same color across different visualizations / filters
color_map = generate_color_map(
    dataframe=df_total,
    clustering_column=clustering_column,
    clustering_name_column=clustering_name_column,
    display_mode=display_mode
)

# Visualize the clusters
st.subheader("Cluster Visualization")
if not filtered_df.empty:
    x_col = f"{clustering_column}_x"
    y_col = f"{clustering_column}_y"
    z_col = f"{clustering_column}_z" if selected_view == "3D" else None
    missing_cols = [col for col in [x_col, y_col, z_col] if col and col not in df_total.columns]

    if missing_cols:
        st.error(f"One or more selected columns are missing: {missing_cols}")
    else:
        fig = visualize_embeddings(
            filtered_df,
            x_col=x_col,
            y_col=y_col,
            z_col=z_col,
            review_text_column='sentence',
            colour_by_column=clustering_name_column if display_mode == "Name" else clustering_column,
            color_map=color_map  # Pass the color map for consistent coloring
        )
        st.plotly_chart(fig)
else:
    st.warning("No data available for the selected filters (visualization).")
# endregion


# region DataFrame

# Mandatory columns
mandatory_columns = ['topic', 'sentence', 'category', 'sentiment']
optional_columns = [col for col in df_total.columns if col not in mandatory_columns and col not in [
    clustering_column, clustering_name_column]]

# User-selectable columns
selected_columns = st.sidebar.multiselect(
    "Select Additional Columns to Display",
    optional_columns,
    default=optional_columns[:0]            # Default: dont display any additional columns
)

# Final columns for the table
columns_to_display = mandatory_columns + selected_columns

st.subheader("Filtered Data Table")
if not filtered_df.empty:
    st.dataframe(filtered_df[columns_to_display])
else:
    st.warning("No data available for the selected filters (table).")
# endregion


# region Sentiment bar and requests chart
st.subheader("Cluster Sentiment and Request Distribution")
col1, col2 = st.columns(2)

with col1:
    if not filtered_df.empty:
        fig_sentiment = plot_sentiments(
            filtered_df,
            sentiment_col='sentiment',
            cluster_name_col=clustering_name_column if display_mode == "Name" else clustering_column
        )
        st.plotly_chart(fig_sentiment)
    else:
        st.warning("No sentiment data available for the selected filters.")

with col2:
    if not filtered_df.empty:
        fig_request_count = plot_request_count_by_cluster(
            filtered_df,
            cluster_name_col=clustering_name_column if display_mode == "Name" else clustering_column
        )
        st.plotly_chart(fig_request_count)
    else:
        st.warning("No request count data available for the selected filters.")

# endregion

