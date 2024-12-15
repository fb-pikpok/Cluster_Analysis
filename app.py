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

# Define path to precomputed JSON file
s_root = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis\Data/'           # Root
s_project = r'Steamapps\Market/'                                                      # Project
#s_project = r'HRC/'                                                                     # Project
s_db_table_preprocessed_json = os.path.join(s_root, s_project, 'openai_3_named.json')          # Input data

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

    # Steam review Stuff
    try:
        df['playtime_at_review'] = df['author'].apply(lambda x: x.get('playtime_at_review', 0))
    except:
        pass

    # Handle 'weighted_vote_score'
    try:
        df['weighted_vote_score'] = pd.to_numeric(df['weighted_vote_score'], errors='coerce').fillna(0)
    except:
        pass

    # Convert 'timestamp_created' to a datetime object and extract the month
    try:
        df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], unit='ms')  # Convert from milliseconds
        df['month'] = df['timestamp_created'].dt.to_period('M').astype(str)  # Convert to string for JSON serialization
    except:
        pass

    # endregion
    return df

df_total = load_data(s_db_table_preprocessed_json)


# region Sidebar filters
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

#endregion


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


# region Filters
# Optional filters in st_source.filter_functions.py
filtered_df = apply_optional_filters(df_total, optional_filters)

# Hide noise if selected
if hide_noise:
    if display_mode == "ID":
        filtered_df = filtered_df[filtered_df[clustering_column] != -1]
    else:
        filtered_df = filtered_df[filtered_df[clustering_name_column] != "Unknown"]


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


# region Additional Charts
# Bar Chart (horizontal) for Sentiment per Cluster
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

# Add a selection button for data type
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



# Bar chart (vertical) for Sentiment over Time
st.subheader("Sentiment Over Time")
try:
    fig_sentiment_time = plot_sentiments_over_time(
        df=filtered_df,
        sentiment_col='sentiment',
        month_col='month'
    )
    st.plotly_chart(fig_sentiment_time)
except:
    st.warning("No data available for sentiment over time.")

# endregion



