import streamlit as st
import pandas as pd
import json
from st_source.visuals import visualize_embeddings, plot_diverging_sentiments, plot_request_count_by_cluster
from st_source.keywordSearch import initialize_miniLM, index_embedding, get_top_keyword_result
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from plotly.colors import qualitative

# Set page layout to wide
st.set_page_config(layout="wide")

# Define path to precomputed JSON file
s_root = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis/'
s_db_table_preprocessed_json = 'Data/db_final.json'  # Precomputed JSON

# Load precomputed data
@st.cache_data(show_spinner=False)
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Handle 'prestige_rank' column: replace empty strings with 0 and convert to integers
    df['prestige_rank'] = pd.to_numeric(df['prestige_rank'].replace("", 0), errors='coerce').fillna(0).astype(int)

    # Handle 'ever_been_subscriber': replace empty strings with 0 and convert to integers
    df['ever_been_subscriber'] = pd.to_numeric(df['ever_been_subscriber'].replace("", 0), errors='coerce').fillna(0).astype(int)

    return df

df_total = load_data(s_root + s_db_table_preprocessed_json)

# Cache the embedding model
@st.cache_resource
def load_embedding_model():
    return initialize_miniLM()

embed_model = load_embedding_model()

# Sidebar filters
st.sidebar.header("Visualization and Clustering Options")
dimensionality_options = ["UMAP", "PCA", "tSNE"]
clustering_options = ["hdbscan", "kmeans"]
selected_dimensionality = st.sidebar.selectbox("Select Dimensionality Reduction", dimensionality_options)
selected_clustering = st.sidebar.selectbox("Select Clustering Algorithm", clustering_options)

# KMeans cluster size selection, only shows if KMeans is selected
if selected_clustering == "kmeans":
    kmeans_cluster_sizes = [15, 20, 25, 50]
    selected_kmeans_size = st.sidebar.selectbox("Select Number of KMeans Clusters", kmeans_cluster_sizes)

# View selection (2D or 3D)
view_options = ["2D", "3D"]
selected_view = st.sidebar.radio("Select View", view_options)

# Sentiment filter
selected_sentiment = st.sidebar.multiselect("Select Sentiment", df_total['sentiment'].unique(),
                                            default=df_total['sentiment'].unique())

# Prestige rank slider
prestige_rank_min = st.sidebar.slider("Minimum Prestige Rank", min_value=0, max_value=25, value=0)

# Hide Noise Checkbox
hide_noise = st.sidebar.checkbox("Hide Noise", value=False)

# Filter Spenders Checkbox
filter_spenders = st.sidebar.checkbox("Only Show Spenders", value=False)

# Display mode: "ID" for cluster IDs or "Name" for cluster names
display_mode = st.sidebar.selectbox("Select Display Mode", ["ID", "Name"])

# Generate the correct clustering column names for IDs and names
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

# Define filtered DataFrame
filtered_df = df_total[
    (df_total['prestige_rank'] >= prestige_rank_min) &  # Apply prestige rank filter
    (df_total['sentiment'].isin(selected_sentiment))    # Apply sentiment filter
]

# Apply "Hide Noise" filter
if hide_noise:
    if display_mode == "ID":
        filtered_df = filtered_df[filtered_df[clustering_column] != -1]
    else:
        filtered_df = filtered_df[filtered_df[clustering_name_column] != "Unknown"]

# Apply "Only Show Spenders" filter
if filter_spenders:
    filtered_df = filtered_df[filtered_df['ever_been_subscriber'] == 1]

if selected_cluster_value != "All Clusters":
    if display_mode == "ID":
        filtered_df = filtered_df[filtered_df[clustering_column] == selected_cluster_value]
    else:
        filtered_df = filtered_df[filtered_df[clustering_name_column] == selected_cluster_value]

# Generate a fixed color map for clusters
color_palette = qualitative.Set2  # Example Plotly color palette
max_colors = len(color_palette)

# Generate unique cluster values
unique_clusters = sorted(df_total[clustering_column].dropna().unique())
if display_mode == "Name":
    unique_clusters = sorted(df_total[clustering_name_column].dropna().unique())

# Create a color map dictionary
color_map = {cluster: color_palette[i % max_colors] for i, cluster in enumerate(unique_clusters)}

# Cluster Visualization
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

# Cluster Details Table
st.subheader(f"Cluster Details for {selected_cluster_value if selected_cluster_value != 'All Clusters' else 'All Clusters'}")
if not filtered_df.empty:
    st.dataframe(filtered_df[['topic', 'sentence', 'category', 'sentiment', 'is_current_subscriber',
                              'ever_been_subscriber', 'spending', 'prestige_rank']])
else:
    st.warning("No data available for the selected filters (details).")

# Sentiment and Request Count Plots
st.subheader("Cluster Sentiment and Request Distribution")
col1, col2 = st.columns(2)

with col1:
    if not filtered_df.empty:
        fig_sentiment = plot_diverging_sentiments(
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

# Keyword search bar
st.subheader("Keyword Search")
keyword = st.text_input("Enter a keyword to search:", "")
if keyword:
    # Perform keyword search
    keyword_embedding = index_embedding(keyword, embed_model)
    df_total['similarity'] = cosine_similarity(np.vstack(df_total['embedding']), keyword_embedding.reshape(1, -1)).flatten()
    top_results = df_total.nlargest(20, 'similarity')

    # Display top results
    st.subheader("Top 20 Results")
    st.dataframe(top_results[['topic', 'sentence', 'category', 'sentiment', 'similarity']])
