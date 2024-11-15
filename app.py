import streamlit as st
import pandas as pd
import json

# Helper functions
from st_source.visuals import visualize_embeddings, plot_diverging_sentiments, plot_request_count_by_cluster

# Set page layout to wide
st.set_page_config(layout="wide")

# Upload JSON file
st.sidebar.header("Upload Your JSON File")
uploaded_file = st.sidebar.file_uploader("Choose a JSON file", type="json")

# Check if a file is uploaded
if uploaded_file is not None:
    # Load data from uploaded file
    @st.cache_data(show_spinner=False)
    def load_data(file):
        data = json.load(file)
        df = pd.DataFrame(data)
        return df

    df_total = load_data(uploaded_file)

    # Set display mode: "ID" for cluster IDs or "Name" for cluster names
    display_mode = st.sidebar.selectbox("Select Display Mode", ["ID", "Name"])

    # Sidebar options
    st.sidebar.header("Visualization and Clustering Options")
    dimensionality_options = ["UMAP", "PCA", "tSNE"]
    clustering_options = ["hdbscan", "kmeans"]
    selected_dimensionality = st.sidebar.selectbox("Select Dimensionality Reduction", dimensionality_options)
    selected_clustering = st.sidebar.selectbox("Select Clustering Algorithm", clustering_options)

    # KMeans cluster size selection, only shows if KMeans is selected
    if selected_clustering == "kmeans":
        kmeans_cluster_sizes = [5, 10, 15, 20, 35]
        selected_kmeans_size = st.sidebar.selectbox("Select Number of KMeans Clusters", kmeans_cluster_sizes)

    # View selection (2D or 3D)
    view_options = ["2D", "3D"]
    selected_view = st.sidebar.radio("Select View", view_options)

    # Generate the correct clustering column names for IDs and names
    if selected_clustering == "kmeans":
        clustering_column = f"{selected_clustering}_{selected_kmeans_size}_{selected_dimensionality}_{selected_view}"
    else:
        clustering_column = f"{selected_clustering}_{selected_dimensionality}_{selected_view}"

    clustering_name_column = f"{clustering_column}_name"

    if clustering_column not in df_total.columns:
        st.error(f"Clustering column '{clustering_column}' does not exist in the data.")
    else:
        if display_mode == "ID":
            unique_cluster_values = sorted(df_total[clustering_column].unique())
            selected_cluster_value = st.sidebar.selectbox("Select Cluster ID", unique_cluster_values)
        else:
            unique_cluster_values = sorted(df_total[clustering_name_column].dropna().unique())
            selected_cluster_value = st.sidebar.selectbox("Select Cluster Name", unique_cluster_values)

        selected_sentiment = st.sidebar.multiselect("Select Sentiment", df_total['sentiment'].unique(),
                                                    default=df_total['sentiment'].unique())

        if display_mode == "ID":
            filtered_df = df_total[(df_total[clustering_column] == selected_cluster_value) &
                                   (df_total['sentiment'].isin(selected_sentiment))]
        else:
            filtered_df = df_total[(df_total[clustering_name_column] == selected_cluster_value) &
                                   (df_total['sentiment'].isin(selected_sentiment))]

        dimensionality = selected_dimensionality.upper()
        view_suffix = "2D" if selected_view == "2D" else "3D"

        x_col = f"{clustering_column}_x"
        y_col = f"{clustering_column}_y"
        z_col = f"{clustering_column}_z" if selected_view == "3D" else None

        missing_cols = [col for col in [x_col, y_col, z_col] if col and col not in df_total.columns]
        if missing_cols:
            st.error(f"One or more selected columns are missing: {missing_cols}")
        else:
            colour_by_column = clustering_name_column if display_mode == "Name" else clustering_column

            fig = visualize_embeddings(
                df_total,
                x_col=x_col,
                y_col=y_col,
                z_col=z_col,
                review_text_column='sentence',
                colour_by_column=colour_by_column
            )
            st.plotly_chart(fig)

        st.subheader(f"Cluster Details for Cluster {display_mode}: {selected_cluster_value}")
        st.dataframe(filtered_df[['topic', 'sentence', 'category', 'sentiment',
                                  'Please rate your overall experience playing Into the Dead: Our Darkest Days']])

        st.subheader("Cluster Sentiment and Request Distribution")
        col1, col2 = st.columns(2)

        with col1:
            fig_sentiment = plot_diverging_sentiments(df_total, sentiment_col='sentiment', cluster_name_col=colour_by_column)
            st.plotly_chart(fig_sentiment)

        with col2:
            fig_request_count = plot_request_count_by_cluster(df_total, cluster_name_col=colour_by_column)
            st.plotly_chart(fig_request_count)
else:
    st.warning("Please upload a JSON file to continue.")
