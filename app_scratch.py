# region All visuals Work but color scheme is off

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Set page layout to wide
st.set_page_config(layout="wide")

# Define path to precomputed JSON file
s_root = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis/'
s_db_table_hdbscan_json = 'Data/review_db_table_hdbscan.json'


# Load precomputed data with UMAP, HDBSCAN results, and cluster IDs
@st.cache_data(show_spinner=False)
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df


df_total = load_data(s_root + s_db_table_hdbscan_json)

# Sidebar options
view_options = ["2D UMAP", "3D UMAP"]
selected_view = st.sidebar.radio("Select View", view_options)

# Sidebar filters
st.sidebar.header("Filters")

# Select cluster by ID (HDBSCAN or KMeans)
unique_cluster_ids = sorted(df_total['hdbscan_cluster_id'].unique())
selected_cluster_id = st.sidebar.selectbox("Select HDBSCAN Cluster ID", unique_cluster_ids)
selected_sentiment = st.sidebar.multiselect("Select Sentiment", df_total['sentiment'].unique(),
                                            default=df_total['sentiment'].unique())

# Filter data based on selected cluster ID and sentiments
filtered_df = df_total[
    (df_total['hdbscan_cluster_id'] == selected_cluster_id) & (df_total['sentiment'].isin(selected_sentiment))]


# Function to visualize embeddings
def visualize_embeddings(df, x_col, y_col, review_text_column, colour_by_column):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=colour_by_column,
        hover_data={review_text_column: True},
    )

    fig.update_layout(
        legend_title_text=None,
        height=600,
        width=900
    )

    fig.update_traces(
        marker=dict(size=6, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers+text"),
    )

    fig.update_xaxes(title="", showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(title="", showgrid=False, zeroline=False, showticklabels=False)

    return fig


# Function to create diverging sentiment plot by cluster
def plot_diverging_sentiments(df, sentiment_col, cluster_id_col):
    sentiment_data = df[df[sentiment_col].isin(['Positive', 'Negative'])]
    sentiment_counts = sentiment_data.groupby([cluster_id_col, sentiment_col]).size().unstack(fill_value=0)
    sentiment_counts['Positive'] = sentiment_counts.get('Positive', 0)
    sentiment_counts['Negative'] = -sentiment_counts.get('Negative', 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sentiment_counts.index,
        x=sentiment_counts['Positive'],
        orientation='h',
        name='Positive',
        marker=dict(color='green')
    ))
    fig.add_trace(go.Bar(
        y=sentiment_counts.index,
        x=sentiment_counts['Negative'],
        orientation='h',
        name='Negative',
        marker=dict(color='red')
    ))

    fig.update_layout(
        title="Sentiment Frequency by Cluster ID",
        xaxis_title="Sentiment Frequency",
        yaxis_title="Cluster ID",
        barmode='relative',
        showlegend=True,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    return fig


# Function to plot number of requests per cluster
def plot_request_count_by_cluster(df, cluster_id_col):
    request_counts = df[cluster_id_col].value_counts().reset_index()
    request_counts.columns = [cluster_id_col, 'Request Count']

    fig = px.bar(
        request_counts,
        x=cluster_id_col,
        y='Request Count',
        title="Number of Requests per Cluster",
        labels={cluster_id_col: "Cluster ID", 'Request Count': "Count"},
        text='Request Count'
    )

    fig.update_layout(
        xaxis_title="Cluster ID",
        yaxis_title="Request Count",
        showlegend=False
    )

    return fig


# Display UMAP and Cluster Details with larger dimensions
st.subheader("2D UMAP Cluster Visualization" if selected_view == "2D UMAP" else "3D UMAP Cluster Visualization")
if selected_view == "2D UMAP":
    fig = visualize_embeddings(df_total, x_col='umap_x', y_col='umap_y', review_text_column='sentence',
                               colour_by_column='hdbscan_cluster_id')
else:
    fig = px.scatter_3d(
        df_total,
        x='umap_x', y='umap_y', z='umap_z',
        color='hdbscan_cluster_id',
        hover_data={'sentence': True}
    )
    fig.update_layout(
        legend_title_text=None,
        showlegend=True,
        height=600,
        width=900
    )
    fig.update_traces(marker=dict(size=3, line=dict(width=1, color="DarkSlateGrey")))

st.plotly_chart(fig)

# Display Cluster Details Table in an expanded view
st.subheader(f"Cluster Details for HDBSCAN Cluster ID '{selected_cluster_id}'")
st.dataframe(filtered_df[
                 ['Please rate your overall experience playing Into the Dead: Our Darkest Days', 'topic', 'sentence',
                  'ID', 'category', 'sentiment', 'similarity']])

# Display Sentiment Frequency and Request Count plots side by side
st.subheader("Cluster Sentiment and Request Distribution")
col1, col2 = st.columns(2)

with col1:
    fig_sentiment = plot_diverging_sentiments(df_total, sentiment_col='sentiment', cluster_id_col='hdbscan_cluster_id')
    st.plotly_chart(fig_sentiment)

with col2:
    fig_request_count = plot_request_count_by_cluster(df_total, cluster_id_col='hdbscan_cluster_id')
    st.plotly_chart(fig_request_count)

# Additional 3D visualizations for KMeans, PCA, and t-SNE
st.subheader("Additional Cluster Visualizations (KMeans, PCA, t-SNE)")

for method in ["KMeans", "PCA", "t-SNE"]:
    st.write(f"### {method} Visualization")
    view_2d_or_3d = st.radio(f"Select {method} View", ["2D", "3D"], index=0, key=method)

    if method == "KMeans":
        if view_2d_or_3d == "2D":
            fig_kmeans = visualize_embeddings(df_total, x_col='umap_x', y_col='umap_y', review_text_column='sentence',
                                              colour_by_column='kmeans_cluster_id')
        else:
            fig_kmeans = px.scatter_3d(df_total, x='umap_x', y='umap_y', z='umap_z', color='kmeans_cluster_id',
                                       hover_data={'sentence': True})
    elif method == "PCA":
        if view_2d_or_3d == "2D":
            fig_pca = visualize_embeddings(df_total, x_col='pca_x', y_col='pca_y', review_text_column='sentence',
                                           colour_by_column='kmeans_cluster_id')
        else:
            fig_pca = px.scatter_3d(df_total, x='pca_x', y='pca_y', z='pca_z', color='kmeans_cluster_id',
                                    hover_data={'sentence': True})
    else:  # t-SNE
        if view_2d_or_3d == "2D":
            fig_tsne = visualize_embeddings(df_total, x_col='tsne_x', y_col='tsne_y', review_text_column='sentence',
                                            colour_by_column='kmeans_cluster_id')
        else:
            fig_tsne = px.scatter_3d(df_total, x='tsne_x', y='tsne_y', z='tsne_z', color='kmeans_cluster_id',
                                     hover_data={'sentence': True})

    st.plotly_chart(fig_kmeans if method == "KMeans" else fig_pca if method == "PCA" else fig_tsne)

# endregion

# All fine but the 3 new graphs are missing



# endregion


# region everything of Kmeans works somewhat when clusters have names

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

# Helper
from st_source.visuals import visualize_embeddings, plot_diverging_sentiments, plot_request_count_by_cluster


# Set page layout to wide
st.set_page_config(layout="wide")


# Define path to precomputed JSON file
s_root = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis/'
s_db_table_hdbscan_json = 'Data/review_db_table_hdbscan.json'               # Data that is already precomputed
s_db_embed = 'Data/review_db_table.json'                                    # RAW data cluster analysis is performed live


# Load precomputed data with UMAP, HDBSCAN results, and cluster names
@st.cache_data(show_spinner=False)
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

df_total = load_data(s_root + s_db_table_hdbscan_json)

# Sidebar options
view_options = ["2D UMAP", "3D UMAP"]
selected_view = st.sidebar.radio("Select View", view_options)

# Sidebar filters
st.sidebar.header("Filters")

# Select cluster by name rather than ID
unique_cluster_names = sorted(df_total['cluster_name'].unique())
selected_cluster_name = st.sidebar.selectbox("Select Cluster", unique_cluster_names)
selected_sentiment = st.sidebar.multiselect("Select Sentiment", df_total['sentiment'].unique(),
                                            default=df_total['sentiment'].unique())

# Filter data based on selected cluster name and sentiments
filtered_df = df_total[(df_total['cluster_name'] == selected_cluster_name) & (df_total['sentiment'].isin(selected_sentiment))]



# Function to plot number of requests per cluster
def plot_request_count_by_cluster(df, cluster_name_col):
    request_counts = df[cluster_name_col].value_counts().reset_index()
    request_counts.columns = [cluster_name_col, 'Request Count']

    fig = px.bar(
        request_counts,
        x=cluster_name_col,
        y='Request Count',
        title="Number of Requests per Cluster",
        labels={cluster_name_col: "Cluster Name", 'Request Count': "Count"},
        text='Request Count'
    )

    fig.update_layout(
        xaxis_title="Cluster Name",
        yaxis_title="Request Count",
        showlegend=False
    )

    return fig

# Display UMAP and Cluster Details with larger dimensions
st.subheader("2D UMAP Cluster Visualization" if selected_view == "2D UMAP" else "3D UMAP Cluster Visualization")
if selected_view == "2D UMAP":
    df_total['coords'] = df_total[['umap_x', 'umap_y']].values.tolist()
    fig = visualize_embeddings(df_total, coords_col='coords', review_text_column='sentence', colour_by_column='cluster_name')
else:
    df_total['coords'] = df_total[['umap_x', 'umap_y', 'umap_z']].values.tolist()
    fig = px.scatter_3d(
        df_total,
        x='umap_x', y='umap_y', z='umap_z',
        color='cluster_name',
        hover_data={'sentence': True}
    )
    fig.update_layout(
        legend_title_text=None,
        showlegend=True,
        height=600,  # Larger height for 3D plot
        width=900    # Larger width for 3D plot
    )
    fig.update_traces(marker=dict(size=3, line=dict(width=1, color="DarkSlateGrey")))

st.plotly_chart(fig)

# Display Cluster Details Table in an expanded view
st.subheader(f"Cluster Details for '{selected_cluster_name}'")
st.dataframe(filtered_df[['Please rate your overall experience playing Into the Dead: Our Darkest Days','topic', 'sentence', 'category', 'sentiment', 'similarity']])

# Display Sentiment Frequency and Request Count plots side by side
st.subheader("Cluster Sentiment and Request Distribution")
col1, col2 = st.columns(2)

with col1:
    fig_sentiment = plot_diverging_sentiments(df_total, sentiment_col='sentiment', cluster_name_col='cluster_name')
    st.plotly_chart(fig_sentiment)

with col2:
    fig_request_count = plot_request_count_by_cluster(df_total, cluster_name_col='cluster_name')
    st.plotly_chart(fig_request_count)


# Additional 2D and 3D visualizations for PCA and t-SNE
st.subheader("Additional Dimensionality Reduction Visualizations")

# Function to visualize embeddings
def visualize_embeddings_2(df, x_col, y_col, color_col, z_col=None, title="Embedding Visualization"):
    if z_col:
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, hover_data={'sentence': True})
    else:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, hover_data={'sentence': True})

    fig.update_layout(
        legend_title_text=None,
        height=600,  # Increase height for larger visualization
        width=900,   # Increase width for larger visualization
        title=title
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=2, color="DarkSlateGrey")))
    return fig


for method in ["PCA", "tSNE"]:
    st.write(f"### {method} Visualization")
    view_option = st.radio(f"Select {method} View", ["2D", "3D"], index=0, key=method)

    x_col = f'{method.lower()}_x'
    y_col = f'{method.lower()}_y'
    z_col = f'{method.lower()}_z' if view_option == "3D" else None

    fig = visualize_embeddings_2(df_total, x_col=x_col, y_col=y_col, color_col='cluster_name', z_col=z_col, title=f"{method} {view_option} Visualization")
    st.plotly_chart(fig)

# endregion
