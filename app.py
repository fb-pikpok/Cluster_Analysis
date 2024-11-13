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

# Function to visualize embeddings
def visualize_embeddings(df, coords_col, review_text_column, colour_by_column):
    if any(len(coords) != 2 for coords in df[coords_col]):
        raise ValueError(f"Each entry in '{coords_col}' must have exactly 2 elements (x and y coordinates)")

    temp_df = df.copy()
    temp_df[["x", "y"]] = temp_df[coords_col].to_list()

    fig = px.scatter(
        temp_df,
        x="x",
        y="y",
        color=colour_by_column,
        hover_data={review_text_column: True},
    )

    fig.update_layout(
        legend_title_text=None,
        height=600,  # Increase height for larger visualization
        width=900    # Increase width for larger visualization
    )

    fig.update_traces(
        marker=dict(size=6, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers+text"),
    )

    for trace in fig.data:
        if trace.name == "-1":
            trace.visible = "legendonly"

    fig.update_xaxes(title="", showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(title="", showgrid=False, zeroline=False, showticklabels=False)

    return fig

# Function to create diverging sentiment plot by cluster
def plot_diverging_sentiments(df, sentiment_col, cluster_name_col):
    # Filter for positive and negative sentiments only
    sentiment_data = df[df[sentiment_col].isin(['Positive', 'Negative'])]

    # Calculate sentiment counts
    sentiment_counts = sentiment_data.groupby([cluster_name_col, sentiment_col]).size().unstack(fill_value=0)

    # Separate positive and negative counts
    sentiment_counts['Positive'] = sentiment_counts.get('Positive', 0)
    sentiment_counts['Negative'] = -sentiment_counts.get('Negative', 0)  # Flip negative values for left-side plotting

    # Create a diverging bar chart
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
        title="Sentiment Frequency by Cluster Name",
        xaxis_title="Sentiment Frequency",
        yaxis_title="Cluster Name",
        barmode='relative',
        showlegend=True,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    return fig

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
