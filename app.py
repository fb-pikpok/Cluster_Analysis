import streamlit as st
import pandas as pd
import plotly.express as px
import json

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
selected_cluster = st.sidebar.selectbox("Select Cluster", sorted(df_total['cluster_id'].unique()))
selected_sentiment = st.sidebar.multiselect("Select Sentiment", df_total['sentiment'].unique(), default=df_total['sentiment'].unique())

# Filter data based on selections
filtered_df = df_total[(df_total['cluster_id'] == selected_cluster) & (df_total['sentiment'].isin(selected_sentiment))]

# Function to visualize embeddings
def visualize_embeddings(df, coords_col, review_text_column, colour_by_column):
    # Ensure each entry in coords_col has exactly 2 elements
    if any(len(coords) != 2 for coords in df[coords_col]):
        raise ValueError(f"Each entry in '{coords_col}' must have exactly 2 elements (x and y coordinates)")

    # Create a temporary DataFrame for plotting
    temp_df = df.copy()
    temp_df[["x", "y"]] = temp_df[coords_col].to_list()

    # Create the interactive plot
    fig = px.scatter(
        temp_df,
        x="x",
        y="y",
        color=colour_by_column,
        hover_data={review_text_column: True},
    )

    fig.update_layout(
        legend_title_text=None
    )

    # Customize the layout
    fig.update_traces(
        marker=dict(size=5, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers+text"),
    )

    # Hide noise clusters by default (if any)
    for trace in fig.data:
        if trace.name == "-1":  # Assuming -1 is the noise label
            trace.visible = "legendonly"

    # Remove axis labels and grid lines
    fig.update_xaxes(title="", showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(title="", showgrid=False, zeroline=False, showticklabels=False)

    return fig

# Function to plot reviews over time
def plot_over_time(df, date_col):
    daily_counts = (
        pd.to_datetime(df[date_col])
        .dt.floor("D")
        .to_frame()
        .groupby(date_col)
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(daily_counts, x=date_col, y="count")
    fig.update_yaxes(title="", showgrid=False, zeroline=False, showticklabels=False)
    return fig

# Select coordinates column based on view
if selected_view == "2D UMAP":
    df_total['coords'] = df_total[['umap_x', 'umap_y']].values.tolist()
    st.subheader("2D UMAP Cluster Visualization")
    fig = visualize_embeddings(df_total, coords_col='coords', review_text_column='sentence', colour_by_column='cluster_name')
else:
    df_total['coords'] = df_total[['umap_x', 'umap_y', 'umap_z']].values.tolist()
    st.subheader("3D UMAP Cluster Visualization")
    fig = px.scatter_3d(
        df_total,
        x='umap_x', y='umap_y', z='umap_z',
        color='cluster_name',
        hover_data={'sentence': True}
    )

    # Hide axis labels and set color options
    fig.update_layout(
        legend_title_text=None,
        showlegend=True
    )
    fig.update_traces(marker=dict(size=3, line=dict(width=1, color="DarkSlateGrey")))

# Display the plot in Streamlit
st.plotly_chart(fig)

# Display Cluster Details
st.subheader("Cluster Details")
st.write(f"Showing details for cluster {selected_cluster}")
st.dataframe(filtered_df[['cluster_name', 'topic', 'sentence', 'similarity', 'category', 'sentiment']])


# Function to plot sentiment frequency by cluster name
def plot_sentiment_frequency_by_cluster(df, sentiment_col, cluster_name_col):
    sentiment_counts = df.groupby([cluster_name_col, sentiment_col]).size().reset_index(name='count')

    fig = px.bar(
        sentiment_counts,
        x=cluster_name_col,
        y='count',
        color=sentiment_col,
        barmode='group',
        labels={cluster_name_col: "Cluster Name", sentiment_col: "Sentiment", 'count': 'Frequency'}
    )

    fig.update_layout(
        title="Sentiment Frequency by Cluster Name",
        xaxis_title="Cluster Name",
        yaxis_title="Frequency",
        legend_title_text="Sentiment",
        showlegend=True
    )

    return fig

# Plot sentiment frequency by cluster
st.subheader("Sentiment Frequency by Cluster Name")
fig_sentiment = plot_sentiment_frequency_by_cluster(df_total, sentiment_col='sentiment', cluster_name_col='cluster_name')
st.plotly_chart(fig_sentiment)