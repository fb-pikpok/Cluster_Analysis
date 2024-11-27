import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative



# Function to create diverging sentiment plot by cluster
def plot_sentiments(df, sentiment_col, cluster_name_col):
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

def visualize_embeddings(df, x_col, y_col, z_col=None, review_text_column=None, colour_by_column=None, color_map=None):
    """
    Visualize embeddings in 2D or 3D with consistent colors for clusters.

    Parameters:
    - df: DataFrame containing the data to visualize.
    - x_col: Column name for the x-axis.
    - y_col: Column name for the y-axis.
    - z_col: Optional column name for the z-axis (for 3D visualization).
    - review_text_column: Column name for the review text to display on hover.
    - colour_by_column: Column name for the values to color by.
    - color_map: Dictionary mapping cluster values to colors.

    Returns:
    - fig: A Plotly figure object.
    """
    if z_col:
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=colour_by_column,
            hover_data={review_text_column: True} if review_text_column else {},
            color_discrete_map=color_map  # Use the color map
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=colour_by_column,
            hover_data={review_text_column: True} if review_text_column else {},
            color_discrete_map=color_map  # Use the color map
        )

    # General layout adjustments
    fig.update_layout(
        legend_title_text=None,
        height=600,  # Set height for visualization
        width=900,   # Set width for visualization
        title=f"Visualization of {x_col}, {y_col}" + (f", {z_col}" if z_col else ""),
        xaxis=dict(title="", showgrid=True, zeroline=True, showticklabels=True),
        yaxis=dict(title="", showgrid=True, zeroline=True, showticklabels=True)
    )

    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="DarkSlateGrey")))

    return fig




def generate_color_map(dataframe, clustering_column, clustering_name_column, display_mode, palette=None):
    """
    Generates a color map for clusters in the dataframe.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing cluster information.
        clustering_column (str): The column name for cluster IDs.
        clustering_name_column (str): The column name for cluster names.
        display_mode (str): "ID" for cluster IDs, "Name" for cluster names.
        palette (list): Optional custom color palette. Defaults to Plotly's qualitative Set2.

    Returns:
        dict: A dictionary mapping unique clusters to colors.
    """
    if palette is None:
        palette = qualitative.Set2  # Default Plotly color palette

    max_colors = len(palette)

    # Determine unique clusters based on the display mode
    if display_mode == "Name":
        unique_clusters = sorted(dataframe[clustering_name_column].dropna().unique())
    else:
        unique_clusters = sorted(dataframe[clustering_column].dropna().unique())

    # Create the color map dictionary
    color_map = {cluster: palette[i % max_colors] for i, cluster in enumerate(unique_clusters)}

    return color_map


