import _plotly_utils.colors
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import qualitative
import pandas as pd



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

def plot_request_count_by_cluster(df, cluster_name_col, data_type):
    """
    Plots a bar chart for facts, requests, or both (cluster size) per cluster.

    Args:
        df (pd.DataFrame): Filtered DataFrame.
        cluster_name_col (str): Name of the cluster column.
        data_type (str): Type of data to display ("requests", "facts", or "both").

    Returns:
        plotly.graph_objects.Figure: Bar chart.
    """
    # Filter the DataFrame based on the selected category
    if data_type == "requests":
        filtered_df = df[df['category'] == 'request']
    elif data_type == "facts":
        filtered_df = df[df['category'] == 'fact']
    elif data_type == "both":
        filtered_df = df  # Include all categories

    # Count occurrences per cluster
    data = filtered_df[cluster_name_col].value_counts().reset_index()
    data.columns = [cluster_name_col, 'Count']

    # Create the bar chart
    fig = px.bar(
        data,
        x=cluster_name_col,
        y='Count',
        title=f"Number of {data_type.capitalize()} per Cluster",
        labels={cluster_name_col: "Cluster Name", 'Count': "Count"},
        text='Count'
    )

    fig.update_layout(
        xaxis_title="Cluster Name",
        yaxis_title="Count",
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
    # Add hover data dynamically
    hover_data = {}
    if review_text_column:
        hover_data[review_text_column] = True  # Include the review text
    if 'topic' in df.columns:
        hover_data['topic'] = True  # Include the topic, if it exists

    if z_col:
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=colour_by_column,
            hover_data=hover_data,
            color_discrete_map=color_map  # Use the color map
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=colour_by_column,
            hover_data=hover_data,
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



import matplotlib.cm as cm
import matplotlib.colors as mcolors

# def generate_color_map(dataframe, clustering_column, clustering_name_column, display_mode, colormap_name='tab20'):
#     """
#     Dynamically generates a color map for clusters, ensuring enough unique colors for all clusters.
#
#     Args:
#         dataframe (pd.DataFrame): The input dataframe containing cluster information.
#         clustering_column (str): The column name for cluster IDs.
#         clustering_name_column (str): The column name for cluster names.
#         display_mode (str): "ID" for cluster IDs, "Name" for cluster names.
#         colormap_name (str): The name of the matplotlib colormap to use. Defaults to 'tab20'.
#
#     Returns:
#         dict: A dictionary mapping unique clusters to colors.
#     """
#     # Determine unique clusters based on the display mode
#     if display_mode == "Name":
#         unique_clusters = sorted(dataframe[clustering_name_column].dropna().unique())
#     else:
#         unique_clusters = sorted(dataframe[clustering_column].dropna().unique())
#
#     num_clusters = len(unique_clusters)
#
#     # Generate distinct colors using HSV color space
#     colors = [
#         mcolors.to_hex(mcolors.hsv_to_rgb((i / num_clusters, 0.8, 0.8)))  # Hue, Saturation, Value
#         for i in range(num_clusters)
#     ]
#
#     # Create the color map dictionary
#     color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
#
#     return color_map


# def generate_color_map(dataframe, clustering_column, clustering_name_column, display_mode, palette=None):
#     """
#     Generates a color map for clusters in the dataframe.
#
#     Args:
#         dataframe (pd.DataFrame): The input dataframe containing cluster information.
#         clustering_column (str): The column name for cluster IDs.
#         clustering_name_column (str): The column name for cluster names.
#         display_mode (str): "ID" for cluster IDs, "Name" for cluster names.
#         palette (list): Optional custom color palette. Defaults to Plotly's qualitative Set2.
#
#     Returns:
#         dict: A dictionary mapping unique clusters to colors.
#     """
#     if palette is None:
#         palette = _plotly_utils.colors.qualitative.__all__
#
#
#
#     max_colors = len(palette)
#
#     # Determine unique clusters based on the display mode
#     if display_mode == "Name":
#         unique_clusters = sorted(dataframe[clustering_name_column].dropna().unique())
#     else:
#         unique_clusters = sorted(dataframe[clustering_column].dropna().unique())
#
#     # Create the color map dictionary
#     color_map = {cluster: palette[i % max_colors] for i, cluster in enumerate(unique_clusters)}
#
#     return color_map


def generate_color_map(dataframe, clustering_column, clustering_name_column, display_mode):
    """
    Generates a color map for clusters using CSS color names.

    Args:
        dataframe (pd.DataFrame): The input dataframe containing cluster information.
        clustering_column (str): The column name for cluster IDs.
        clustering_name_column (str): The column name for cluster names.
        display_mode (str): "ID" for cluster IDs, "Name" for cluster names.

    Returns:
        dict: A dictionary mapping unique clusters to CSS colors.
    """
    # List of CSS color names
    css_colors = [
        "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black", "blanchedalmond", "blue",
        "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue", "cornsilk",
        "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgrey", "darkgreen", "darkkhaki",
        "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue",
        "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dimgrey",
        "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite", "gold", "goldenrod",
        "gray", "grey", "green", "greenyellow", "honeydew", "hotpink", "indianred", "indigo", "ivory", "khaki", "lavender",
        "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow",
        "lightgray", "lightgrey", "lightgreen", "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray",
        "lightslategrey", "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
        "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise",
        "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive",
        "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred",
        "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "red", "rosybrown", "royalblue",
        "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver", "skyblue", "slateblue",
        "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan", "teal", "thistle", "tomato", "turquoise",
        "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"
    ]

    # Determine unique clusters based on the display mode
    if display_mode == "Name":
        unique_clusters = sorted(dataframe[clustering_name_column].dropna().unique())
    else:
        unique_clusters = sorted(dataframe[clustering_column].dropna().unique())

    # Check if there are more clusters than colors
    num_clusters = len(unique_clusters)
    if num_clusters > len(css_colors):
        raise ValueError(f"Too many clusters! Maximum supported: {len(css_colors)}, but got {num_clusters}.")

    # Assign colors to clusters
    color_map = {cluster: css_colors[i] for i, cluster in enumerate(unique_clusters)}

    return color_map

# region Sentiment over time

def get_sentiment_over_time(df):
    """
    Groups data by month and counts positive and negative sentiments.

    Args:
        df (pd.DataFrame): Filtered DataFrame with sentiment and timestamp columns.

    Returns:
        pd.DataFrame: Aggregated DataFrame with sentiment counts by month.
    """
    if 'month' not in df.columns or 'sentiment' not in df.columns:
        return pd.DataFrame()  # Return an empty DataFrame if required columns are missing

    # Group by month and sentiment, then count occurrences
    sentiment_over_time = (
        df.groupby(['month', 'sentiment'])
        .size()
        .unstack(fill_value=0)  # Create columns for each sentiment
        .reset_index()
    )

    return sentiment_over_time


import plotly.graph_objects as go

def plot_sentiments_over_time(df, sentiment_col, month_col):
    """
    Creates a diverging bar chart for sentiment counts over time (by month).

    Args:
        df (pd.DataFrame): DataFrame containing sentiment and month data.
        sentiment_col (str): The name of the sentiment column.
        month_col (str): The name of the month column.

    Returns:
        plotly.graph_objects.Figure: Diverging bar chart for sentiment counts over time.
    """
    # Filter for positive and negative sentiments only
    sentiment_data = df[df[sentiment_col].isin(['Positive', 'Negative'])]

    # Calculate sentiment counts grouped by month
    sentiment_counts = sentiment_data.groupby([month_col, sentiment_col]).size().unstack(fill_value=0)

    # Separate positive and negative counts
    sentiment_counts['Positive'] = sentiment_counts.get('Positive', 0)
    sentiment_counts['Negative'] = -sentiment_counts.get('Negative', 0)  # Flip negative values for diverging bars

    # Create a diverging bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sentiment_counts.index.astype(str),  # Convert PeriodIndex to string for display
        y=sentiment_counts['Positive'],
        name='Positive',
        marker=dict(color='green')
    ))
    fig.add_trace(go.Bar(
        x=sentiment_counts.index.astype(str),  # Convert PeriodIndex to string for display
        y=sentiment_counts['Negative'],
        name='Negative',
        marker=dict(color='red')
    ))

    # Update layout for the chart
    fig.update_layout(
        title="Sentiment Over Time",
        xaxis_title="Month",
        yaxis_title="Sentiment Frequency",
        barmode='relative',  # Allow bars to diverge
        showlegend=True,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    return fig


