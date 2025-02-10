########################
# visualisations.py
########################

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



# region Shared Helper Functions for Sentiment Counts

def compute_sentiment_counts_by_cluster(df, cluster_col, sentiment_col="sentiment"):
    """
    Computes positive and negative sentiment counts per cluster, ignoring all other sentiments.

    Args:
        df (pd.DataFrame): Input DataFrame (already filtered if needed).
        cluster_col (str): Column indicating the cluster name/ID.
        sentiment_col (str): Column with sentiment labels (e.g., 'Positive', 'Negative').

    Returns:
        pd.DataFrame: Index = cluster values, columns = ['Positive', 'Negative'] (raw counts).
                      Does not flip negatives to negative yet.
                      If there's no Positive/Negative data, returns an empty DataFrame.
    """
    # Filter for Positive & Negative
    filtered = df[df[sentiment_col].isin(["Positive", "Negative"])].copy()
    if filtered.empty:
        return pd.DataFrame(columns=["Positive", "Negative"])

    # Group by [cluster, sentiment], then pivot
    grouped = filtered.groupby([cluster_col, sentiment_col]).size().unstack(fill_value=0)
    # Ensure we have both columns
    if "Positive" not in grouped.columns:
        grouped["Positive"] = 0
    if "Negative" not in grouped.columns:
        grouped["Negative"] = 0

    # Reorder columns for consistency
    return grouped[["Positive", "Negative"]]


def compute_sentiment_over_time(df, time_col="timestamp_updated", cluster_col=None, granularity="M"):
    """
    Aggregates sentiment counts (Positive, Negative) over time, optionally grouped by cluster.

    Args:
        df (pd.DataFrame): DataFrame with 'sentiment' and a datetime column.
                           Should already be filtered for date range, clusters, etc.
        time_col (str): Datetime-like column name. Defaults to 'timestamp_updated'.
        cluster_col (str or None): If provided, group by cluster and time.
                                   If None, aggregate all clusters together.
        granularity (str): Pandas offset alias for resampling (e.g. 'M', 'W', 'D', 'H').

    Returns:
        pd.DataFrame:
            - If cluster_col is None => columns: [time_col, positive_count, total_count, negative_count]
            - If cluster_col is set => columns: [cluster_col, time_col, positive_count, total_count, negative_count]
            negative_count is stored as negative integers to facilitate “diverging” plots.
            Returns empty DataFrame if no Positive/Negative data.
    """
    # Keep only Positive / Negative
    df = df[df["sentiment"].isin(["Positive", "Negative"])].copy()
    if df.empty:
        return pd.DataFrame()

    # Make sure time_col is the index for resampling
    df = df.set_index(time_col).sort_index()

    # Numeric indicator: 1 if Positive, 0 otherwise
    df["is_positive"] = (df["sentiment"] == "Positive").astype(int)

    # Group & resample
    if cluster_col:
        grouped = (
            df.groupby(cluster_col)
              .resample(granularity)
              .agg({"is_positive": ["sum", "count"]})
        )
        grouped.columns = ["positive_count", "total_count"]
        grouped.reset_index(inplace=True)
    else:
        # Single aggregated approach
        grouped = df.resample(granularity).agg({"is_positive": ["sum", "count"]})
        grouped.columns = ["positive_count", "total_count"]
        grouped.reset_index(inplace=True)

    # negative_count = total_count - positive_count (store as negative for diverging plots)
    grouped["negative_count"] = grouped["total_count"] - grouped["positive_count"]
    grouped["negative_count"] = -grouped["negative_count"]

    return grouped


def generate_color_map(dataframe, clustering_column, clustering_name_column, display_mode):
    """
    Generates a color map for clusters using a fixed list of CSS color names.
    Ensures consistent coloring for up to len(css_colors) clusters.

    Args:
        dataframe (pd.DataFrame): Input with cluster columns.
        clustering_column (str): Column for cluster IDs.
        clustering_name_column (str): Column for cluster names.
        display_mode (str): "ID" or "Name".

    Returns:
        dict: Mapping {cluster_value: css_color}.
    """
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

    if display_mode == "Name":
        unique_clusters = sorted(dataframe[clustering_name_column].dropna().unique())
    else:
        unique_clusters = sorted(dataframe[clustering_column].dropna().unique())

    num_clusters = len(unique_clusters)
    if num_clusters > len(css_colors):
        raise ValueError(f"Too many clusters! Maximum supported: {len(css_colors)}, but got {num_clusters}.")

    color_map = {cluster: css_colors[i] for i, cluster in enumerate(unique_clusters)}
    return color_map


#endregion


# region Plotting Functions

def plot_sentiments(df, sentiment_col, cluster_name_col):
    """
    Creates a horizontal diverging bar chart (Positive vs Negative) per cluster.

    Args:
        df (pd.DataFrame): DataFrame, already filtered if needed.
        sentiment_col (str): Column with sentiment labels ('Positive', 'Negative').
        cluster_name_col (str): Column with cluster name/ID.

    Returns:
        plotly.graph_objects.Figure
    """
    # Use the shared helper to get raw counts
    sentiment_counts = compute_sentiment_counts_by_cluster(
        df, cluster_col=cluster_name_col, sentiment_col=sentiment_col
    )

    if sentiment_counts.empty:
        fig = go.Figure()
        fig.update_layout(title="No Positive/Negative sentiment data available.")
        return fig

    # Flip negatives to appear on the left side
    sentiment_counts["Negative"] = -sentiment_counts["Negative"]

    # Build diverging bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sentiment_counts.index,
        x=sentiment_counts["Positive"],
        orientation='h',
        name='Positive',
        marker=dict(color='green')
    ))
    fig.add_trace(go.Bar(
        y=sentiment_counts.index,
        x=sentiment_counts["Negative"],
        orientation='h',
        name='Negative',
        marker=dict(color='red')
    ))

    fig.update_layout(
        title="Sentiment Frequency by Cluster Name",
        xaxis_title="Sentiment Frequency",
        yaxis_title="Cluster Name",
        barmode='relative',  # bars overlap at zero
        showlegend=True,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=False, zeroline=False)
    )

    return fig


def plot_sentiments_over_time_line(aggregated_df, cluster_col=None, color_map=None, title="Time-based Sentiment Analysis"):
    """
    Builds a line chart (positive vs negative) over time using the result of compute_sentiment_over_time().

    Args:
        aggregated_df (pd.DataFrame): result of compute_sentiment_over_time()
        cluster_col (str or None): if None => single aggregated line,
                                   else => multiple lines by cluster
        color_map (dict): mapping of cluster -> color
        title (str): chart title

    Returns:
        plotly.graph_objects.Figure or None
    """
    if aggregated_df.empty:
        return None

    # Distinguish "all clusters" vs multi-cluster scenario
    is_multi_cluster = (cluster_col is not None and cluster_col in aggregated_df.columns)

    if not is_multi_cluster:
        # Single aggregated approach => 2 lines total
        plot_df = aggregated_df.melt(
            id_vars=["timestamp_updated"],
            value_vars=["positive_count", "negative_count"],
            var_name="Sentiment",
            value_name="Count"
        )
        fig = px.line(
            plot_df,
            x="timestamp_updated",
            y="Count",
            color="Sentiment",
            title=title,
            labels={
                "timestamp_updated": "Time",
                "Count": "Count (+ = Positive, - = Negative)"
            }
        )
    else:
        # Multi-cluster => group by cluster_col => 2 lines (pos & neg) per cluster
        plot_df = aggregated_df.melt(
            id_vars=[cluster_col, "timestamp_updated"],
            value_vars=["positive_count", "negative_count"],
            var_name="Sentiment",
            value_name="Count"
        )
        fig = px.line(
            plot_df,
            x="timestamp_updated",
            y="Count",
            color=cluster_col,
            line_dash="Sentiment",  # dashed line for negative
            color_discrete_map=color_map if color_map else {},
            title=title,
            labels={
                "timestamp_updated": "Time",
                "Count": "Count (+ = Positive, - = Negative)",
                cluster_col: "Cluster"
            }
        )

    fig.update_layout(
        yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    )
    return fig


def plot_sentiments_over_time(df, sentiment_col, month_col):
    """
    Creates a diverging bar chart for Positive vs Negative sentiments over time (by 'month' or any period label).

    Args:
        df (pd.DataFrame): DataFrame containing sentiment & time-based column.
        sentiment_col (str): Name of the sentiment column.
        month_col (str): Name of a "time label" column (e.g., 'month').

    Returns:
        plotly.graph_objects.Figure
    """
    # Filter for positive & negative
    sentiment_data = df[df[sentiment_col].isin(['Positive', 'Negative'])]
    if sentiment_data.empty:
        fig = go.Figure()
        fig.update_layout(title="No Positive/Negative data for the specified time period.")
        return fig

    # Group by time label & sentiment
    sentiment_counts = sentiment_data.groupby([month_col, sentiment_col]).size().unstack(fill_value=0)
    sentiment_counts['Positive'] = sentiment_counts.get('Positive', 0)
    sentiment_counts['Negative'] = -sentiment_counts.get('Negative', 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sentiment_counts.index.astype(str),  # e.g. converting PeriodIndex to string
        y=sentiment_counts['Positive'],
        name='Positive',
        marker=dict(color='green')
    ))
    fig.add_trace(go.Bar(
        x=sentiment_counts.index.astype(str),
        y=sentiment_counts['Negative'],
        name='Negative',
        marker=dict(color='red')
    ))

    fig.update_layout(
        title="Aggregated Bar Chart",
        xaxis_title="Time",
        yaxis_title="Sentiment Frequency",
        barmode='relative',
        showlegend=True,
        xaxis=dict(showgrid=True, zeroline=True),
        yaxis=dict(showgrid=False, zeroline=False)
    )
    return fig


def plot_request_count_by_cluster(df, cluster_name_col, data_type):
    """
    Creates a bar chart for the count of a specific category ('request', 'fact', or both) per cluster.

    Args:
        df (pd.DataFrame): DataFrame (already filtered if needed).
        cluster_name_col (str): Name of the cluster column.
        data_type (str): 'requests', 'facts', or 'both'.

    Returns:
        plotly.graph_objects.Figure
    """
    if data_type == "requests":
        filtered_df = df[df['category'] == 'request']
    elif data_type == "facts":
        filtered_df = df[df['category'] == 'fact']
    else:
        filtered_df = df  # 'both': no filter

    counts = filtered_df[cluster_name_col].value_counts().reset_index()
    counts.columns = [cluster_name_col, 'Count']

    fig = px.bar(
        counts,
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

    Args:
        df (pd.DataFrame): Data with columns [x_col, y_col, z_col(optional), cluster].
        x_col (str): x-axis column name.
        y_col (str): y-axis column name.
        z_col (str or None): z-axis column for 3D. If None => 2D.
        review_text_column (str or None): If provided, included in hover data.
        colour_by_column (str or None): The column for color-coding (e.g., cluster).
        color_map (dict or None): Mapping cluster -> color.

    Returns:
        plotly.graph_objects.Figure
    """
    hover_data = {}
    if review_text_column:
        hover_data[review_text_column] = True
    if 'topic' in df.columns:
        hover_data['topic'] = True

    if z_col:
        fig = px.scatter_3d(
            df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=colour_by_column,
            hover_data=hover_data,
            color_discrete_map=color_map
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=colour_by_column,
            hover_data=hover_data,
            color_discrete_map=color_map
        )

    fig.update_layout(
        legend_title_text=None,
        height=600,
        width=900,
        title=f"Visualization of {x_col}, {y_col}" + (f", {z_col}" if z_col else ""),
        xaxis=dict(title="", showgrid=True, zeroline=True, showticklabels=True),
        yaxis=dict(title="", showgrid=True, zeroline=True, showticklabels=True)
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0.5, color="DarkSlateGrey")))

    return fig

#endregion



# region Here are the leftovers from previous versions of the color MAP


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


# endregion
