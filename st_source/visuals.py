import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# region Shared Helper Functions for Sentiment Counts
def compute_sentiment_counts_by_cluster(df, cluster_col, sentiment_col="sentiment"):
    """
    Counts the number of Positive and Negative sentiments per cluster.

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


def aggregate_sentiment_over_time(
    df,
    time_col="timestamp_updated",
    sentiment_col="sentiment",
    cluster_col=None,
    granularity="ME"
):
    """
    Aggregates sentiment counts (Positive, Negative) over time, with optional cluster grouping.
    Negative counts are stored as negative for diverging charts.

    Args:
        df (pd.DataFrame): Input data, already filtered as needed. Must contain 'sentiment' column.
        time_col (str): Name of a datetime column. Defaults to 'timestamp_updated'.
        sentiment_col (str): Name of the sentiment column (e.g., 'sentiment').
        cluster_col (str or None): If None => single aggregated. Otherwise => group by cluster & time.
        granularity (str): Pandas offset alias for resampling ('M', 'W', 'D', 'H', etc.).

    Returns:
        pd.DataFrame: If cluster_col is None, columns = [time_col, positive_count, negative_count, total_count].
                      Else, columns = [cluster_col, time_col, positive_count, negative_count, total_count].
                      Returns empty if no Positive/Negative data.
    """
    # Filter for Positive & Negative only
    df = df[df[sentiment_col].isin(["Positive", "Negative"])].copy()
    if df.empty:
        return pd.DataFrame()  # nothing to aggregate

    # Make sure the time_col is a proper datetime index
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.set_index(time_col).sort_index()

    # Create a numeric indicator for positivity
    df["is_positive"] = (df[sentiment_col] == "Positive").astype(int)

    # Group & resample
    if cluster_col:
        grouped = (
            df.groupby(cluster_col)
              .resample(granularity)["is_positive"]
              .agg(["sum", "count"])
              .rename(columns={"sum": "positive_count", "count": "total_count"})
              .reset_index()
        )
    else:
        grouped = (
            df.resample(granularity)["is_positive"]
              .agg(["sum", "count"])
              .rename(columns={"sum": "positive_count", "count": "total_count"})
              .reset_index()
        )

    # negative_count = total_count - positive_count, stored negative for diverging
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
        "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "gold", "goldenrod",
        "gray", "grey", "green", "greenyellow", "honeydew", "hotpink", "indianred", "indigo", "ivory", "khaki", "lavender",
        "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow",
        "lightgreen", "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray",
        "lightslategrey", "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
        "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise",
        "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive",
        "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred",
        "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "red", "rosybrown", "royalblue",
        "saddlebrown", "salmon", "seagreen", "seashell", "sienna", "silver", "skyblue", "slateblue",
        "slategray", "slategrey", "snow", "springgreen", "steelblue", "tan", "teal", "thistle", "tomato", "turquoise",
        "violet", "wheat", "yellow", "yellowgreen"
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

# Main Graph for Cluster Analysis
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


# Sentiment per Cluster (Bar Chart)
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


# Display cluster size (Bar Chart)
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




def plot_sentiment_over_time_bar(
    aggregated_df,
    cluster_col=None,
    title="Aggregated Bar Chart"
):
    """
    Creates a diverging bar chart for Positive vs Negative sentiment over time.
    If cluster_col is provided in aggregated_df, it will produce stacked groups by cluster.

    Args:
        aggregated_df (pd.DataFrame): Output of aggregate_sentiment_over_time.
        cluster_col (str or None): If None => single aggregated approach.
                                   If present => multiple clusters.
        title (str): Chart title.

    Returns:
        plotly.graph_objects.Figure or None
    """
    if aggregated_df.empty:
        return None

    # Distinguish single aggregated vs multi-cluster data
    is_multi_cluster = cluster_col and (cluster_col in aggregated_df.columns)

    # If single aggregated => We expect columns [timestamp_updated, positive_count, negative_count, total_count]
    if not is_multi_cluster:
        x_values = aggregated_df["timestamp_updated"]
        pos_vals = aggregated_df["positive_count"]
        neg_vals = aggregated_df["negative_count"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x_values,
            y=pos_vals,
            name="Positive",
            marker=dict(color="green")
        ))
        fig.add_trace(go.Bar(
            x=x_values,
            y=neg_vals,
            name="Negative",
            marker=dict(color="red")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Sentiment Frequency (Positive/Negative)",
            barmode='relative',  # diverging effect
            showlegend=True
        )
        return fig

    else:
        # Multi-cluster => columns: [cluster_col, timestamp_updated, positive_count, negative_count, total_count]
        # We'll pivot so each cluster is a separate group on the x-axis.
        # Or we can create a stacked bar for each cluster over time.
        # For simplicity, let's plot each time point grouped by cluster.

        # Convert to wide form: index=[timestamp, cluster], columns=[positive_count, negative_count]
        # Then we can create separate traces per cluster.
        # Alternatively, we iterate row by row in a custom approach.

        # We'll do a simpler approach: We'll meltdown, then manually create traces.
        # Might be more straightforward to interpret as time on x-axis, total bars per cluster at each time.
        # Each cluster has 2 stacked bars: positive, negative.

        # Step 1: melt
        melted = aggregated_df.melt(
            id_vars=[cluster_col, "timestamp_updated"],
            value_vars=["positive_count", "negative_count"],
            var_name="Sentiment",
            value_name="Count"
        )

        fig = go.Figure()

        # We'll group by cluster and sentiment, then plot them for each time
        for c_id, c_data in melted.groupby(cluster_col):
            # c_data has "timestamp_updated", "Sentiment" (pos or neg), "Count"
            # We'll separate pos/neg
            c_data_pos = c_data[c_data["Sentiment"] == "positive_count"]
            c_data_neg = c_data[c_data["Sentiment"] == "negative_count"]

            fig.add_trace(go.Bar(
                x=c_data_pos["timestamp_updated"],
                y=c_data_pos["Count"],
                name=f"{c_id} - Positive",
                marker=dict(color="green")
            ))
            fig.add_trace(go.Bar(
                x=c_data_neg["timestamp_updated"],
                y=c_data_neg["Count"],
                name=f"{c_id} - Negative",
                marker=dict(color="red")
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Sentiment Frequency",
            barmode='relative',  # diverging
            showlegend=True
        )
        return fig



def plot_sentiment_over_time_line(
    aggregated_df,
    cluster_col=None,
    color_map=None,
    title="Time-based Sentiment Analysis"
):
    """
    Builds a line chart (positive vs negative) over time.
    If cluster_col is None => single aggregated line, else multiple lines by cluster.

    Args:
        aggregated_df (pd.DataFrame): output of aggregate_sentiment_over_time()
        cluster_col (str or None): if None => aggregated (2 lines). Else => multi-cluster
        color_map (dict): cluster -> color
        title (str): chart title

    Returns:
        plotly.graph_objects.Figure or None
    """
    if aggregated_df.empty:
        return None

    # Check single vs multi cluster
    is_multi_cluster = (cluster_col is not None) and (cluster_col in aggregated_df.columns)

    if not is_multi_cluster:
        # Single aggregated => columns: [timestamp_updated, positive_count, negative_count, total_count]
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
        # Multi-cluster => columns: [cluster_col, timestamp_updated, positive_count, negative_count, total_count]
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
