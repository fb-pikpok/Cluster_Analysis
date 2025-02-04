import streamlit as st
import os
from st_source.visuals import *
from st_source.filter_functions import *
import json


# Set page layout
st.set_page_config(layout="wide")

# Define path to precomputed JSON file
s_root = r'S:\SID\Analytics\Working Files\Individual\Florian\Projects\DataScience\cluster_analysis\Data\HRC\Cluster_tests'           # Root

s_db_table_preprocessed_json = os.path.join(s_root, 'db_final.json')          # Input data

# Load precomputed data
@st.cache_data(show_spinner=False)
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # region Data Preprocessing
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
        df['playtime_at_review_minutes'] = df['author'].apply(lambda x: x.get('playtime_at_review_minutes', 0))
    except:
        pass

    # Handle 'weighted_vote_score'
    try:
        df['weighted_vote_score'] = pd.to_numeric(df['weighted_vote_score'], errors='coerce').fillna(0)
    except:
        pass

    # Convert 'timestamp_created' to a datetime object and extract the month
    try:
        df['timestamp_updated'] = pd.to_datetime(df['timestamp_updated'], unit='s')  # Convert from milliseconds
        df['month'] = df['timestamp_updated'].dt.to_period('M').astype(str)  # Convert to string for JSON serialization
    except:
        pass

    # endregion
    return df

df_total = load_data(s_db_table_preprocessed_json)

# ----------------------------------------------------------------------
# 2) Generate a color map from the FULL dataset
# ----------------------------------------------------------------------
# This ensures that each cluster gets a consistent color even if it’s filtered out later.
clustering_name_column = "cluster_name"
clustering_column = "cluster_id"
display_mode = "Name"

# This is your custom function or logic that assigns colors.
# Make sure it uses df_total (the full data), not df_filtered.
full_color_map = generate_color_map(
    df_total,
    clustering_column=clustering_column,
    clustering_name_column=clustering_name_column,
    display_mode=display_mode
)

# ----------------------------------------------------------------------
# 3) Sidebar Controls
# ----------------------------------------------------------------------
st.sidebar.header("Filter Options")

# (a) Dimensionality for cluster scatter
dimensionality_options = ["UMAP", "PCA", "tSNE"]
selected_dimensionality = st.sidebar.selectbox("Dimensionality Reduction", dimensionality_options)

# (b) Hide Noise
hide_noise = st.sidebar.checkbox("Hide Noise", value=False)

# (c) 2D or 3D
view_options = ["2D", "3D"]
selected_view = st.sidebar.radio("Select View", view_options)

# (d) Time-based filters
st.sidebar.subheader("Time-based Sentiment Analysis")
min_date = df_total["timestamp_updated"].min().date() if not df_total.empty else None
max_date = df_total["timestamp_updated"].max().date() if not df_total.empty else None

start_date = st.sidebar.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

granularity_options = {
    "Months": "M",
    "Weeks": "W",
    "Days": "D",
    "Hours": "H"
}
granularity_label = st.sidebar.selectbox("Granularity", list(granularity_options.keys()))
selected_granularity = granularity_options[granularity_label]

# (e) Multi-select clusters, including "All Clusters"
all_clusters_list = sorted(df_total[clustering_name_column].unique())
cluster_choices = ["All Clusters"] + all_clusters_list
selected_clusters = st.sidebar.multiselect(
    "Select Clusters (multi-select)",
    cluster_choices,
    default=["All Clusters"]
)

# ----------------------------------------------------------------------
# 4) Apply Filters => df_filtered
# ----------------------------------------------------------------------
df_filtered = df_total.copy()

# (i) Date Range
df_filtered = df_filtered[
    (df_filtered["timestamp_updated"].dt.date >= start_date) &
    (df_filtered["timestamp_updated"].dt.date <= end_date)
    ]

# (ii) Cluster Filter
if "All Clusters" not in selected_clusters:
    df_filtered = df_filtered[df_filtered[clustering_name_column].isin(selected_clusters)]

# (iii) Hide Noise
if hide_noise:
    df_filtered = df_filtered[df_filtered[clustering_name_column] != "Unknown"]

if df_filtered.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# ----------------------------------------------------------------------
# 5) Cluster Visualization (using df_filtered) but color map from df_total
# ----------------------------------------------------------------------
st.subheader("Cluster Visualization")

x_col = f"{selected_dimensionality}_2D_x"
y_col = f"{selected_dimensionality}_2D_y"
z_col = f"{selected_dimensionality}_3D_z" if selected_view == "3D" else None

missing_cols = [col for col in [x_col, y_col, z_col] if col and col not in df_filtered.columns]
if missing_cols:
    st.error(f"Missing columns for scatter plot: {missing_cols}")
else:
    fig = visualize_embeddings(
        df_filtered,
        x_col,
        y_col,
        z_col,
        'sentence',  # or whichever hover column
        clustering_name_column,
        full_color_map  # <-- use the full color map
    )
    st.plotly_chart(fig)

# ----------------------------------------------------------------------
# 6) Time-based Sentiment Analysis (using df_filtered)
# ----------------------------------------------------------------------
st.subheader("Time-based Sentiment Analysis")

df_time = df_filtered.set_index("timestamp_updated").sort_index()

if "All Clusters" in selected_clusters:
    # -- Aggregated approach
    agg_df = df_time.resample(selected_granularity).agg({
        "voted_up": ["sum", "count"]
    })
    agg_df.columns = ["positive_count", "total_count"]
    agg_df["negative_count"] = agg_df["total_count"] - agg_df["positive_count"]
    agg_df["negative_count"] = -agg_df["negative_count"]
    agg_df = agg_df.reset_index()

    plot_df = agg_df.melt(
        id_vars=["timestamp_updated"],
        value_vars=["positive_count", "negative_count"],
        var_name="Sentiment",
        value_name="Count"
    )

    fig_sentiment = px.line(
        plot_df,
        x="timestamp_updated",
        y="Count",
        color="Sentiment",  # you'll just have 2 lines: positive, negative
        title="Sentiment Over Time (All Clusters)",
        labels={
            "timestamp_updated": "Time",
            "Count": "Count (+ = Positive, - = Negative)"
        }
    )
    fig_sentiment.update_layout(yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'))
    st.plotly_chart(fig_sentiment, use_container_width=True)
else:
    # -- Multi-cluster approach
    grouped = (
        df_time
        .groupby(clustering_name_column)
        .resample(selected_granularity)
        .agg({"voted_up": ["sum", "count"]})
    )
    grouped.columns = ["positive_count", "total_count"]
    grouped.reset_index(inplace=True)

    grouped["negative_count"] = grouped["total_count"] - grouped["positive_count"]
    grouped["negative_count"] = -grouped["negative_count"]

    plot_df = grouped.melt(
        id_vars=[clustering_name_column, "timestamp_updated"],
        value_vars=["positive_count", "negative_count"],
        var_name="Sentiment",
        value_name="Count"
    )

    fig_sentiment = px.line(
        plot_df,
        x="timestamp_updated",
        y="Count",
        color=clustering_name_column,
        line_dash="Sentiment",
        # IMPORTANT: use the same color map created from df_total
        color_discrete_map=full_color_map,
        title="Sentiment Over Time (Selected Clusters)",
        labels={
            "timestamp_updated": "Time",
            "Count": "Count (+ = Positive, - = Negative)",
            clustering_name_column: "Cluster"
        }
    )
    fig_sentiment.update_layout(yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'))
    st.plotly_chart(fig_sentiment, use_container_width=True)


