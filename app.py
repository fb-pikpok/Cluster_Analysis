import os
from st_source.visuals import *
from st_source.filter_functions import *
import json

# Set page layout
st.set_page_config(layout="wide")


# Upload JSON file
st.sidebar.header("Upload Your JSON File")
uploaded_file = st.sidebar.file_uploader("Choose a JSON file", type="json")


if uploaded_file is not None:
    @st.cache_data(show_spinner=False)
    def load_data(file):
        data = json.load(file)
        df = pd.DataFrame(data)

        # region Data Preprocessing
        try:
            df['prestige_rank'] = pd.to_numeric(df['prestige_rank'].replace("", 0), errors='coerce').fillna(0).astype(int)
        except:
            pass
        try:
            df['ever_been_subscriber'] = pd.to_numeric(df['ever_been_subscriber'].replace("", 0), errors='coerce').fillna(0).astype(int)
        except:
            pass
        try:
            df['is_current_subscriber'] = pd.to_numeric(df['is_current_subscriber'].replace("", 0), errors='coerce').fillna(0).astype(int)
        except:
            pass
        try:
            df['spending'] = pd.to_numeric(df['spending'].replace("", 0), errors='coerce').fillna(0).astype(int)
        except:
            pass
        try:
            df['playtime_at_review_minutes'] = df['author'].apply(lambda x: x.get('playtime_at_review_minutes', 0))
        except:
            pass
        try:
            df['weighted_vote_score'] = pd.to_numeric(df['weighted_vote_score'], errors='coerce').fillna(0)
        except:
            pass
        try:
            df['timestamp_updated'] = pd.to_datetime(df['timestamp_updated'], unit='s', errors="coerce")
            df['month'] = df['timestamp_updated'].dt.to_period('M').astype(str)
        except:
            pass
        # endregion

        return df

    df_total = load_data(uploaded_file)


    # region 1) Filters

    # Make a copy of the DataFrame to avoid modifying the original data
    filtered_df = df_total.copy()

    st.sidebar.header("Filter Options")

    #########################
    # 1. Expander: Display and Algorithm selection
    #########################
    with st.sidebar.expander("Display & Algorithm selection", expanded=False):
        display_mode = st.selectbox("Display Mode", ["Name", "ID"])

        dimensionality_options = ["UMAP", "PCA", "tSNE"]
        selected_dimensionality = st.selectbox("Dimensionality Reduction", dimensionality_options)

        clustering_options = ["hdbscan", "kmeans"]
        selected_clustering = st.selectbox("Clustering Algorithm", clustering_options)

        view_options = ["2D", "3D"]
        selected_view = st.radio("Select View", view_options)

        # If KMeans is selected, allow user to choose the cluster size
        # since we don't know what cluster sizes have been computed we generate a dynamic list of available sizes
        # by checking the columns of the dataframe
        if selected_clustering == "kmeans":
            kmeans_cols = [c for c in df_total.columns if c.startswith("kmeans_")]
            if kmeans_cols:
                available_sizes = sorted({int(c.split("_")[1]) for c in kmeans_cols if c.split("_")[1].isdigit()})
                selected_kmeans_size = st.selectbox("Cluster Size", options=available_sizes)
            else:
                selected_kmeans_size = st.number_input("Enter Number of KMeans Clusters", min_value=1, max_value=100, value=15, step=1)

    ######
    # Cluster ID columns
    ######
    # Determine the column where the cluster ID is stored based on user selection 'name' or 'ID'
        if selected_clustering == "kmeans":
            clustering_column = f"{selected_clustering}_{selected_kmeans_size}_id"
            clustering_name_column = f"{clustering_column}_name"
        else:
            clustering_column = "hdbscan_id"
            clustering_name_column = "hdbscan_id_name"

    #########################
    # Expander 2: Data Exploration
    #########################
    with st.sidebar.expander("Cluster Selection", expanded=False):
        if display_mode == "Name":
            all_clusters_list = sorted(df_total[clustering_name_column].dropna().unique())
        else:
            all_clusters_list = sorted(df_total[clustering_column].dropna().unique())

        cluster_options = ["All Clusters"] + list(all_clusters_list)
        selected_clusters = st.multiselect("Select Clusters (Multi-Select)", cluster_options, default=["All Clusters"])

        hide_noise = st.checkbox("Hide Noise", value=False)


    #########################
    # Expander 3: Time Filters
    #########################

    # with st.sidebar.expander("Time Frame", expanded=False):
    #     if not filtered_df.empty:
    #         min_date = filtered_df["timestamp_updated"].min().date()
    #         max_date = filtered_df["timestamp_updated"].max().date()
    #
    #         # By default, the date range is the entire dataset
    #         start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    #         end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    #
    #         # Only allow user inputs within the min and max date range
    #         filtered_df = filtered_df[
    #             (filtered_df["timestamp_updated"].dt.date >= start_date) &
    #             (filtered_df["timestamp_updated"].dt.date <= end_date)
    #         ]
    #
    #         if filtered_df.empty:
    #             st.warning("No data available for the selected date range.")
    #             st.stop()
    #
    #         granularity_options = {
    #             "Days": "D",
    #             "Weeks": "W",
    #             "Months": "ME"
    #         }
    #         granularity_label = st.selectbox("Select Time Granularity", list(granularity_options.keys()), index=2)
    #         selected_granularity = granularity_options[granularity_label]
    #     else:
    #         st.warning("No data left to apply Time Range filters.")
    #         st.stop()

    ##############################
    # 4) Expander: Optional Filters
    ##############################
    with st.sidebar.expander("Additional Filters", expanded=False):
        # Optional filters get dynamically created based on the current project.
        # These filters are defined in the filter_functions.py file.
        filtered_df = apply_optional_filters(filtered_df, container=st)

        # Adjust the colums that are beein displayed in the data table
        # Mandatory columns must be displayed
        mandatory_columns = ["topic", "sentence", "category", "sentiment"]

        # User can select via dropdown that is dynamically created based on the col names of the dataframe
        optional_columns = [
            c for c in df_total.columns
            if c not in mandatory_columns
               and c not in [clustering_column, clustering_name_column]
        ]

        selected_columns = st.multiselect(
            "Select Additional Columns to Display",
            options=optional_columns,
            default=[]
        )

    columns_to_display = mandatory_columns + selected_columns


    #########################
    # Apply Filters
    #########################
    if hide_noise:
        if display_mode == "ID":
            filtered_df = filtered_df[filtered_df[clustering_column] != -1]
        else:
            filtered_df = filtered_df[filtered_df[clustering_name_column] != "Noise"]

    # Filter by selected clusters
    if "All Clusters" not in selected_clusters:
        if display_mode == "ID":
            filtered_df = filtered_df[filtered_df[clustering_column].isin(selected_clusters)]
        else:
            filtered_df = filtered_df[filtered_df[clustering_name_column].isin(selected_clusters)]

    if filtered_df.empty:
        st.warning("No data available after applying all filters.")
        st.stop()



    # endregion


    # region 2) Cluster Visualization

    #####
    # Color Map
    #####
    # This color Map is used to ensure the same cluster has the same color across different visualizations / filters
    color_map = generate_color_map(df_total, clustering_column, clustering_name_column, display_mode)


    #####
    # Cluster Visualization
    #####
    st.subheader("Cluster Visualization")
    if not filtered_df.empty:
        x_col = f"{selected_clustering}_{selected_dimensionality}_2D_x"
        y_col = f"{selected_clustering}_{selected_dimensionality}_2D_y"
        z_col = f"{selected_clustering}_{selected_dimensionality}_3D_z" if selected_view == "3D" else None

        missing_cols = [col for col in [x_col, y_col, z_col] if col and col not in df_total.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
        else:
            fig = visualize_embeddings(
                filtered_df,
                x_col,
                y_col,
                z_col,
                'sentence',  # hover info
                clustering_name_column if display_mode == "Name" else clustering_column,
                color_map
            )
            st.plotly_chart(fig)
    else:
        st.warning("No data available for visualization.")

    # endregion


    # region 3) Data Table

    columns_to_display = mandatory_columns + selected_columns

    st.subheader("Data Table")
    if not filtered_df.empty:
        st.dataframe(filtered_df[columns_to_display])
    else:
        st.warning("No data available for the selected filters (table).")

    # endregion


    # region 4) Sentiment per cluster and cluster Size

    #########################
    # Sentiment per Cluster
    #########################
    st.subheader("Sentiment per Cluster")

    # Horizontal bar chart for Sentiment per cluster
    # +x = green (positive), -x = red (negative)
    if not filtered_df.empty:
        fig_sentiment = plot_sentiments(
            filtered_df,
            sentiment_col='sentiment',
            cluster_name_col=clustering_name_column if display_mode == "Name" else clustering_column
        )
        st.plotly_chart(fig_sentiment)
    else:
        st.warning("No sentiment data available for the selected filters.")


    #########################
    # Cluster Size (Bar Chart)
    #########################
    st.subheader("Requests per Cluster")

    # Determine the size of the cluster by number of facts, requests, or both
    # One vertical Bar represents the size of one cluster
    data_type = st.radio(
        "Select data to display:",
        options=["requests", "facts", "both"],
        index=0,
        horizontal=True
    )
    if not filtered_df.empty:
        if 'category' not in filtered_df.columns:
            st.warning("The 'category' column is not available in the dataset.")
        else:
            fig_request_count = plot_request_count_by_cluster(
                filtered_df,
                cluster_name_col=clustering_name_column if display_mode == "Name" else clustering_column,
                data_type=data_type
            )
            st.plotly_chart(fig_request_count)
    else:
        st.warning("No request count data available for the selected filters.")

    # endregion


    # region 5) Sentiment Over Time
    #########################
    # Sentiment Over Time
    #########################
    st.subheader("Sentiment Over Time")

    # Did the user select one or more specific clusters or all clusters?
    cluster_col = None
    if "All Clusters" not in selected_clusters:
        # Use whichever cluster column is relevant (ID or name)
        cluster_col = clustering_name_column if display_mode == "Name" else clustering_column

    # Did the user select a specific time range and granularity (e.g. days, weeks, months)?
    aggregated_data = aggregate_sentiment_over_time(df=filtered_df, time_col="timestamp_updated", sentiment_col="sentiment",
                                                    cluster_col=cluster_col, granularity=selected_granularity)


    ######
    # Aggregated Bar Chart
    ######
    fig_bar = plot_sentiment_over_time_bar(
        aggregated_df=aggregated_data,
        cluster_col=cluster_col,
        title="Aggregated Bar Chart"
    )
    if fig_bar:
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("No data available for time-based sentiment (Bar).")


    ######
    # Individual Line Chart
    ######
    fig_line = plot_sentiment_over_time_line(
        aggregated_df=aggregated_data,
        cluster_col=cluster_col,
        color_map=color_map,
        title="Individual Line Chart"
    )
    if fig_line:
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("No data available for time-based sentiment (Line).")

    # endregion

else:
    st.warning("Please upload a JSON file to begin.")

