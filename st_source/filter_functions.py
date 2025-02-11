import streamlit as st

# Define the optional filters
optional_filters = {
    "sentiment": {
        "description": "Select Sentiment",
        "logic": lambda df, selected_values: df["sentiment"].isin(selected_values),
        "type": "multiselect"
    },
    "prestige_rank": {
        "description": "Prestige rank",
        "logic": lambda df, min_val, max_val: (
                (df["prestige_rank"] >= min_val) & (df["prestige_rank"] <= max_val)
        ),
        "type": "slider"
    },
    "spending": {
        "description": "Only Show Spenders",
        "logic": lambda df: df["spending"] > 0,
        "type": "checkbox"
    },
    "is_current_subscriber": {
        "description": "Only Show Current Subscribers",
        "logic": lambda df: df["is_current_subscriber"] == 1,
        "type": "checkbox"
    },
    "ever_been_subscriber": {
        "description": "Only Show Previous Subscribers",
        "logic": lambda df: df["ever_been_subscriber"] == 1,
        "type": "checkbox"
    },
    "playtime_at_review_minutes": {
        "description": "Playtime at Review (Minutes)",
        "logic": lambda df, min_val, max_val: (
                (df["playtime_at_review_minutes"] >= min_val)
                & (df["playtime_at_review_minutes"] <= max_val if max_val < 12000 else True)
        ),
        "type": "slider"
    },
    "weighted_vote_score": {
        "description": "Most Helpful Reviews",
        "logic": lambda df: df["weighted_vote_score"] > 0.70,
        "type": "checkbox"
    }
}


def apply_optional_filters(dataframe, container=st.sidebar):
    """
    Dynamically applies optional filters to a DataFrame based on column presence and user input.
    By default, places widgets in st.sidebar, but you can pass any container (like an expander).

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        container (Streamlit container): Where to place the filter widgets (sidebar, expander, etc.).

    Returns:
        pd.DataFrame: The filtered DataFrame after applying the optional filters.
    """
    selected_filters = {}
    df_filtered = dataframe.copy()  # Work on a copy

    # Dynamically add filters to the specified container
    for col, filter_details in optional_filters.items():
        if col in df_filtered.columns:
            filter_type = filter_details.get("type", "checkbox")
            description = filter_details["description"]

            if filter_type == "slider":
                if col == "playtime_at_review_minutes":
                    # Hard-coded slider range for playtime
                    slider_min, slider_max = 0, 12000
                    selected_filters[col] = container.slider(
                        description,
                        min_value=slider_min,
                        max_value=slider_max,
                        value=(slider_min, slider_max)
                    )
                else:
                    # Generic slider for numeric columns
                    min_val = int(df_filtered[col].min())
                    max_val = int(df_filtered[col].max())
                    selected_filters[col] = container.slider(
                        description,
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )

            elif filter_type == "checkbox":
                selected_filters[col] = container.checkbox(description, value=False)

            elif filter_type == "multiselect":
                unique_values = df_filtered[col].unique()
                selected_filters[col] = container.multiselect(
                    description,
                    options=unique_values,
                    default=unique_values
                )

    # Apply filters to the DataFrame
    for col, user_input in selected_filters.items():
        if col in df_filtered.columns:
            ftype = optional_filters[col].get("type")
            logic_func = optional_filters[col]["logic"]

            if ftype == "slider":
                min_val, max_val = user_input
                df_filtered = df_filtered[logic_func(df_filtered, min_val, max_val)]

            elif ftype == "multiselect":
                df_filtered = df_filtered[logic_func(df_filtered, user_input)]

            elif ftype == "checkbox" and user_input:
                df_filtered = df_filtered[logic_func(df_filtered)]

    return df_filtered
