import streamlit as st


optional_filters = {
    "sentiment": {
        "description": "Select Sentiment",
        "logic": lambda df, selected_values: df["sentiment"].isin(selected_values),
        "type": "multiselect"
    },
    "prestige_rank": {
        "description": "Prestige rank",
        "logic": lambda df, min_val, max_val: (df["prestige_rank"] >= min_val) & (
                    df["prestige_rank"] <= max_val),
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
            (df["playtime_at_review_minutes"] >= min_val) &
            (df["playtime_at_review_minutes"] <= max_val if max_val < 12000 else True)
        ),
        "type": "slider"
    },
    "weighted_vote_score": {
        "description": "Most Helpful Reviews",
        "logic": lambda df: df["weighted_vote_score"] > 0.70,
        "type": "checkbox"
    }


}


def apply_optional_filters(dataframe):
    """
    Dynamically applies optional filters to a DataFrame based on column presence and user input.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        optional_filters (dict): A dictionary where keys are column names and values are dictionaries
                                 with "description" (str), "logic" (function), and optional "type" (e.g., slider, checkbox, multiselect).

    Returns:
        pd.DataFrame: The filtered DataFrame after applying the optional filters.
    """
    selected_filters = {}

    # Dynamically add filters to the sidebar based on column presence
    for col, filter_details in optional_filters.items():
        if col in dataframe.columns:
            filter_type = filter_details.get("type", "checkbox")  # Default to checkbox if type is not specified
            if filter_type == "slider":
                # Handle playtime_at_review_minutes with a fixed slider range
                if col == "playtime_at_review_minutes":
                    slider_min = 0
                    slider_max = 12000  # Cap the slider at 12,000 (200 hours)
                    selected_filters[col] = st.sidebar.slider(
                        filter_details["description"],
                        min_value=slider_min,
                        max_value=slider_max,
                        value=(slider_min, slider_max)
                    )
                else:
                    # Default slider logic
                    min_val = int(dataframe[col].min())
                    max_val = int(dataframe[col].max())
                    selected_filters[col] = st.sidebar.slider(
                        filter_details["description"],
                        min_value=min_val,
                        max_value=max_val,
                        value=(min_val, max_val)
                    )
            elif filter_type == "checkbox":
                # Add a checkbox-based filter
                selected_filters[col] = st.sidebar.checkbox(filter_details["description"], value=False)
            elif filter_type == "multiselect":
                # Add a multiselect-based filter
                unique_values = dataframe[col].unique()
                selected_filters[col] = st.sidebar.multiselect(
                    filter_details["description"],
                    options=unique_values,
                    default=unique_values
                )

    # Apply filters to the DataFrame
    for col, user_input in selected_filters.items():
        if col in dataframe.columns:
            if optional_filters[col].get("type") == "slider":
                # Extract slider range
                min_val, max_val = user_input
                dataframe = dataframe[optional_filters[col]["logic"](dataframe, min_val, max_val)]
            elif optional_filters[col].get("type") == "multiselect":
                # Apply multiselect logic
                dataframe = dataframe[optional_filters[col]["logic"](dataframe, user_input)]
            elif optional_filters[col].get("type") == "checkbox" and user_input:
                # Apply checkbox logic
                dataframe = dataframe[optional_filters[col]["logic"](dataframe)]

    return dataframe






