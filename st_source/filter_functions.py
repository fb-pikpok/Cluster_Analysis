import streamlit as st


# Define optional filters
optional_filters = {
    "spending": {
        "description": "Only Show Spenders",
        "logic": lambda df: df["spending"] > 0
    },
    "is_current_subscriber": {
        "description": "Only Show Current Subscribers",
        "logic": lambda df: df["is_current_subscriber"] == 1
    },
    "ever_been_subscriber": {
        "description": "Only Show Previous Subscribers",
        "logic": lambda df: df["ever_been_subscriber"] == 1
    }
}

def apply_optional_filters(dataframe):
    """
    Dynamically applies optional filters to a DataFrame based on column presence and user input.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        optional_filters (dict): A dictionary where keys are column names and values are dictionaries
                                 with "description" (str) and "logic" (function) for each filter.

    Returns:
        pd.DataFrame: The filtered DataFrame after applying the optional filters.
    """
    selected_filters = {}

    # Dynamically add filters to the sidebar based on column presence
    for col, filter_details in optional_filters.items():
        if col in dataframe.columns:
            selected_filters[col] = st.sidebar.checkbox(filter_details["description"], value=False)

    # Apply filters to the DataFrame
    for col, is_selected in selected_filters.items():
        if is_selected:
            dataframe = dataframe[optional_filters[col]["logic"](dataframe)]

    return dataframe




