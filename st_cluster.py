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
        df['timestamp_updated'] = pd.to_datetime(df['timestamp_updated'], unit='ms')  # Convert from milliseconds
        df['month'] = df['timestamp_updated'].dt.to_period('M').astype(str)  # Convert to string for JSON serialization
    except:
        pass

    # endregion
    return df

df_total = load_data(s_db_table_preprocessed_json)


# Display the data
# Title
st.title('HRC Cluster Analysis')

# Display the data in a table (optional, or limit the rows)
st.subheader("Data Preview")
st.dataframe(df_total.head(10))

# --- 3D Scatter Plot ---
st.subheader("3D Coordinates with Cluster Name")

# 1. Extract x, y, z from 'embedding' if present
if "embedding" in df_total.columns:
    # Each 'embedding' is presumably a list of 3 floats
    df_total[['x', 'y', 'z']] = pd.DataFrame(df_total['embedding'].tolist(), index=df_total.index)
else:
    st.warning("No 'embedding' column found in the data. Cannot plot 3D.")
    st.stop()

# 2. Create the 3D scatter plot
# Make sure there's a 'cluster_name' column
if "cluster_name" not in df_total.columns:
    st.warning("No 'cluster_name' column found in the data. Using default color.")
    fig = px.scatter_3d(
        df_total,
        x='x', y='y', z='z',
        hover_data=['x', 'y', 'z']
    )
else:
    fig = px.scatter_3d(
        df_total,
        x='x', y='y', z='z',
        color='cluster_name',
        hover_data=['cluster_name']
    )

# 3. Display it in Streamlit
st.plotly_chart(fig, use_container_width=True)
