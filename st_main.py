import streamlit as st
import os
from st_source.visuals import *
from st_source.filter_functions import *
import json

st.set_page_config(
    page_title="Loading the Data",
    page_icon="👋",
    layout="wide",
)

# Define path to precomputed JSON file
s_root = r'S:\SID\Analytics\Working Files\Individual\Florian\Projects\DataScience\cluster_analysis\Data\HRC\Cluster_tests'           # Root

s_db_table_preprocessed_json = os.path.join(s_root, 'db_final.json')          # Input data

st.write(
    """This demo illustrates a combination of plotting and animation with
Streamlit. We're generating a bunch of random numbers in a loop for around
5 seconds. Enjoy!"""
)

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