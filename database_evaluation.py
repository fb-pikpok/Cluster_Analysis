"""
In this Python file all the data evaluation methods for the steam reviews, that are currently handled in the
Jupyter file are condensed into one script.
The script performs the following steps:

This first step is already performed during the scraping process in helper.steam_reviews.py:
0. Filter reviews with
    - high numbers per character ratio.
    - Reviews with less than 3 words (some chinese reviews get falsely detected as being to short and are removed)
    - Consisting of special characters (language symbols are allowed)


Status Quo in this script
1. Load the data
2. Translate reviews to English
    - language is detected via 'Lingua' ( can be replaced as soon as we store the player language from the Steam API)
3. Analyse reviews by extracting topic and sentiment via OpenAI API (gpt-4o-mini)


This part will be added later as soon as we know how and if we store the embeddings:
4. embedd the sentences extracted vie OpenAI API (ada-3)
5. reduce Dimensions
6. Cluster the reviews
7. Name the clusters
"""

# General modules
import os
import openai
from docs.conf import language
from dotenv import load_dotenv
import logging


# Setup
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)      # Supress API HTTP request logs


# My imports
from helper.redshift_conector_standalone import *
from helper.data_analysis import *

# OpenAI
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
client = openai.Client()
chat_model_name = 'gpt-4o-mini'

configure_api(client, chat_model_name)

# Parameters
root_dir = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis\Data\Database'
steam_title = 'Nightingale'

file_translated = os.path.join(root_dir, f'{steam_title}_1_translated.pkl')
file_analysed = os.path.join(root_dir, f'{steam_title}_2_analysed.pkl')


# region SQL Query Redshift
sql_query = """
SELECT * 
FROM steam_review 
where app_id_name = '1928980_Nightingale' LIMIT 13
"""
logger.info(f"Query Redshift with: {sql_query}")

try:
    results_json, results_df = fetch_query_results(sql_query)
    # Print the first row of the DataFrame
    logger.info("Successfully fetched query results, with shape: %s", results_df.shape)
except Exception as e:
    logger.error(f"Error fetching query results: {e}")
    raise

#endregion


# region Translate Reviews

# try:
#     results_df = detect_language_in_dataframe(results_df, text_column='review_text', language_column='language')
#     print(results_df.head())
# except Exception as e:
#     logger.error(f"Error running language detection: {e}")


# Process reviews
updated_df = translate_reviews(
    df=results_df,
    file_path=file_translated,
    id_column='recommendationid',
    text_column='review_text',
    language_column='language'
)

updated_df.to_pickle(file_translated)

# endregion



# region Analyse Reviews
# read the translated reviews
df_translated = pd.read_pickle(file_translated)








#
# path_db_embedded = os.path.join(root_dir, steam_title, "db_embedded.json")
#
# path_db_clustered = os.path.join(root_dir, steam_title, "db_clustered.json")
# path_db_named = os.path.join(root_dir, steam_title, "db_named.json")
#
#
# # my imports
# from helper.utils import *
# from helper.cluster_analysis import *
#
# configure_api(client, chat_model_name)