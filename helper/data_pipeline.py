import datetime
import os
import json
import pandas as pd

from helper.utils import configure_api, logger
from helper.redshift_conector_standalone import fetch_query_results, logger, save_to_json
from helper.data_analysis import normalize_topics_key, process_entry, api_settings
from helper.data_analysis import read_json, translate_reviews
from helper.prompt_templates import prompt_template_topic, prompt_template_sentiment
from helper.embedding import get_embedding, get_offline_embedding, process_embedding, flatten_data


def gather_data(root_dir, data_source,
                query_function,
                query_function_args=None,
                id_column=None,
                text_column='review',
                timestamp_column=None,
                datetime_function=None,
                longname=None):
    path_db_prepared = os.path.join(root_dir, data_source, "db_prepared.json")
    results_json = query_function(*query_function_args) if query_function_args is not None else query_function()

    # Save the json
    parsed_json = json.loads(results_json)
    for review in parsed_json:
        review["pp_data_source"] = data_source
        review["pp_review_id"] = review.pop(id_column)
        review["pp_review"] = review.pop(text_column)
        review["pp_timestamp"] = review.pop(timestamp_column)
        if datetime_function is not None:
            review["pp_timestamp"] = str(datetime_function(review["pp_timestamp"]))
        if longname is not None:
            review["pp_longname"] = longname

    # 2) Then pretty-print with indentation
    save_to_json(parsed_json, path_db_prepared)


def translate_data(root_dir, data_source, language_column):
    path_db_prepared = os.path.join(root_dir, data_source, "db_prepared.json")
    path_db_translated = os.path.join(root_dir, data_source, "db_translated.json")

    # Get Language Tag
    data = read_json(path_db_prepared)
    df = pd.DataFrame(data)

    # Translate the data
    translate_reviews(df,
                      path_db_translated,                   # checks what & if data is already translated (backup)
                      id_column="pp_review_id",             # column with unique identifier
                      text_column="pp_review",
                      language_column=language_column)      # column with language tag


def analyse_data(root_dir, data_source):
    path_db_translated = os.path.join(root_dir, data_source, "db_translated.json")
    path_db_analysed = os.path.join(root_dir, data_source, "db_analysed.json")


    # TODO: currently we store the data sometimes as dataframe and sometimes as JSON. This should be unified at some point.
    # For now, we will simply transform the pandas dataframe into a JSON object.
    data = pd.read_pickle(path_db_translated)
    data_prepared = data.to_dict(orient='records')

    id_column = "pp_review_id"  # The column that contains unique identifiers
    columns_of_interest = "pp_review"  # The column that are going to be analyzed
    all_entries = []
    processed_ids = set()

    # If the analyzed file already exists, load it
    if os.path.exists(path_db_analysed):
        all_entries = read_json(path_db_analysed)
        processed_ids = {entry[id_column] for entry in all_entries}  # set for O(1) membership checks

    # Process all unprocessed entries
    for i, entry in enumerate(data_prepared):
        current_id = entry[id_column]

        # If we've already processed this entry, skip it
        if current_id in processed_ids:
            logger.info(f"Skipping entry {i} (ID: {current_id}) - already processed.")
            continue

        # Otherwise, process and append
        process_entry(
            entry,
            id_column,
            prompt_template_topic,
            prompt_template_sentiment,
            api_settings,
            columns_of_interest
        )
        all_entries.append(entry)
        processed_ids.add(current_id)  # mark as processed

        # Save intermediate progress every 10 entries
        if (i % 10) == 0 and i != 0:
            save_to_json(all_entries, path_db_analysed)
            logger.info(f"Progress saved at index {i}.")

    # Final save after the loop
    save_to_json(all_entries, path_db_analysed)
    logger.info(f"###### All entries have been analysed and final results saved. ###### {os.linesep}")


def embed_data(root_dir, data_source, embed_key):
    path_db_analysed = os.path.join(root_dir, data_source, "db_analysed.json")
    path_db_embedded = os.path.join(root_dir, data_source, "db_embedded.json")

    data = read_json(path_db_analysed)

    data_embedded = process_embedding(data, embed_key)
    data_flattened = flatten_data(data_embedded)
    #TODO: When Data is flattened, create an ID for each entry => pp_statement_id (pp_review_id + counter?)

    # Save the embedded data
    save_to_json(data_flattened, path_db_embedded)
