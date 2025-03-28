import logging
import json
import random
import numpy as np
import pandas as pd
import openai

# environment variables
from dotenv import dotenv_values
import os
d = dotenv_values()
for k in d.keys():
    os.environ[k] = d[k]

# Setup the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.ERROR)      # Supress API HTTP request logs

# Initialize global token counters
prompt_tokens = 0
completion_tokens = 0

def track_tokens(response):
    """
    Track Tokens when OpenAI API is used.
    """
    global prompt_tokens, completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens

# region API
api_settings = {"client": None, "model": None}

def configure_api(model_name):
    """
    Configures the global API client and model.
    Args:
        model_name (str): The model name to use.
    """
    global api_settings
    openai.api_key = os.getenv("OPENAI_API_KEY")
    api_settings["client"] = openai.Client()
    api_settings["model"] = model_name
# endregion

# region Readers and Writers
def save_to_json(data, output_path):
    """
    Saves data to a JSON file with proper encoding.
    Args:
        data (list): Data to save.
        output_path (str): Path to the output JSON file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data successfully saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        raise

def save_df_as_json(data, file_path):
    """
    Saves data to a JSON file.
    Args:
        data (list or pd.DataFrame): The data to save.
        file_path (str): The file path to save the JSON.
    """
    logger.info(f"Saving data to {file_path}")
    if isinstance(data, pd.DataFrame):
        data = data.to_dict(orient="records")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def save_data_for_streamlit(df, output_path):
    """
    Saves the updated DataFrame to a JSON file.
    Args:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): Path to save the JSON file.
    """
    logger.info(f"Saving updated data to {output_path}")
    df.to_json(output_path, orient="records", indent=4)
    logger.info("Data saved successfully.")

def load_json_into_df(json_path):
    """
    Loads the JSON data into a DataFrame.
    Args:
        json_path (str): Path to the JSON file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    logger.info(f"Loading data from {json_path}")
    return pd.read_json(json_path, orient="records")

def read_json(file_path):
    """
    Reads a JSON file and returns its contents as a Python object.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        object: The contents of the JSON file as a Python data structure (e.g., dict or list).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from file '{file_path}': {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred while reading the file '{file_path}': {e}")
        raise

# endregion

# region Data Processing
def get_random_sample(data, sample_size, seed=None):
    """
    Returns a random sample of the specified size from the dataset.
    """
    if seed is not None:
        random.seed(seed)  # Set the random seed for reproducibility

    if sample_size > len(data):
        raise ValueError(f"Sample size ({sample_size}) cannot exceed the dataset size ({len(data)}).")

    random_sample = random.sample(data, sample_size)
    logger.info(f"Generating a random sample of size {sample_size} with seed {seed}.")
    return random_sample

def filter_and_enrich_data(data, columns_of_interest):
    """
    Filters and enriches a dataset by specified columns.

    Args:
        data (list): List of dictionaries representing the dataset.
        columns_of_interest (list): List of column names to check in each entry.

    Returns:
        list: Filtered and enriched dataset.
    """
    removed_count = 0  # Track the number of removed entries
    filtered_data = []  # List to store the remaining entries

    for entry_idx, entry in enumerate(data):
        try:
            # Check if all specified columns are empty
            if all(
                    not str(entry.get(col, "")).strip() for col in columns_of_interest
            ):
                removed_count += 1  # Count this entry as removed
                logger.debug(f"Removed entry #{entry_idx}: All columns empty: {entry}")
            else:
                # Create a new key 'player_response' with concatenated text from specified columns
                player_response = " ".join(
                    str(entry.get(col, "")).strip()
                    for col in columns_of_interest
                    if str(entry.get(col, "")).strip()
                )

                # Check if player_response has more than 3 words
                if len(player_response.split()) > 3:
                    entry["player_response"] = player_response
                    filtered_data.append(entry)
                else:
                    removed_count += 1  # Count this entry as removed
                    logger.debug(f"Removed entry #{entry_idx}: player_response too short: {entry}")
        except Exception as e:
            logger.error(f"Error processing entry #{entry_idx}: {e}")
            raise

    # Log and print the number of removed entries
    logger.info(f"Total entries removed: {removed_count}")

    return filtered_data

def clean_json_data(data):
    """
    Cleans a list of dictionaries to ensure all entries are JSON-serializable
    and handles missing or invalid values appropriately.
    Args:
        data (list): List of dictionaries representing the dataset.
    Returns:
        list: Cleaned list of dictionaries.
    """
    def is_serializable(value):
        """
        Checks if a value can be serialized to JSON.
        """
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    def clean_value(value):
        """
        Cleans individual values in the dataset:
        - Replaces NaN or None with an empty string.
        - Leaves JSON-serializable values unchanged.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        if is_serializable(value):
            return value
        # For any non-serializable value, convert it to a string
        return str(value)

    def clean_entry(entry):
        """
        Cleans a single dictionary by applying `clean_value` to each field.
        """
        return {key: clean_value(value) for key, value in entry.items()}

    cleaned_data = [clean_entry(entry) for entry in data]
    original_count = len(data)
    cleaned_count = len(cleaned_data)
    logger.info(f"Cleaned {original_count - cleaned_count} entries from the dataset.")
    return cleaned_data

def generate_ID(data):
    """
     Adds a unique 'response_ID' to each tuple in the dataset if it doesn't already exist.

     Args:
         data (list): List of JSON-like dictionaries.

     Returns:
         list: Updated dataset with unique 'response_ID' for each tuple.
     """
    for idx, entry in enumerate(data, start=1):
        if "response_ID" not in entry:
            entry["response_ID"] = idx
    return data
# endregion