import os
import json
import random
import re
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_excel_to_data(excel_path):
    """
    Loads an Excel file and converts it into a list of dictionaries.
    Args:
        excel_path (str): Path to the input Excel file.
    Returns:
        list: List of dictionaries representing the data.
    """
    try:
        logger.info("Loading Excel file: %s", excel_path)
        dataframe = pd.read_excel(excel_path)
        data_as_dict = dataframe.to_dict(orient='records')
        logger.info("Excel data successfully loaded and converted to dictionary.")
        return data_as_dict
    except Exception as e:
        logger.error("Error loading Excel: %s", e)
        raise

def clean_data(data):
    """
    Cleans data by removing entries with non-parsable strings.
    Args:
        data (list): List of dictionaries representing the data.
    Returns:
        list: Cleaned data with invalid entries removed.
    """
    def is_valid_string(s):
        return bool(re.match(r'^[\u0000-\u007F]*$', s))

    def is_clean_entry(entry):
        return all(isinstance(value, str) and is_valid_string(value) if isinstance(value, str) else True
                   for key, value in entry.items())

    try:
        logger.info("Cleaning data.")
        initial_count = len(data)
        clean_data = [entry for entry in data if is_clean_entry(entry)]
        removed_count = initial_count - len(clean_data)
        logger.info("Data cleaning completed. Entries removed: %d", removed_count)
        return clean_data
    except Exception as e:
        logger.error("Error cleaning data: %s", e)
        raise

def sample_data(data, sample_size, seed=None):
    """
    Creates a sampled subset of the data and assigns unique IDs to each entry.
    Args:
        data (list): List of dictionaries representing the data.
        sample_size (int): Number of entries to sample.
        seed (int, optional): Seed value for reproducible sampling. Defaults to None.
    Returns:
        list: Sampled data with unique IDs assigned.
    """
    try:
        if seed is not None:
            logger.info("Setting random seed to: %d", seed)
            random.seed(seed)
        logger.info("Sampling data. Sample size: %d", sample_size)
        sample_size = min(sample_size, len(data))
        sampled_data = random.sample(data, sample_size)

        # Assign unique IDs to each entry
        for idx, entry in enumerate(sampled_data, start=1):
            entry["ID"] = idx

        logger.info("Data sampling completed with unique IDs assigned.")
        return sampled_data
    except Exception as e:
        logger.error("Error sampling data: %s", e)
        raise

def save_data_to_json(data, output_json_path):
    """
    Saves data to a JSON file.
    Args:
        data (list): List of dictionaries representing the data.
        output_json_path (str): Path to save the JSON file.
    """
    try:
        logger.info("Saving data to JSON: %s", output_json_path)
        with open(output_json_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
        logger.info("Data successfully saved to JSON.")
    except Exception as e:
        logger.error("Error saving JSON: %s", e)
        raise

def prepare_file_path(root_path, relative_path):
    """
    Dynamically constructs file paths based on a root directory and a relative path.
    Args:
        root_path (str): Root directory path.
        relative_path (str): Relative path from the root directory.
    Returns:
        str: Full path.
    """
    return os.path.join(root_path, relative_path)

def load_json(file_path):
    """
    Loads data from a JSON file.
    Args:
        file_path (str): The file path to load.
    Returns:
        list: The loaded JSON data.
    """
    logger.info(f"Loading data from {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Example usage in main script:
if __name__ == "__main__":
    root_dir = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis/'
    print(root_dir)
    # File paths
    excel_path = prepare_file_path(root_dir, "Data/2024 Trimester 1.xlsx")
    final_json_path = prepare_file_path(root_dir, "Data/db_prepared_HRC.json")

    # Process steps
    data = load_excel_to_data(excel_path)
    cleaned_data = clean_data(data)
    sampled_data = sample_data(cleaned_data, 25, seed=42)  # Set seed for reproducibility
    save_data_to_json(sampled_data, final_json_path)
