import json
import gc
import os
import torch
import pandas as pd
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
import logging

from helper.data_preparation import load_json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def initialize_embedding_model(model_name):
    """
    Initializes the embedding model using HuggingFace embeddings.
    Args:
        model_name (str): The name of the embedding model to load.
    Returns:
        LangchainEmbedding: The initialized embedding model.
    """
    logger.info(f"Loading embedding model: {model_name}")
    embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))
    return embed_model

def embed_text(text, embed_model):
    """
    Generates embeddings for the provided text.
    Args:
        text (str): The text to embed.
        embed_model (LangchainEmbedding): The embedding model.
    Returns:
        list: The embedding vector.
    """
    text = text.encode(encoding="ASCII", errors="ignore").decode()
    return embed_model.get_text_embedding(text)

def process_batch(batch, embed_model, b_override, embed_key="topic"):
    """
    Processes a batch of review entries, embedding specified fields.
    Args:
        batch (list): A list of review entries (dictionaries).
        embed_model (LangchainEmbedding): The embedding model.
        b_override (bool): Whether to override existing embeddings.
        embed_key (str): The key in each entry to embed (e.g., "topic" or "sentence").
    Returns:
        list: The batch with updated embeddings.
    """
    for review_entry in batch:
        if isinstance(review_entry, dict) and "topics" in review_entry and isinstance(review_entry["topics"], list):
            for d_topic in review_entry["topics"]:
                if isinstance(d_topic, dict):
                    # Check if the key exists and embedding should be created or overridden
                    if embed_key in d_topic and ("embedding" not in d_topic or b_override):
                        d_topic["embedding"] = embed_text(d_topic[embed_key], embed_model)
                        # Release memory
                        torch.cuda.empty_cache()
                        gc.collect()
    return batch

def convert_to_table(json_data):
    """
    Converts the embedded JSON data into a flat table format.
    Args:
        json_data (list): List of review entries containing topics and embeddings.
    Returns:
        pd.DataFrame: A flattened table of all topics with additional fields.
    """
    logger.info("Converting JSON data to a table format.")
    df_total = pd.DataFrame()

    for review_entry in json_data:
        if "topics" in review_entry and isinstance(review_entry["topics"], list):
            df_gp = pd.DataFrame(review_entry["topics"])
            for key, value in review_entry.items():
                if key != "topics":
                    df_gp[key] = value
            df_total = pd.concat([df_total, df_gp], ignore_index=True)

    logger.info("Conversion to table format completed.")
    return df_total

def save_to_json(data, file_path):
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



if __name__ == "__main__":
    # Example setup for testing the helper
    s_root = r"C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis"
    input_file = os.path.join(s_root, "Data", "db_analysed.json")
    output_file = os.path.join(s_root, "Data", "db_embedded_table.json")
    embed_model_name = "all-MiniLM-L6-v2"  # Change this to your preferred model
    batch_size = 10
    b_override = False  # Change to True if embeddings should be overwritten
    embed_key = "topic"  # Change to "sentence" if you want to embed sentences

    # Load the JSON data
    data = load_json(input_file)

    # Initialize the embedding model once
    embed_model = initialize_embedding_model(model_name=embed_model_name)

    # Process data in batches
    for batch_start in range(0, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch = data[batch_start:batch_end]
        logger.info(f"Processing batch {batch_start // batch_size + 1} ({batch_start} to {batch_end})")
        batch = process_batch(batch, embed_model, b_override, embed_key=embed_key)
        data[batch_start:batch_end] = batch

    # Convert the data to table format
    df_table = convert_to_table(data)

    # Save the final JSON table
    save_to_json(df_table, output_file)
    logger.info("Embedding and conversion to table format completed.")