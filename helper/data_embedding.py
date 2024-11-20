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
    processed_batch = []  # Collect processed results
    for review_entry in batch:
        if isinstance(review_entry, dict) and "topics" in review_entry and isinstance(review_entry["topics"], list):
            for d_topic in review_entry["topics"]:
                if isinstance(d_topic, dict):
                    # Check if the key exists and embedding should be created or overridden
                    if embed_key in d_topic and ("embedding" not in d_topic or b_override):
                        d_topic["embedding"] = embed_text(d_topic[embed_key], embed_model)
            processed_batch.append(review_entry)  # Append the processed entry
            # Release memory
            torch.cuda.empty_cache()
            gc.collect()
    return processed_batch




if __name__ == "__main__":
    # Example setup for testing the helper
    s_root = r"C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis"
    input_file = os.path.join(s_root, "Data", "db_analysed_empty.json")
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
    df_table = json_to_table(data)

    # Save the final JSON table
    save_to_json(df_table, output_file)
    logger.info("Embedding and conversion to table format completed.")