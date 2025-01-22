import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define paths
s_root = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis'
s_db_table_json = 'Data/review_db_table.json'

model_name = 'sentence-transformers/all-MiniLM-L6-v2'

def initialize_miniLM():
    """
    Initializes and returns the embedding model.

    Returns:
    - A LangchainEmbedding instance with the specified model.
    """
    return LangchainEmbedding(HuggingFaceEmbeddings(model_name=model_name))

def index_embedding(text, embed_model):
    """
    Generates an embedding for a given text using the provided model.

    Parameters:
    - text: The input text to embed.
    - embed_model: The preloaded embedding model.

    Returns:
    - A NumPy array representing the text embedding.
    """
    text = text.encode(encoding='ASCII', errors='ignore').decode()
    return np.array(embed_model.get_text_embedding(text))

def get_top_keyword_result(keyword, embed_model):
    """
    Finds the top 1 result from the dataset most similar to the given keyword.

    Parameters:
    - keyword: The keyword to compare against embeddings.
    - embed_model: The preloaded embedding model.

    Returns:
    - A dictionary containing the row with the highest similarity to the keyword.
    """
    # Load the DataFrame with embeddings
    df = pd.read_json(os.path.join(s_root, s_db_table_json), orient='records')
    df = df[df['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    # Convert embeddings to matrix form
    mat = np.array(df['embedding'].tolist())

    # Generate embedding for the keyword
    keyword_embed = index_embedding(keyword, embed_model).reshape(1, -1)

    # Calculate cosine similarity
    similarities = cosine_similarity(mat, keyword_embed).flatten()

    # Find the index of the maximum similarity score
    top_index = np.argmax(similarities)

    # Extract the row with the highest similarity
    top_result = df.iloc[top_index].to_dict()
    top_result['similarity'] = similarities[top_index]  # Add similarity score for reference

    return top_result

# Example usage
if __name__ == "__main__":
    # Initialize the model once
    embed_model = initialize_miniLM()

    # Perform keyword search
    keyword = 'zombie'
    top_result = get_top_keyword_result(keyword, embed_model)
    print(f"Top result for keyword '{keyword}':")
    print(json.dumps(top_result, indent=4))
