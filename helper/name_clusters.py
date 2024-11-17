from dotenv import load_dotenv
load_dotenv()

import os
import openai

openai_api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = openai_api_key
client = openai.Client()

chat_model_name = 'gpt-4o-mini'


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from langchain.prompts import PromptTemplate
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings

# Define paths and parameters
s_root = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis/'
s_db_table_preprocessed_json = 'Data/sentence_embeddings.json'

# Load precomputed data
df_total = pd.read_json(os.path.join(s_root, s_db_table_preprocessed_json), orient='records')
mat = np.array(df_total['embedding'].tolist())

# Clustering options and dimensionality reduction methods
clustering_algorithms = ["hdbscan", "kmeans"]
dimensionality_methods = ["UMAP", "PCA", "tSNE"]
kmeans_clusters = [5, 10, 15, 20, 35]

# Define prompt template for cluster naming
prompt_template_cluster_naming = '''Based on the following topics, generate a concise name (5 words or fewer) that best describes the general theme of this cluster.

TOPICS: {topics}
CLUSTER NAME: '''


def find_representative_topics(df, cluster_column, cluster_id, max_topics=8):
    """Finds up to max_topics representative topics based on centroid proximity."""
    cluster_data = df[df[cluster_column] == cluster_id]
    cluster_embeddings = np.array(cluster_data['embedding'].tolist())
    centroid = np.mean(cluster_embeddings, axis=0)
    distances = cosine_distances([centroid], cluster_embeddings).flatten()
    closest_indices = np.argsort(distances)[:max_topics]
    return cluster_data.iloc[closest_indices]['topic'].tolist()


def generate_cluster_name(topics_list):
    """Generates a concise name for the cluster using an API call."""
    topics = ", ".join(topics_list)
    prompt_cluster_naming = prompt_template_cluster_naming.format(topics=topics)

    # API call to OpenAI's chat completion model
    # API call to OpenAI's completion model
    cluster_name_response = client.chat.completions.create(
        model=chat_model_name,
        messages=[
            {"role": "system", "content": "You are an expert at summarizing topics into concise names."},
            {"role": "user", "content": prompt_cluster_naming},
        ],
        max_tokens=100  # Adjust tokens to limit response length
    )
    cluster_name = cluster_name_response.choices[0].message.content.strip()
    return cluster_name


# Iterate over each algorithm, dimensionality reduction method, and cluster size (for K-means)
cluster_names = {}
for method in dimensionality_methods:
    for view in ["2D", "3D"]:
        for algorithm in clustering_algorithms:
            if algorithm == "kmeans":
                for n_clusters in kmeans_clusters:
                    cluster_column = f"{algorithm}_{n_clusters}_{method}_{view}"
                    if cluster_column in df_total.columns:
                        unique_clusters = df_total[cluster_column].unique()
                        for cluster_id in unique_clusters:
                            topics = find_representative_topics(df_total, cluster_column, cluster_id)
                            cluster_name = generate_cluster_name(topics)
                            cluster_names[(cluster_column, cluster_id)] = cluster_name
                        # Map names to a new column in df_total for each K-means cluster set
                        df_total[f"{cluster_column}_name"] = df_total[cluster_column].map(
                            lambda x: cluster_names.get((cluster_column, x), "Unknown")
                        )
            else:
                cluster_column = f"{algorithm}_{method}_{view}"
                if cluster_column in df_total.columns:
                    unique_clusters = df_total[cluster_column].unique()
                    for cluster_id in unique_clusters:
                        topics = find_representative_topics(df_total, cluster_column, cluster_id)
                        cluster_name = generate_cluster_name(topics)
                        cluster_names[(cluster_column, cluster_id)] = cluster_name
                    # Map names to a new column in df_total for HDBSCAN clusters
                    df_total[f"{cluster_column}_name"] = df_total[cluster_column].map(
                        lambda x: cluster_names.get((cluster_column, x), "Unknown")
                    )

# Save updated DataFrame with cluster names to JSON
output_path = os.path.join(s_root, s_db_table_preprocessed_json)
df_total.to_json(output_path, orient='records')
print(f"Cluster names saved to {output_path}")
