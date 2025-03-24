from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import pandas as pd

from helper.prompt_templates import prompt_template_cluster_naming
from helper.utils import track_tokens, prompt_tokens, completion_tokens, logger, api_settings, configure_api


def find_representative_topics(df, cluster_column, cluster_id, max_topics=8):
    """
    Finds representative topics for a cluster based on proximity to the centroid.

    Args:
        df (pd.DataFrame): DataFrame containing topic embeddings and cluster assignments.
        cluster_column (str): Name of the column representing cluster IDs.
        cluster_id (int): The cluster ID to process.
        max_topics (int): Maximum number of topics to return.

    Returns:
        list: List of representative topics for the cluster.
    """
    cluster_data = df[df[cluster_column] == cluster_id]
    if cluster_data.empty:
        logger.warning(f"No data found for cluster ID {cluster_id} in column {cluster_column}.")
        return []

    cluster_embeddings = np.array(cluster_data['embedding'].tolist())
    centroid = np.mean(cluster_embeddings, axis=0)
    distances = cosine_distances([centroid], cluster_embeddings).flatten()
    closest_indices = np.argsort(distances)[:max_topics]
    representative_topics = cluster_data.iloc[closest_indices]['sentence'].tolist()

    logger.info(f"Found {len(representative_topics)} Topics for {cluster_column} ID: {cluster_id}")
    return representative_topics


def generate_cluster_name(topics_list):
    """
    Generates a concise name for the cluster using OpenAI API.

    Args:
        topics_list (list): List of topics to summarize.
        api_settings (dict): API configuration with 'client' and 'model'.

    Returns:
        str: Generated cluster name.
    """
    if not topics_list:
        logger.warning("No topics found to generate a cluster name.")
        return "Unknown"

    topics = ", ".join(topics_list)
    prompt_cluster_naming = prompt_template_cluster_naming.format(topics=topics)

    try:
        response = api_settings["client"].chat.completions.create(
            model=api_settings["model"],
            messages=[
                {"role": "system", "content": "You are an expert at summarizing topics into concise names."},
                {"role": "user", "content": prompt_cluster_naming},
            ],
            max_tokens=100
        )

        # Extract cluster name
        if response.choices and response.choices[0].message.content:
            cluster_name = response.choices[0].message.content.strip()
            track_tokens(response)
            logger.info(f"Generated cluster name: {cluster_name}")
            logger.info(f'Tokens used so far: Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}')
            return cluster_name
        else:
            logger.error("Unexpected API response structure.")
            return "Error"
    except Exception as e:
        logger.error(f"Error in generate_cluster_name function: {e}")
        return "Error"



def name_clusters(df: pd.DataFrame,
                  cluster_col: str = "hdbscan_id",
                  embedding_col: str = "embedding",
                  text_col: str = "document",
                  top_k: int = 10,
                  skip_noise_label: int = -1,
                  openai_llm: str = "gpt-4o-mini"
            ) -> pd.DataFrame:
    df_out = df.copy()
    unique_ids = df_out[cluster_col].unique()
    cluster_id_to_name = {}

    configure_api(model_name=openai_llm)

    for c_id in unique_ids:
        # Optionally skip noise
        if skip_noise_label is not None and c_id == skip_noise_label:
            continue

        cluster_data = df_out[df_out[cluster_col] == c_id]
        if cluster_data.empty:
            continue

        # compute centroid
        embeddings = np.array(cluster_data[embedding_col].tolist())
        centroid = embeddings.mean(axis=0, keepdims=True)

        # find top_k closest
        dists = cosine_distances(centroid, embeddings).flatten()
        top_indices = np.argsort(dists)[:top_k]
        representative_texts = cluster_data.iloc[top_indices][text_col].tolist()

        # LLM or rule-based name generation

        cluster_name = generate_cluster_name(representative_texts)
        cluster_id_to_name[c_id] = cluster_name

    name_col = f"{cluster_col}_name"
    df_out[name_col] = df_out[cluster_col].apply(lambda cid: cluster_id_to_name.get(cid, "Noise"))
    return df_out
