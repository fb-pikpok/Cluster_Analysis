import os
import json
from sklearn.metrics.pairwise import cosine_distances

from helper.prompt_templates import prompt_template_cluster_naming
from helper.utils import *


import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


from dotenv import load_dotenv
import openai
# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client()


# Initialize global token counters
prompt_tokens = 0
completion_tokens = 0


def track_tokens(response):
   """
   Updates the global token counters based on the API response.


   Args:
       response: The API response containing token usage.
   """
   global prompt_tokens, completion_tokens
   prompt_tokens += response.usage.prompt_tokens
   completion_tokens += response.usage.completion_tokens


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


def generate_cluster_name(topics_list, api_settings):
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
           logger.info(f' Tokens used so far: Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}')
           return cluster_name
       else:
           logger.error("Unexpected API response structure.")
           return "Error"
   except Exception as e:
       logger.error(f"Error in generate_cluster_name function: {e}")
       return "Error"


def process_clusters(df, dimensionality_methods, clustering_algorithms, max_centers, api_settings, kmeans_clusters=None):
    """
    Processes each cluster to generate names and updates the DataFrame with cluster names.

    Args:
        df (pd.DataFrame): DataFrame containing clusters and topics.
        dimensionality_methods (list): List of dimensionality reduction methods.
        clustering_algorithms (list): List of clustering algorithms.
        max_centers (int): Maximum number of representative topics for each cluster.
        api_settings (dict): API configuration with 'client' and 'model'.
        kmeans_clusters (list, optional): List of KMeans cluster counts. Required if KMeans is in clustering_algorithms.

    Returns:
        pd.DataFrame: Updated DataFrame with cluster names.
    """
    cluster_names = {}

    for method in dimensionality_methods:
        cluster_topics = {}  # Store topics for 2D clusters to reuse in 3D

        for view in ["2D", "3D"]:
            dim_suffix = "2D" if view == "2D" else "3D"

            for algorithm in clustering_algorithms:
                if algorithm == "kmeans":
                    if not kmeans_clusters:
                        raise ValueError("kmeans_clusters must be provided if 'kmeans' is in clustering_algorithms.")
                    # Process each KMeans cluster count
                    for n_clusters in kmeans_clusters:
                        cluster_column = f"{algorithm}_{n_clusters}_{method}_{dim_suffix}"
                        if cluster_column not in df.columns:
                            logger.warning(f"Column {cluster_column} not found in the DataFrame. Skipping.")
                            continue

                        unique_clusters = df[cluster_column].unique()
                        for cluster_id in unique_clusters:
                            if cluster_id == -1:  # Skip noise clusters
                                continue

                            if view == "2D":
                                # Find representative topics for the 2D cluster
                                topics = find_representative_topics(df, cluster_column, cluster_id, max_centers)

                                # Generate the cluster name using the topics
                                cluster_name = generate_cluster_name(topics, api_settings)
                                cluster_names[(cluster_column, cluster_id)] = cluster_name

                                # Store topics for reuse in 3D naming
                                cluster_topics[cluster_id] = topics

                            else:
                                # For 3D, reuse the name and topics from the 2D representation
                                topics = cluster_topics.get(cluster_id, [])
                                cluster_name = cluster_names.get(
                                    (f"{algorithm}_{n_clusters}_{method}_2D", cluster_id), "Unknown"
                                )
                                logger.info(
                                    f"KMeans Cluster ID {cluster_id} ({method}, k={n_clusters}, 3D): Reused 2D cluster name: {cluster_name}"
                                )

                            # Update the cluster names in the DataFrame
                            df[f"{algorithm}_{n_clusters}_{method}_2D_name"] = df[f"{algorithm}_{n_clusters}_{method}_2D"].map(
                                lambda x: cluster_names.get((f"{algorithm}_{n_clusters}_{method}_2D", x), "Unknown")
                            )
                            df[f"{algorithm}_{n_clusters}_{method}_3D_name"] = df[f"{algorithm}_{n_clusters}_{method}_3D"].map(
                                lambda x: cluster_names.get((f"{algorithm}_{n_clusters}_{method}_2D", x), "Unknown")
                            )
                elif algorithm == "hdbscan":
                    # Handle HDBSCAN clustering
                    cluster_column = f"{algorithm}_{method}_{dim_suffix}"
                    if cluster_column not in df.columns:
                        logger.warning(f"Column {cluster_column} not found in the DataFrame. Skipping.")
                        continue

                    unique_clusters = df[cluster_column].unique()
                    for cluster_id in unique_clusters:
                        if cluster_id == -1:  # Skip noise clusters
                            continue

                        if view == "2D":
                            # Find representative topics for the 2D cluster
                            topics = find_representative_topics(df, cluster_column, cluster_id, max_centers)

                            # Generate the cluster name using the topics
                            cluster_name = generate_cluster_name(topics, api_settings)
                            cluster_names[(cluster_column, cluster_id)] = cluster_name

                            # Store topics for reuse in 3D naming
                            cluster_topics[cluster_id] = topics

                            logger.info(f"HDBSCAN Cluster ID {cluster_id} ({method} 2D): {cluster_name}")
                        else:
                            # For 3D, reuse the name and topics from the 2D representation
                            topics = cluster_topics.get(cluster_id, [])
                            cluster_name = cluster_names.get(
                                (f"{algorithm}_{method}_2D", cluster_id), "Unknown"
                            )
                            logger.info(
                                f"HDBSCAN Cluster ID {cluster_id} ({method} 3D): Reused 2D cluster name: {cluster_name}"
                            )

                        # Update the cluster names in the DataFrame
                        df[f"{algorithm}_{method}_2D_name"] = df[f"{algorithm}_{method}_2D"].map(
                            lambda x: cluster_names.get((f"{algorithm}_{method}_2D", x), "Unknown")
                        )
                        df[f"{algorithm}_{method}_3D_name"] = df[f"{algorithm}_{method}_3D"].map(
                            lambda x: cluster_names.get((f"{algorithm}_{method}_2D", x), "Unknown")
                        )

    logger.info("Cluster naming process completed.")
    return df




if __name__ == "__main__":
   # Paths and parameters
   s_root = r"C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis"
   input_file = os.path.join(s_root, "Data", "db_clustered.json")
   output_file = os.path.join(s_root, "Data", "db_final.json")

   dimensionality_methods = ["UMAP", "PCA", "tSNE"]
   clustering_algorithms = ["hdbscan", "kmeans"]
   kmeans_clusters = [15]  # Number of clusters for KMeans
   max_centers = 8  # Maximum number of topics for naming

   # Load data
   df_total = load_json_into_df(input_file)

   # API settings
   chat_model_name = 'gpt-4o-mini'
   configure_api(client, chat_model_name)

   # Process clusters and generate names
   df_total = process_clusters(df_total, dimensionality_methods, clustering_algorithms, kmeans_clusters, max_centers,
                               api_settings)

   # Save results
   save_df_as_json(df_total, output_file)

