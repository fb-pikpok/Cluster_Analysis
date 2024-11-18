import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(json_path):
    """
    Loads data from a JSON file and filters valid embeddings.
    Args:
        json_path (str): Path to the JSON file.
    Returns:
        pd.DataFrame: DataFrame with valid embeddings.
    """
    logger.info(f"Loading data from {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    logger.info(f"Loaded {len(df)} valid entries with embeddings.")
    return df

def dimensionality_reduction(mat, method, n_components=3):
    """
    Applies dimensionality reduction on the embeddings.
    Args:
        mat (np.ndarray): Matrix of embeddings.
        method (str): Dimensionality reduction method ('UMAP', 'PCA', 'tSNE').
        n_components (int): Number of components for reduction.
    Returns:
        np.ndarray: Reduced embeddings.
    """
    logger.info(f"Applying {method} with {n_components} components.")
    if method == 'UMAP':
        model = umap.UMAP(n_components=n_components, random_state=42)
    elif method == 'PCA':
        model = PCA(n_components=n_components)
    elif method == 'tSNE':
        model = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Invalid dimensionality reduction method specified.")
    return model.fit_transform(mat)


def apply_clustering(df, mat, dimensionality_methods, kmeans_clusters, output_path):
    """
    Performs clustering and dimensionality reduction on the embeddings.
    Args:
        df (pd.DataFrame): DataFrame with embeddings.
        mat (np.ndarray): Matrix of embeddings.
        dimensionality_methods (list): List of dimensionality reduction methods ('UMAP', 'PCA', 'tSNE').
        kmeans_clusters (list): List of cluster counts for KMeans.
        output_path (str): Path to save the output JSON file.
    """
    results = []  # List to hold all new columns and their values

    for method in dimensionality_methods:
        for n_dims in [2, 3]:
            dim_suffix = '2D' if n_dims == 2 else '3D'

            # Dimensionality Reduction
            reduced_coords = dimensionality_reduction(mat, method, n_components=n_dims)

            # HDBSCAN Clustering
            logger.info(f"Applying HDBSCAN on {method} {dim_suffix}.")
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=8, cluster_selection_epsilon=0.5)
            hdbscan_labels = hdbscan_clusterer.fit_predict(reduced_coords)

            # Collect HDBSCAN results
            results.append(pd.DataFrame({
                f'hdbscan_{method}_{dim_suffix}_x': reduced_coords[:, 0],
                f'hdbscan_{method}_{dim_suffix}_y': reduced_coords[:, 1],
                f'hdbscan_{method}_{dim_suffix}_z': reduced_coords[:, 2] if n_dims == 3 else None,
                f'hdbscan_{method}_{dim_suffix}': hdbscan_labels
            }))

            # KMeans Clustering
            for n_clusters in kmeans_clusters:
                logger.info(f"Applying KMeans with {n_clusters} clusters on {method} {dim_suffix}.")
                kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans_labels = kmeans_model.fit_predict(reduced_coords)

                # Collect KMeans results
                results.append(pd.DataFrame({
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}_x': reduced_coords[:, 0],
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}_y': reduced_coords[:, 1],
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}_z': reduced_coords[:, 2] if n_dims == 3 else None,
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}': kmeans_labels
                }))

    # Concatenate all new columns to the original DataFrame
    df_new_columns = pd.concat(results, axis=1)
    df = pd.concat([df, df_new_columns], axis=1)

    # Save the final DataFrame to JSON
    logger.info(f"Saving results to {output_path}")
    df.to_json(output_path, orient='records', indent=4)
    logger.info("Clustering and dimensionality reduction completed.")


if __name__ == "__main__":
    # Paths
    s_root = r"C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis"
    input_file = os.path.join(s_root, "Data", "db_embedded_table.json")
    output_file = os.path.join(s_root, "Data", "sentence_embeddings.json")

    # Adjustable parameters
    dimensionality_methods = ['UMAP', 'PCA', 'tSNE']  # Dimensionality reduction methods
    kmeans_clusters = [5, 10, 15, 20, 35]  # Number of clusters for KMeans

    # Load data
    df_total = load_data(input_file)
    mat = np.array(df_total['embedding'].tolist())

    # Apply dimensionality reduction and clustering
    apply_clustering(df_total, mat, dimensionality_methods, kmeans_clusters, output_file)
