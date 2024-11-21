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

def load_embedded_data(json_path):
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

def dimensionality_reduction(mat, method, n_components, seed, perplexity=None):
    """
    Applies dimensionality reduction on the embeddings.
    Args:
        mat (np.ndarray): Matrix of embeddings.
        method (str): Dimensionality reduction method ('UMAP', 'PCA', 'tSNE').
        n_components (int): Number of components for reduction.
        seed (int): Random seed for reproducibility.
        perplexity (int, optional): Perplexity parameter for t-SNE. Ignored for other methods.
    Returns:
        np.ndarray: Reduced embeddings.
    """
    logger.info(f"Applying {method} with {n_components} components.")
    if method == 'UMAP':
        model = umap.UMAP(n_components=n_components, random_state=seed)
    elif method == 'PCA':
        model = PCA(n_components=n_components)
    elif method == 'tSNE':
        if perplexity is None:
            # Automatically set perplexity if not provided
            perplexity = min(30, mat.shape[0] - 1)  # Ensure it's less than n_samples
            logger.info(f"Perplexity not provided, setting to {perplexity} based on sample size.")
        else:
            # Validate perplexity
            if perplexity >= mat.shape[0]:
                raise ValueError(f"Invalid perplexity: {perplexity}. Must be less than n_samples ({mat.shape[0]}).")
            logger.info(f"Using perplexity={perplexity} for t-SNE.")
        model = TSNE(n_components=n_components, random_state=seed, perplexity=perplexity)
    else:
        raise ValueError("Invalid dimensionality reduction method specified.")
    return model.fit_transform(mat)


def apply_clustering(
    df,
    mat,
    dimensionality_methods,
    kmeans_clusters,
    output_path,
    hdbscan_params=None,
    kmeans_seed=42,
    include_2d=True,
    include_3d=True
):
    """
    Performs clustering and dimensionality reduction on the embeddings.
    Args:
        df (pd.DataFrame): DataFrame with embeddings.
        mat (np.ndarray): Matrix of embeddings.
        dimensionality_methods (list): List of dimensionality reduction methods ('UMAP', 'PCA', 'tSNE').
        kmeans_clusters (list): List of cluster counts for KMeans.
        output_path (str): Path to save the output JSON file.
        hdbscan_params (dict, optional): Parameters for HDBSCAN clustering.
        kmeans_seed (int, optional): Random seed for KMeans.
        include_2d (bool, optional): Whether to include 2D dimensionality reduction results.
        include_3d (bool, optional): Whether to include 3D dimensionality reduction results.
    """
    if hdbscan_params is None:
        hdbscan_params = {"min_cluster_size": 10, "min_samples": 8, "cluster_selection_epsilon": 0.5}

    results = []  # List to hold all new columns and their values

    for method in dimensionality_methods:
        for n_dims in [2, 3]:
            if (n_dims == 2 and not include_2d) or (n_dims == 3 and not include_3d):
                continue

            dim_suffix = '2D' if n_dims == 2 else '3D'

            # Dimensionality Reduction
            reduced_coords = dimensionality_reduction(mat, method, n_components=n_dims, seed=kmeans_seed)

            # HDBSCAN Clustering
            logger.info(f"Applying HDBSCAN on {method} {dim_suffix} with params: {hdbscan_params}")
            hdbscan_clusterer = hdbscan.HDBSCAN(**hdbscan_params)
            hdbscan_labels = hdbscan_clusterer.fit_predict(reduced_coords)

            # Collect HDBSCAN results
            hdbscan_data = pd.DataFrame({
                f'hdbscan_{method}_{dim_suffix}_x': reduced_coords[:, 0],
                f'hdbscan_{method}_{dim_suffix}_y': reduced_coords[:, 1],
                f'hdbscan_{method}_{dim_suffix}_z': reduced_coords[:, 2] if n_dims == 3 else None,
                f'hdbscan_{method}_{dim_suffix}': hdbscan_labels
            })
            results.append(hdbscan_data)

            # KMeans Clustering
            for n_clusters in kmeans_clusters:
                logger.info(f"Applying KMeans with {n_clusters} clusters on {method} {dim_suffix}.")
                kmeans_model = KMeans(n_clusters=n_clusters, random_state=kmeans_seed)
                kmeans_labels = kmeans_model.fit_predict(reduced_coords)

                # Collect KMeans results
                kmeans_data = pd.DataFrame({
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}_x': reduced_coords[:, 0],
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}_y': reduced_coords[:, 1],
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}_z': reduced_coords[:, 2] if n_dims == 3 else None,
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}': kmeans_labels
                })
                results.append(kmeans_data)

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
    output_file = os.path.join(s_root, "Data", "db_clustered.json")

    # Adjustable parameters
    dimensionality_methods = ['UMAP', 'PCA', 'tSNE']  # Dimensionality reduction methods
    kmeans_clusters = [5, 10, 15]  # Number of clusters for KMeans
    kmeans_seed = 42  # Seed for reproducibility
    include_2d = True  # Whether to include 2D results
    include_3d = True  # Whether to include 3D results
    hdbscan_params = {"min_cluster_size": 5, "min_samples": 3, "cluster_selection_epsilon": 0.2}  # HDBSCAN params

    # t-SNE specific parameter
    perplexity = 15  # Set to a default or user-defined value

    # Load data
    df_total = load_embedded_data(input_file)
    mat = np.array(df_total['embedding'].tolist())

    # Apply dimensionality reduction and clustering
    apply_clustering(
        df_total,
        mat,
        dimensionality_methods,
        kmeans_clusters,
        output_file,
        hdbscan_params=hdbscan_params,
        kmeans_seed=kmeans_seed,
        include_2d=include_2d,
        include_3d=include_3d
    )
