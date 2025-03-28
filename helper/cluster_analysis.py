import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan

from helper.utils import logger


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

def dimensionality_reduction(mat, method, n_components, perplexity=None):
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
        model = umap.UMAP(n_components=n_components)
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
        model = TSNE(n_components=n_components, perplexity=perplexity)
    else:
        raise ValueError("Invalid dimensionality reduction method specified.")
    return model.fit_transform(mat)


def apply_hdbscan(
        df,
        mat,
        dimensionality_methods,
        hdbscan_params=None,
        include_2d=True,
        include_3d=True
):
    """
    Applies HDBSCAN clustering and dimensionality reduction for visualization.

    Args:
        df (pd.DataFrame): Input DataFrame containing embeddings
        mat (np.ndarray): Embedding matrix
        dimensionality_methods (list): List of dimensionality reduction methods
        hdbscan_params (dict, optional): Parameters for HDBSCAN clustering
        include_2d (bool, optional): Whether to include 2D reduction
        include_3d (bool, optional): Whether to include 3D reduction

    Returns:
        pd.DataFrame: DataFrame with clustering results and dimensional coordinates
    """
    # Default HDBSCAN parameters
    if hdbscan_params is None:
        hdbscan_params = {
            "min_cluster_size": 10,
            "min_samples": 8,
            "cluster_selection_epsilon": 0.5
        }

    # Perform Cluster Analysis
    logger.info(f"Applying HDBSCAN with: {hdbscan_params}")
    hdbscan_clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    cluster_labels = hdbscan_clusterer.fit_predict(mat)

    logger.info(f"Found {len(np.unique(cluster_labels))} clusters.")

    # Add primary cluster label
    df['hdbscan_cluster_id'] = cluster_labels

    # Dimensionality Reduction Results
    reduction_results = {}

    # Perform dimensionality reduction for each method
    for method in dimensionality_methods:
        # 2D Reduction
        if include_2d:
            coords_2d = dimensionality_reduction(mat, method, n_components=2)
            reduction_results[f'{method}_2D'] = {
                'x': coords_2d[:, 0],
                'y': coords_2d[:, 1]
            }

        # 3D Reduction
        if include_3d:
            coords_3d = dimensionality_reduction(mat, method, n_components=3)
            reduction_results[f'{method}_3D'] = {
                'x': coords_3d[:, 0],
                'y': coords_3d[:, 1],
                'z': coords_3d[:, 2]
            }

    # Add dimensional coordinates to DataFrame
    for method_dim, coords in reduction_results.items():
        for axis, values in coords.items():
            df[f'hdbscan_{method_dim}_{axis}'] = values

    logger.info("HDBSCAN clustering and dimensionality reduction completed.")
    return df


def apply_kmeans(
        df,
        mat,
        dimensionality_methods,
        kmeans_clusters,
        kmeans_seed=42,
        include_2d=True,
        include_3d=True
):
    """
    Applies KMeans clustering first, followed by dimensionality reduction.
    """
    results = []  # List to hold KMeans results

    for n_clusters in kmeans_clusters:
        logger.info(f"Applying KMeans with {n_clusters} clusters in high-dimensional space.")

        # Perform clustering in high-dimensional space
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=kmeans_seed)
        kmeans_labels = kmeans_model.fit_predict(mat)

        # Add high-dimensional cluster labels to the DataFrame
        df[f'kmeans_{n_clusters}_highdim'] = kmeans_labels

        for method in dimensionality_methods:
            for n_dims in [2, 3]:
                if (n_dims == 2 and not include_2d) or (n_dims == 3 and not include_3d):
                    continue

                dim_suffix = '2D' if n_dims == 2 else '3D'

                # Dimensionality Reduction
                logger.info(f"Applying {method} in {dim_suffix}.")
                reduced_coords = dimensionality_reduction(mat, method, n_components=n_dims, seed=kmeans_seed)

                # Collect results for visualization
                kmeans_data = pd.DataFrame({
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}_x': reduced_coords[:, 0],
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}_y': reduced_coords[:, 1],
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}_z': reduced_coords[:, 2] if n_dims == 3 else None,
                    f'kmeans_{n_clusters}_{method}_{dim_suffix}': kmeans_labels  # Reuse the same cluster labels
                })
                results.append(kmeans_data)

    # Combine KMeans results and return
    df_new_columns = pd.concat(results, axis=1)
    df = pd.concat([df, df_new_columns], axis=1)
    logger.info("KMeans clustering and dimensionality reduction completed.")

    return df

