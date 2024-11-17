import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import hdbscan

# Paths
s_root = os.path.join(os.path.dirname(__file__), '../')
s_db_embed = os.path.join(s_root, 'Data/review_db_table.json')
s_output = os.path.join(s_root, 'Data/sentence_embeddings.json')


# Load DataFrame with embeddings
def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    return df


# Dimensionality reduction methods
def dimensionality_reduction(mat, method, n_components=3):
    if method == 'UMAP':
        model = umap.UMAP(n_components=n_components, random_state=42)
    elif method == 'PCA':
        model = PCA(n_components=n_components)
    elif method == 'tSNE':
        model = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Invalid method specified")
    return model.fit_transform(mat)


# Perform HDBSCAN and KMeans clustering with multiple configurations
def preprocess_data():
    # Load data and embeddings
    df_total = load_data(s_db_embed)
    mat = np.array(df_total['embedding'].tolist())

    # Define dimensionality reduction and clustering configurations
    dimensionality_methods = ['UMAP', 'PCA', 'tSNE']
    kmeans_clusters = [5, 10, 15, 20, 35]

    for method in dimensionality_methods:
        # Reduce to 2D and 3D for each method
        for n_dims in [2, 3]:
            reduced_coords = dimensionality_reduction(mat, method, n_components=n_dims)
            dim_suffix = '2D' if n_dims == 2 else '3D'

            # Apply HDBSCAN on reduced data and save coordinates with clustering algorithm prefix
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=8, cluster_selection_epsilon=0.5)
            hdbscan_labels = hdbscan_clusterer.fit_predict(reduced_coords)

            # Assign coordinates and labels with updated naming convention for HDBSCAN
            df_total[f'hdbscan_{method}_{dim_suffix}_x'] = reduced_coords[:, 0]
            df_total[f'hdbscan_{method}_{dim_suffix}_y'] = reduced_coords[:, 1]
            if n_dims == 3:
                df_total[f'hdbscan_{method}_{dim_suffix}_z'] = reduced_coords[:, 2]
            df_total[f'hdbscan_{method}_{dim_suffix}'] = hdbscan_labels

            # Apply KMeans with multiple cluster counts and save coordinates with clustering algorithm prefix
            for n_clusters in kmeans_clusters:
                kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans_labels = kmeans_model.fit_predict(reduced_coords)

                # Assign coordinates and labels with updated naming convention for KMeans
                df_total[f'kmeans_{n_clusters}_{method}_{dim_suffix}_x'] = reduced_coords[:, 0]
                df_total[f'kmeans_{n_clusters}_{method}_{dim_suffix}_y'] = reduced_coords[:, 1]
                if n_dims == 3:
                    df_total[f'kmeans_{n_clusters}_{method}_{dim_suffix}_z'] = reduced_coords[:, 2]
                df_total[f'kmeans_{n_clusters}_{method}_{dim_suffix}'] = kmeans_labels

    # Save results to JSON
    df_total.to_json(s_output, orient='records')
    print(f"Preprocessing complete. Results saved to {s_output}")


if __name__ == "__main__":
    preprocess_data()
