import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances
import hdbscan
import umap
import plotly.express as px
import matplotlib.pyplot as plt
import os

# Load the JSON data
root_dir = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis\Data\Steamapps\Market/'
input_path = os.path.join(root_dir, "openai_2_cluster.json")
data = pd.read_json(input_path)

# Extract embeddings
embeddings = np.array(data['embedding'].tolist())

# Step 1: Perform KMeans clustering
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['kmeans_cluster_id'] = kmeans.fit_predict(embeddings)

# Step 2: Perform DBSCAN within each KMeans cluster
sub_cluster_data = []
for cluster_id in range(n_clusters):
    cluster_indices = data[data['kmeans_cluster_id'] == cluster_id].index
    cluster_embeddings = embeddings[cluster_indices]

    if len(cluster_embeddings) < 2:  # Skip clusters with too few points
        data.loc[cluster_indices, 'dbscan_cluster_id'] = -1
        continue

    dbscan = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
    sub_cluster_labels = dbscan.fit_predict(cluster_embeddings)
    data.loc[cluster_indices, 'dbscan_cluster_id'] = sub_cluster_labels

# Step 3: Reduce dimensions for visualization
umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
umap_embeddings = umap_reducer.fit_transform(embeddings)
data['umap_x'] = umap_embeddings[:, 0]
data['umap_y'] = umap_embeddings[:, 1]

# Step 4: Visualization of KMeans clusters
plt.figure(figsize=(10, 7))
plt.scatter(data['umap_x'], data['umap_y'], c=data['kmeans_cluster_id'], cmap='tab20', s=10)
plt.title("KMeans Clusters (UMAP)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.colorbar(label="KMeans Cluster ID")
plt.show()

# Step 5: Visualization of DBSCAN sub-clusters within KMeans clusters
fig = px.scatter(
    data, x='umap_x', y='umap_y', color='dbscan_cluster_id',
    facet_col='kmeans_cluster_id',
    title='DBSCAN Sub-Clusters within KMeans Clusters',
    hover_data=['sentence', 'topic', 'sentiment']
)
fig.show()

# Save results to JSON for further use
output_path = os.path.join(root_dir, "openai_3_named.json")
data.to_json(output_path, orient='records', lines=True)
