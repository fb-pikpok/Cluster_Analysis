import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px

s_root = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis/'
s_db_table_pca_json = 'Data/review_db_table_pca.json'

# Load Data
@st.cache_data
def load_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df

# Embed keyword for similarity (replace with actual embedding function)
def index_embedding(keyword):
    # Placeholder function for keyword embedding; replace with actual embedding logic
    return np.random.rand(len(df_total['embedding'][0]))

# Load preprocessed data with embeddings and clustering results
df_total = load_data(s_root + s_db_table_pca_json)

# Sidebar filters
st.sidebar.header("Filters")
selected_cluster = st.sidebar.selectbox("Select Cluster", sorted(df_total['kmeans'].unique()))
selected_sentiment = st.sidebar.multiselect("Select Sentiment", df_total['sentiment'].unique(), default=df_total['sentiment'].unique())

# Filter data based on selections
filtered_df = df_total[(df_total['kmeans'] == selected_cluster) & (df_total['sentiment'].isin(selected_sentiment))]

# Full PCA 3D Visualization with varied colors
st.subheader("3D PCA Visualization")
fig_pca_3d = px.scatter_3d(
    df_total,
    x='first_dim_PCA',
    y='second_dim_PCA',
    z='third_dim_PCA',
    color='kmeans',
    hover_data=['topic', 'sentence', 'similarity'],
    color_discrete_sequence=px.colors.qualitative.Bold  # or use any other color palette
)
st.plotly_chart(fig_pca_3d)

# Full UMAP Visualization with varied colors
st.subheader("UMAP Visualization")
fig_umap = px.scatter(
    df_total,
    x='first_dim_UMAP', y='second_dim_UMAP',
    color='kmeans',
    hover_data=['topic', 'sentence', 'similarity'],
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig_umap)

# Full t-SNE Visualization with varied colors
st.subheader("t-SNE Visualization")
fig_tsne = px.scatter(
    df_total,
    x='first_dim_t-SNE', y='second_dim_t-SNE',
    color='kmeans',
    hover_data=['topic', 'sentence', 'similarity'],
    color_discrete_sequence=px.colors.qualitative.Bold
)
st.plotly_chart(fig_tsne)

# Cluster Details
st.subheader("Cluster Details")
st.write(f"Showing details for cluster {selected_cluster}")
st.dataframe(filtered_df[['topic', 'sentence', 'similarity', 'category', 'embedding']])

# Keyword Similarity Search
st.subheader("Keyword Similarity Search")
keyword = st.text_input("Enter keyword to search similar topics")
if keyword:
    keyword_embed = index_embedding(keyword)
    filtered_df['similarity'] = filtered_df['embedding'].apply(lambda x: np.dot(x, keyword_embed))
    similar_topics = filtered_df.sort_values(by='similarity', ascending=False)
    st.write("Top Matches:")
    st.dataframe(similar_topics[['topic', 'sentence', 'similarity']])

# Run the Streamlit app with `streamlit run app.py`
