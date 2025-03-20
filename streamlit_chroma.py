import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
import umap
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

import logging
 # so we can use chromadb.PersistentClient

from helper.cluster_naming import generate_cluster_name
from helper.chroma_handler import query_chroma
from helper.utils import configure_api

openai_embedding_model = "text-embedding-3-small"
configure_api(model_name=openai_embedding_model)

# -----------------------------------------------------------------------------
# 0) Initial Setup & Logging
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide")  # wide layout
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)


# collection_name = 'HRC_multi_source'
# persist_path = r'S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Database\ChromaDB'
# collection_name = "my_collection_1536"
# persist_path = r"S:\SID\Analytics\Working Files\Individual\Florian\Projects\semantic_search\Data\ChromaDB"

initial_top_n = 1000
persist_path = r'S:\SID\Analytics\Working Files\Individual\Florian\Projects\DataScience\cluster_analysis\Database\ChromaDB'
collection_name = 'HRC_Zendesk'


@st.cache_data
def cluster_and_reduce(df: pd.DataFrame,
                       min_cluster_size: int,
                       min_samples: int,
                       cluster_selection_epsilon: float,
                       calculation_methode: str = "eom"
                       ) -> pd.DataFrame:
    df = df[df['embedding'].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    if len(df) == 0:
        return df

    mat = np.array(df['embedding'].tolist())

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=calculation_methode
    )
    labels = clusterer.fit_predict(mat)
    df["hdbscan_id"] = labels

    reducer = umap.UMAP(n_components=2)
    coords_2d = reducer.fit_transform(mat)
    df["x"] = coords_2d[:, 0]
    df["y"] = coords_2d[:, 1]

    # 3) t-SNE Dimensionality Reduction
    tsne = TSNE(n_components=2, init='pca')
    coords_tsne = tsne.fit_transform(mat)
    df["hdbscan_tSNE_2D_x"] = coords_tsne[:, 0]
    df["hdbscan_tSNE_2D_y"] = coords_tsne[:, 1]

    return df



def name_clusters(df: pd.DataFrame,
                  cluster_col: str = "hdbscan_id",
                  embedding_col: str = "embedding",
                  text_col: str = "document",
                  top_k: int = 10,
                  skip_noise_label: int = -1) -> pd.DataFrame:
    df_out = df.copy()
    unique_ids = df_out[cluster_col].unique()
    cluster_id_to_name = {}

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


def assemble_where_filters(
    enable_data_source: bool, data_sources_selected: bool,
    enable_category: bool, category_value: str,
    enable_language: bool, language_value: str
) -> dict | None:
    """
    Dynamically build the metadata filters dict based on user input.
    Each filter is only added if its checkbox is enabled.
    """
    conditions = []

    # DATA SOURCE Filter
    if enable_data_source and data_sources_selected:
        if len(data_sources_selected) == 1:
            conditions.append({"pp_data_source": {"$eq": data_sources_selected[0]}})
        else:
            conditions.append({"pp_data_source": {"$in": data_sources_selected}})

    # CATEGORY Filter
    if enable_category and category_value:
        if len(category_value) == 1:
            conditions.append({"category": {"$eq": category_value[0]}})
        else:
            conditions.append({"category": {"$in": category_value}})

    # LANGUAGE Filter
    if enable_language and language_value:
        if len(language_value) == 1:
            conditions.append({"language": {"$eq": language_value[0]}})
        else:
            conditions.append({"language": {"$in": language_value}})


    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def main():
    st.title("Database interaction DEMO")

    # -------------------------------
    # SEMANTIC SEARCH
    # -------------------------------
    enable_semantic = st.checkbox("**Enable Semantic Search?**", value=False)

    col_sem_left, col_sem_right = st.columns([0.7, 0.3])
    query_text = None
    similarity_threshold = 0.54

    if enable_semantic:
        with col_sem_left:

            st.markdown("#### What topic do you want to investigate (Semantic)?")
            query_text = st.text_input(
                "Enter your query:",
                value="The developers do a great job with the updates"
            )

            similarity_threshold = st.slider(
                "Distance Threshold (similarity filter)",
                min_value=0.0, max_value=1.0,
                value=0.75, step=0.01
            )

        with col_sem_right:
            st.markdown("#### Tips for Semantic Search")
            st.markdown(":blue[A threshold between **0.45** and **0.65** often works well.]")
            st.markdown(":blue[If the results are not relevant enough, try **lowering** the threshold to be more strict.]")


    st.markdown("---")

    # -------------------------------
    # FILTER by DATA SOURCE
    # -------------------------------
    enable_data_source = st.checkbox("Filter by Data Source?", value=False)
    data_sources_selected = []

    col_ds_left, col_ds_right = st.columns([0.7, 0.3])

    if enable_data_source:
        with col_ds_left:
            st.markdown("#### Select the Data Sources to Include")
            possible_sources = ["Google Play", "Survey_Brand_Trust_07022025", "Steam"]
            data_sources_selected = st.multiselect(
                "Select data source(s):",
                options=possible_sources,
                default=["Google Play"]
            )

        with col_ds_right:
            st.markdown("#### Tips for Data Source Filter")
            st.markdown(":blue[Restrict your results to only certain sources, e.g. 'Google Play', 'Survey', etc.]")

    st.markdown("---")

    # -------------------------------
    # FILTER by CATEGORY
    # -------------------------------
    enable_category = st.checkbox("Filter by Category?", value=False)
    category_value = []

    col_cat_left, col_cat_right = st.columns([0.7, 0.3])
    if enable_category:
        with col_cat_left:

            st.markdown("#### Specify Category to Include")
            category_value = st.multiselect(
                "Select category:",
                options=["fact", "request"],
                default=["request"]
            )

        with col_cat_right:
            st.markdown("#### Tips for Category Filter")
            st.markdown(":blue[**Category** is either **fact** or **request**]")
            st.markdown(":blue[If you want both just disable this filter.]")

    st.markdown("---")

    # -------------------------------
    # FILTER by language
    # -------------------------------

    enable_language = st.checkbox("Filter by Language?", value=False)
    language_value = []

    col_lang_left, col_lang_right = st.columns([0.7, 0.3])
    with col_lang_left:
        if enable_language:
            st.markdown("#### Specify Language to Include")
            language_value = st.multiselect(
                "Select language:",
                options=["english", "chinese"],
                default=["english"]
            )

    with col_lang_right:
        if enable_language:
            st.markdown("#### Tips for Language Filter")
            st.markdown(":blue[**Language** is either **english** or **chinese**]")
            st.markdown(":blue[If you want both just disable this filter.]")

    st.markdown("---")



    # --------------------------------------------------------------------------
    where_filters = assemble_where_filters(
        enable_data_source, data_sources_selected,
        enable_category, category_value,
        enable_language, language_value
    )

    if "query_done" not in st.session_state:
        st.session_state["query_done"] = False

    # --------------------------------------------------------------------------
    # STEP 1: Run Query
    # --------------------------------------------------------------------------
    if st.button("Run Query"):
        with st.spinner("Querying Chroma..."):
            df_results = query_chroma(persist_path= persist_path,
                                      collection_name=collection_name,
                                      query_text=query_text if enable_semantic else None,
                                      similarity_threshold=similarity_threshold, initial_top_n=initial_top_n,
                                      where_filters=where_filters)

        st.session_state["results_df"] = df_results
        st.session_state["query_done"] = True
        st.session_state["cluster_done"] = False
        st.session_state["name_done"] = False

        st.session_state["headline"] = (
            f"**{len(df_results)}** results found "
        )

    if st.session_state["query_done"] and "results_df" in st.session_state:
        df = st.session_state["results_df"]
        if len(df) == 0:
            st.warning("No results found. Try adjusting your filters or threshold.")
            st.stop()
        else:
            st.subheader(st.session_state["headline"])

            if "distance" in df.columns:
                df_sorted = df.sort_values(by="distance", ascending=True)
                c1, c2, c3 = st.columns([0.35, 0.35, 0.3])
                with c1:
                    st.write("**Closest (Head)**")
                    st.dataframe(df_sorted.head(5))
                with c2:
                    st.write("**Farthest (Tail)**")
                    st.dataframe(df_sorted.tail(5))
                with c3:
                    st.markdown("#### How to interpret these results?")
                    st.markdown(":blue[**Head** = statements with the smallest distance to your query ⇒ more similar.]")
                    st.markdown(":blue[**Tail** = statements with the largest distance still below the threshold, possibly less similar.]")
            else:
                st.info("No distance column (metadata-only retrieval). Showing first 20 rows:")
                st.dataframe(df.head(20))

    else:
        st.stop()

    # --------------------------------------------------------------------------
    # STEP 2: Clustering
    # --------------------------------------------------------------------------
    if st.session_state["query_done"] and len(st.session_state["results_df"]) > 0:
        st.markdown("---")
        st.markdown("### Step 2: Cluster the Data")

        col3, col4 = st.columns([0.6, 0.4])
        with col3:
            min_cluster_size = st.number_input("min_cluster_size", value=10, min_value=2, max_value=100)
            min_samples = st.number_input("min_samples", value=2, min_value=1, max_value=100)
            cluster_selection_epsilon = st.slider("cluster_selection_epsilon",
                                                  min_value=0.0, max_value=1.0,
                                                  value=0.15, step=0.01)
            calculation_methode = st.selectbox("Calculation Methode", ["eom", "leaf"])

            if st.button("Cluster Data"):
                df_query_results = st.session_state["results_df"]
                if len(df_query_results) == 0:
                    st.warning("No data to cluster. Please run a query first.")
                else:
                    with st.spinner("Clustering + dimensionality reduction..."):
                        df_clustered = cluster_and_reduce(
                            df_query_results,
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_epsilon=cluster_selection_epsilon,
                            calculation_methode=calculation_methode
                        )
                    st.session_state["df_clustered"] = df_clustered
                    st.session_state["cluster_done"] = True
                    st.session_state["name_done"] = False

        with col4:
            st.markdown("#### Tips for Clustering Parameters")
            st.markdown(''':blue-background[**min_cluster_size**: smallest cluster you'd expect to see]''')
            st.markdown(''':blue-background[**min_samples**: how dense a region must be to form a cluster]''')
            st.markdown(''':blue-background[**cluster_selection_epsilon**: 0 for default; increase to merge clusters]''')

    if st.session_state["cluster_done"] and "df_clustered" in st.session_state:
        df_clustered = st.session_state["df_clustered"]
        if len(df_clustered) == 0:
            st.warning("Clustering produced no valid rows. Possibly no embeddings exist in the filtered data.")
        else:
            n_clusters = df_clustered["hdbscan_id"].nunique()
            st.subheader(f"Found {n_clusters} cluster(s) (including noise).")

            col5, col6 = st.columns([0.7, 0.3])
            with col5:
                fig = px.scatter(
                    df_clustered,
                    x="x",
                    y="y",
                    # Instead of color="hdbscan_id", pass the categorical version:
                    color=df_clustered["hdbscan_id"].astype("category"),
                    hover_data=["hdbscan_id", "document"],
                    width=800,
                    height=600
                )
                st.plotly_chart(fig)

            with col6:
                st.markdown("#### Interpreting the cluster plot")
                st.markdown(''':blue-background[Each point is an embedding in 2D via UMAP. Colors = cluster IDs from HDBSCAN.]''')
    else:
        st.stop()

    # --------------------------------------------------------------------------
    # STEP 3: Name Clusters
    # --------------------------------------------------------------------------
    if st.session_state["cluster_done"] and len(st.session_state["df_clustered"]) > 0:
        st.markdown("---")
        if st.button("Name Clusters"):
            df_clustered = st.session_state["df_clustered"]
            with st.spinner("Naming clusters..."):
                df_named = name_clusters(
                    df_clustered,
                    cluster_col="hdbscan_id",
                    embedding_col="embedding",
                    text_col="document",
                    top_k=10,
                    skip_noise_label=-1
                )
            st.session_state["df_clustered"] = df_named
            st.session_state["name_done"] = True

    if st.session_state["name_done"] and "df_clustered" in st.session_state:
        df_named = st.session_state["df_clustered"]
        st.success("Cluster naming complete!")
        col7, col8 = st.columns([0.7, 0.3])
        with col7:
            fig2 = px.scatter(
                df_named,
                x="x", y="y",
                color="hdbscan_id_name",
                hover_data=["hdbscan_id_name", "document"],
                width=800, height=600
            )
            st.plotly_chart(fig2)

        with col8:
            st.markdown("#### Interpreting Named Clusters")
            st.markdown(''':blue-background[We used an LLM or a function to label clusters with a short descriptive name.]''')
            st.markdown(''':blue-background[Noise points (ID = -1) remain "Noise".]''')

        # Download JSON
        st.markdown("---")
        st.info("All steps done! You can now download the final DataFrame below:")


        #for testing purposes apply some mutations so the other streamlit app can handle the data
        final_df = df_named.copy()
        final_df.rename(columns={"x": "hdbscan_UMAP_2D_x", "y": "hdbscan_UMAP_2D_y"}, inplace=True)
        # populate all google play entries with a number between 10 and 1000 for the playtime in minutes
        # final_df.loc[final_df['pp_data_source'] == 'Google Play', 'playtime_at_review_minutes'] = np.random.randint(10, 1000, len(final_df[final_df['pp_data_source'] == 'Google Play']))



        json_str = final_df.to_json(orient="records")
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name="clustered_data.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
