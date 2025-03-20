import json
import pandas as pd
import numpy as np
import umap

import logging
import chromadb

from helper.embedding import get_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.ERROR)


# region 0: Helper functions
# for the check existing embeddings to work the data needs to be flattened after the embedding step (directly before we use it)
def flatten_data(data):
    flattened = []
    for entry in data:
        base_copy = dict(entry)
        topics = base_copy.pop("topics", [])

        for topic in topics:
            new_entry = dict(base_copy)
            new_entry.update(topic)
            flattened.append(new_entry)
    return flattened

# Prepare the Data to be stored in the ChromaDB
def prepare_dataframe(
    data_source: str,
    input_json_path: str,
    output_csv_path: str = None,
    dimensions: int = 25
) -> pd.DataFrame:
    """
    1) Load JSON data into a Pandas DataFrame.
    2) Flatten the data.
    3) Generate an incremental Statement ID: 'steam_1', 'steam_2', ...
    4) Reduce embedding dimensions to 25 via UMAP, save as 'embedding_short'.
    5) Optional: Write the DataFrame to a CSV file.
    """

    # 1) Load JSON data
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # data is expected to be a list of dicts

    # 2) Flatten the data
    data = flatten_data(data)
    df = pd.DataFrame(data)

    # 3) Generate an incremental ID: steam_1, steam_2, ...
    df["pp_statement_id"] = [f"{data_source}_{i+1}" for i in range(len(df))]

    # 4) Generate embedding_short with UMAP
    embeddings_matrix = np.vstack(df["embedding"].values)
    reducer = umap.UMAP(n_components=dimensions)
    embedding_25d = reducer.fit_transform(embeddings_matrix)
    df["embedding_short"] = embedding_25d.tolist()

    # Optionally write to CSV
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)

    # Fill NaN values with empty strings
    df = df.fillna("")
    return df
#endregion


#region 1 Chroma interaction
def upsert_chroma_data(
    df: pd.DataFrame,
    collection_name: str,
    persist_path: str = "chroma_data",
    batch_size: int = 100
) -> None:
    """
    Upsert data into Chroma without duplicates:
      1) If the collection doesn't exist, create it and insert ALL rows.
      2) If it does exist, skip any IDs that are already there, and insert only new rows.
         (No duplicates, no warnings).

    DataFrame columns:
      - 'pp_statement_id' (unique ID for each row)
      - 'embedding' (list of floats or JSON string)
      - 'sentence' (optional; will be stored as 'documents')
      - plus other columns for metadata

    Args:
        df (pd.DataFrame): Data with 'pp_statement_id' and 'embedding' columns (and optional 'sentence').
        collection_name (str): The name of the Chroma collection.
        persist_path (str): Folder path for persistent Chroma storage.
        batch_size (int): Number of rows to insert per batch (default=100).
    """
    logger.info("Initializing persistent Chroma client via `PersistentClient`.")
    chroma_client = chromadb.PersistentClient(path=persist_path)

    # 1) Check if collection exists
    existing_collections = chroma_client.list_collections()
    collection_exists = (collection_name in existing_collections)

    # 2) Create or get collection
    if not collection_exists:
        logger.info(f"Collection '{collection_name}' does NOT exist. Creating new collection.")
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "hnsw:search_ef": 100}
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists. Retrieving it.")
        collection = chroma_client.get_collection(name=collection_name)

    # 3) Ensure 'embedding' is a list of floats
    def ensure_list(val):
        if isinstance(val, str):
            return json.loads(val)  # e.g. "[0.123, 0.456]" -> [0.123, 0.456]
        return val

    df["embedding"] = df["embedding"].apply(ensure_list)

    # 4) Prepare all IDs, embeddings, etc.
    all_ids = df["pp_statement_id"].astype(str).tolist()
    all_embeddings = df["embedding"].tolist()

    # If there's a 'sentence' column, use it for documents; otherwise use empty strings
    documents = (
        df["sentence"].fillna("").astype(str).tolist()
        if "sentence" in df.columns else [""] * len(df)
    )

    # Exclude 'pp_statement_id' and 'embedding' from metadata
    exclude_cols = {"pp_statement_id", "embedding"}
    meta_columns = [col for col in df.columns if col not in exclude_cols]

    metadatas = []
    for _, row in df.iterrows():
        meta = {}
        for col in meta_columns:
            val = row[col]
            # If it's a list, store it as JSON
            if isinstance(val, list):
                val = json.dumps(val)
            meta[col] = val
        metadatas.append(meta)

    # 5) If collection already existed, figure out which IDs are new
    if collection_exists:
        logger.info(f"Checking which of the {len(all_ids)} IDs already exist in the collection.")
        # Query the DB for these IDs (include=[] means we don't fetch documents, embeddings, etc.)
        existing_data = collection.get(ids=all_ids, include=[])
        found_ids = set(existing_data["ids"]) if existing_data["ids"] else set()

        # Filter out the rows whose IDs are already in the collection
        new_ids = []
        new_embeddings = []
        new_docs = []
        new_metas = []
        for i, pid in enumerate(all_ids):
            if pid not in found_ids:
                new_ids.append(pid)
                new_embeddings.append(all_embeddings[i])
                new_docs.append(documents[i])
                new_metas.append(metadatas[i])

        # If no new rows, we're done
        if not new_ids:
            logger.info("All IDs already exist in the collection. No new data to insert.")
            return

        logger.info(f"{len(found_ids)} IDs already existed, {len(new_ids)} are new. Inserting only the new ones.")
        # Insert new items in batches
        _insert_in_batches(
            collection,
            new_ids,
            new_embeddings,
            new_docs,
            new_metas,
            batch_size,
            collection_name
        )

    else:
        # Collection is newly created, so everything in df is new
        logger.info(f"Collection is new. Inserting all {len(df)} rows.")
        _insert_in_batches(
            collection,
            all_ids,
            all_embeddings,
            documents,
            metadatas,
            batch_size,
            collection_name
        )

    logger.info(f"Upsert complete. Collection '{collection_name}' updated with no duplicates.")


def _insert_in_batches(collection, ids, embeddings, documents, metadatas, batch_size, collection_name):
    """
    Helper to insert data in batches into a Chroma collection.
    """
    total = len(ids)
    for start_idx in range(0, total, batch_size):
        end_idx = start_idx + batch_size
        sub_ids = ids[start_idx:end_idx]
        sub_emb = embeddings[start_idx:end_idx]
        sub_docs = documents[start_idx:end_idx]
        sub_metas = metadatas[start_idx:end_idx]

        collection.add(
            ids=sub_ids,
            embeddings=sub_emb,
            documents=sub_docs,
            metadatas=sub_metas
        )
        logging.info(f"Inserted rows {start_idx} to {end_idx - 1} into '{collection_name}'.")


def query_chroma(
        collection_name: str,
        persist_path: str,
        query_text: str = None,
        similarity_threshold: float = 0.54,
        initial_top_n: int = 1000,
        where_filters: dict = None
) -> pd.DataFrame:
    """
    Queries a ChromaDB collection. Two modes:

    1) Vector Similarity Search (if `query_text` is provided):
       - Embeds `query_text` using OpenAI
       - Returns up to `initial_top_n` most similar items
       - Then filters them by `distance <= similarity_threshold`
       - Optionally applies a metadata filter (where_filters) at query-time

    2) Metadata-Only Retrieval (if `query_text=None`):
       - Returns ALL items that match `where_filters` (like "WHERE ..." in SQL).
       - No distance or similarity filtering performed (because no query).

    Returns a pandas DataFrame with columns:
      - 'pp_id'
      - 'distance' (only for vector queries; if metadata-only, no 'distance')
      - 'document'
      - plus any metadata fields flattened out
    """
    logger.info(f"Connecting to Chroma at '{persist_path}'.")
    chroma_client = chromadb.PersistentClient(path=persist_path)

    logger.info(f"Retrieving collection '{collection_name}'.")
    collection = chroma_client.get_collection(name=collection_name)

    if query_text is not None:
        # SEMANTIC SEARCH + optional metadata filters
        logger.info("Performing VECTOR-based query (semantic search).")
        query_vector = get_embedding(query_text)

        # Build arguments for the query
        query_args = {
            "query_embeddings": [query_vector],
            "n_results": initial_top_n,
            "include": ["distances", "documents", "metadatas"]
        }
        if where_filters:
            query_args["where"] = where_filters

        results = collection.query(**query_args)
        print("test")

        ids = results["ids"][0]
        distances = results["distances"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Filter by similarity threshold
        filtered_data = [
            (id_, dist, doc, meta)
            for id_, dist, doc, meta in zip(ids, distances, documents, metadatas)
            if dist <= similarity_threshold
        ]
        if not filtered_data:
            logger.info("No results found above the similarity threshold.")
            return pd.DataFrame()

        # Unzip filtered data
        filtered_ids, filtered_distances, filtered_docs, filtered_metas = zip(*filtered_data)
        df_out = pd.DataFrame({
            "pp_id": filtered_ids,
            "distance": filtered_distances,
            "document": filtered_docs
        })
        df_meta = pd.json_normalize(filtered_metas)
        df_final = pd.concat([df_out, df_meta], axis=1)

        # Drop or rename embedding fields if needed
        if "embedding" in df_final.columns:
            df_final.drop(columns=["embedding"], inplace=True)
        if "embedding_short" in df_final.columns:
            df_final["embedding_short"] = df_final["embedding_short"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
            df_final.rename(columns={"embedding_short": "embedding"}, inplace=True)

        return df_final

    else:
        # METADATA-ONLY retrieval
        logger.info("Performing METADATA-based retrieval (no similarity ranking).")

        get_args = {
            "include": ["documents", "metadatas"],
        }
        if where_filters:
            get_args["where"] = where_filters

        results = collection.get(**get_args)

        ids = results["ids"] or []
        documents = results["documents"] or []
        metadatas = results["metadatas"] or []

        if len(ids) == 0:
            logger.info("No documents found for the given filters.")
            return pd.DataFrame()

        df_out = pd.DataFrame({"pp_id": ids, "document": documents})
        df_meta = pd.json_normalize(metadatas)
        df_final = pd.concat([df_out, df_meta], axis=1)

        # For metadata-only retrieval, no distance column
        if "embedding" in df_final.columns:
            df_final.drop(columns=["embedding"], inplace=True)
        if "embedding_short" in df_final.columns:
            df_final["embedding_short"] = df_final["embedding_short"].apply(
                lambda x: json.loads(x) if isinstance(x, str) else x
            )
            df_final.rename(columns={"embedding_short": "embedding"}, inplace=True)

        return df_final

#endregion