import pandas as pd
import openai
from helper.prompt_templates import *

import logging

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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


def generate_cluster_report(
    df: pd.DataFrame
):
    # ---------------------------------------------------
    # STEP 1: Build the "hard facts" table for each cluster
    # ---------------------------------------------------

    # 1a) Count total data points per cluster
    cluster_counts = (
        df.groupby('hdbscan_id_name', dropna=False)['sentence']
          .count()
          .reset_index(name='total_data_points')
    )

    # 1b) Sentiment counts per cluster
    sentiment_counts = (
        df.groupby(['hdbscan_id_name', 'sentiment'])['sentence']
          .count()
          .reset_index(name='count')
    )

    # 1c) Pivot to get separate columns for each sentiment label
    sentiment_pivot = (
        sentiment_counts
        .pivot_table(index='hdbscan_id_name',
                     columns='sentiment',
                     values='count',
                     fill_value=0)
        .reset_index()
    )

    # 1d) Merge pivot with total counts
    cluster_summary_df = cluster_counts.merge(
        sentiment_pivot,
        on='hdbscan_id_name',
        how='left'
    )

    # 1e) Exclude "Noise" cluster
    cluster_summary_df = cluster_summary_df[cluster_summary_df['hdbscan_id_name'] != "Noise"]

    # Rename columns for clarity if they exist
    rename_map = {}
    for col in ['positive', 'negative', 'inconclusive']:
        if col in cluster_summary_df.columns:
            rename_map[col] = f"{col}_count"
    cluster_summary_df.rename(columns=rename_map, inplace=True)

    # Sort by cluster name for a consistent order (optional)
    cluster_summary_df.sort_values(by='hdbscan_id_name', inplace=True)

    logger.info("Data is structured, processing those clusters: ")
    logger.info(cluster_summary_df['hdbscan_id_name'].values)

    # ---------------------------------------------------
    # STEP 2: For each cluster, gather its statements and produce an LLM summary
    # ---------------------------------------------------

    # We'll accumulate everything into one big string of Markdown
    markdown_report = "# Cluster Report\n\n"

    # Create a dict mapping cluster -> list of statements
    cluster_to_statements = (
        df.groupby('hdbscan_id_name')['sentence']
          .apply(list)
          .to_dict()
    )

    for _, row in cluster_summary_df.iterrows():
        cluster_name = row['hdbscan_id_name']

        total_points = row['total_data_points']
        positive_count = int(row.get('positive_count', 0))
        negative_count = int(row.get('negative_count', 0))
        inconclusive_count = int(row.get('inconclusive_count', 0))

        # ---------------------------------------------------
        # 2a) Write "hard facts" as a small Markdown table
        # ---------------------------------------------------
        markdown_report += f"## Cluster: {cluster_name}\n\n"
        markdown_report += "### Hard Facts\n\n"
        markdown_report += "| Metric              | Value |\n"
        markdown_report += "|---------------------|-------|\n"
        markdown_report += f"| **Total Data Points**       | {total_points} |\n"
        markdown_report += f"| **Positive Count**          | {positive_count} |\n"
        markdown_report += f"| **Negative Count**          | {negative_count} |\n"
        markdown_report += f"| **Inconclusive Count**      | {inconclusive_count} |\n\n"

        # ---------------------------------------------------
        # 2b) Get statements and call the OpenAI API for summary
        # ---------------------------------------------------
        statements_for_this_cluster = cluster_to_statements[cluster_name]
        # If many statements, consider chunking or sampling
        statements_text = "\n".join(f"- {s}" for s in statements_for_this_cluster)

#######################
        prompt_topic = prompt_template_summary.format(cluster_name=cluster_name, statements =statements_text)
        logger.info(f"Generate AI summary for cluster {cluster_name}")

        try:
            response = api_settings["client"].chat.completions.create(
                model=api_settings["model"],
                messages=[
                    {"role": "system", "content": "You are an expert summarizing user statements for a video game."},
                    {"role": "user", "content": prompt_topic},
                ]
            )
            track_tokens(response)
            summary_text = response.text
            test = response["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Error summarizing cluster {cluster_name}: {e}")
            return {"error": str(e)}


        # ---------------------------------------------------
        # 2c) Append the summary to the Markdown
        # ---------------------------------------------------
        markdown_report += "### Key Insights\n\n"
        markdown_report += summary_text + "\n\n"
        markdown_report += "---\n\n"

    # ---------------------------------------------------
    # STEP 3: Write the final Markdown to file
    # ---------------------------------------------------
    logger.info("Markdown Report has been written.")

    return markdown_report


if __name__ == "__main__":

    markdown_report = generate_cluster_report(df_example)
