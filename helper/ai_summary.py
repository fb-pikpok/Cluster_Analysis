import pandas as pd
from helper.prompt_templates import *
from helper.utils import api_settings, logger


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


def generate_hard_facts(df):
    """
    Summarize a DataFrame containing columns:
      - 'sentence' : the user/player statement (str)
      - 'sentiment': the sentiment label (e.g. 'positive', 'negative', 'inconclusive')
      - 'hdbscan_id_name': name of the cluster
      - 'topic' : a short "headline" or summary for each statement

    Returns a DataFrame with:
      - total_data_points
      - sentiment counts (positive, negative, inconclusive)
      - a list of all unique topics for that cluster
    """

    # 1. Count the total data points (statements) per cluster
    cluster_counts = (
        df.groupby('hdbscan_id_name', dropna=False)['sentence']
          .count()
          .reset_index(name='total_data_points')
    )

    # 2. Count how many are positive, negative, etc. in each cluster
    sentiment_counts = (
        df.groupby(['hdbscan_id_name', 'sentiment'])['sentence']
          .count()
          .reset_index(name='count')
    )

    # 3. Pivot the sentiment counts so each cluster is one row, with separate sentiment columns
    sentiment_pivot = (
        sentiment_counts
        .pivot_table(index='hdbscan_id_name',
                     columns='sentiment',
                     values='count',
                     fill_value=0)
        .reset_index()
    )

    # 4. Merge the pivot back with the total cluster counts
    cluster_summary_df = cluster_counts.merge(
        sentiment_pivot,
        on='hdbscan_id_name',
        how='left'
    )
    # 4.5. Exclude "Noise" cluster
    cluster_summary_df = cluster_summary_df[cluster_summary_df['hdbscan_id_name'] != "Noise"]

    # 5. Rename the sentiment columns for clarity (if they exist)
    rename_map = {}
    for col in ['positive', 'negative', 'inconclusive']:
        if col in cluster_summary_df.columns:
            rename_map[col] = f"{col}_count"
    cluster_summary_df.rename(columns=rename_map, inplace=True)

    # 6. Get all unique topic values per cluster
    topics_per_cluster = (
        df.groupby('hdbscan_id_name')['topic']
          .unique()
          .reset_index(name='unique_topics')
    )

    # 7. Merge unique topics into the cluster summary
    cluster_summary_df = cluster_summary_df.merge(
        topics_per_cluster,
        on='hdbscan_id_name',
        how='left'
    )

    # Sort by cluster name if you like (optional)
    cluster_summary_df.sort_values(by='hdbscan_id_name', inplace=True)

    return cluster_summary_df


def generate_cluster_report(
    df: pd.DataFrame
):
        # 1. Count the total data points (statements) per cluster
    cluster_counts = (
        df.groupby('hdbscan_id_name', dropna=False)['sentence']
          .count()
          .reset_index(name='number_of_statements')
    )

    # 2. Count how many are positive, negative, etc. in each cluster
    sentiment_counts = (
        df.groupby(['hdbscan_id_name', 'sentiment'])['sentence']
          .count()
          .reset_index(name='count')
    )

    # 3. Pivot the sentiment counts so each cluster is one row, with separate sentiment columns
    sentiment_pivot = (
        sentiment_counts
        .pivot_table(index='hdbscan_id_name',
                     columns='sentiment',
                     values='count',
                     fill_value=0)
        .reset_index()
    )

    # 4. Merge the pivot back with the total cluster counts
    cluster_summary_df = cluster_counts.merge(
        sentiment_pivot,
        on='hdbscan_id_name',
        how='left'
    )
    # 4.5. Exclude "Noise" cluster
    cluster_summary_df = cluster_summary_df[cluster_summary_df['hdbscan_id_name'] != "Noise"]

    # 5. Rename the sentiment columns for clarity (if they exist)
    rename_map = {}
    for col in ['positive', 'negative', 'inconclusive']:
        if col in cluster_summary_df.columns:
            rename_map[col] = f"{col}_count"
    cluster_summary_df.rename(columns=rename_map, inplace=True)


    # Sort by total_data_points DESCENDING so the largest cluster is first
    cluster_summary_df.sort_values(by='number_of_statements', ascending=False, inplace=True)


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

        total_points = row['number_of_statements']
        positive_count = int(row.get('Positive', 0))
        negative_count = int(row.get('Negative', 0))
        inconclusive_count = int(row.get('Inconclusive', 0))

        # ---------------------------------------------------
        # 2a) Write "hard facts" as a small Markdown table
        # ---------------------------------------------------
        markdown_report += f"## Cluster: {cluster_name}\n\n"
        markdown_report += "### Hard Facts\n\n"
        markdown_report += "| Metric              | Value |\n"
        markdown_report += "|---------------------|-------|\n"
        markdown_report += f"| **Total Statements**       | {total_points} |\n"
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
        prompt_topic = prompt_template_summary_short.format(cluster_name=cluster_name, statements =statements_text, video_game = "Rival Stars Horse Racing")
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
            summary_text = response.choices[0].message.content.strip()
            logger.info(f"Total tokens used: {prompt_tokens + completion_tokens}")

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
