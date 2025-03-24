import pandas as pd
from helper.utils import api_settings, logger, track_tokens, prompt_tokens, completion_tokens
from helper.prompt_templates import prompt_template_top5, prompt_template_summary_short
import random

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
    logger.info("Cluster report has been written.")

    return markdown_report


def generate_big_picture_summary(df: pd.DataFrame,
                                 project: str) -> str:
    """
    Produces a Markdown string with overall stats tables AND
    a single LLM-based summary for the top N clusters in each category,
    using your existing prompt framework and api_settings for the OpenAI client.

    Unlike before, we do *not* summarize each cluster individually.
    Instead, we gather all statements from the top 5 clusters of each category,
    then produce a single summary per category.
    """

    # Ensure timestamps are datetime (if not already)
    if not pd.api.types.is_datetime64_any_dtype(df['pp_timestamp']):
        df['pp_timestamp'] = pd.to_datetime(df['pp_timestamp'], errors='coerce')

    # --- 1) Aggregate cluster data ---
    cluster_counts = (
        df.groupby('hdbscan_id_name', dropna=False)['sentence']
          .count()
          .reset_index(name='total_data_points')
    )

    # --- 2) Summaries of sentiment distribution ---
    sentiment_counts = (
        df.groupby(['hdbscan_id_name', 'sentiment'])['sentence']
          .count()
          .reset_index(name='count')
    )
    pivot_sent = (
        sentiment_counts
        .pivot_table(index='hdbscan_id_name', columns='sentiment',
                     values='count', fill_value=0)
        .reset_index()
    )

    cluster_summary_df = cluster_counts.merge(pivot_sent, on='hdbscan_id_name', how='left')

    # Convert numeric columns to int
    for col in ['total_data_points', 'Positive', 'Negative', 'Inconclusive']:
        if col in cluster_summary_df.columns:
            cluster_summary_df[col] = cluster_summary_df[col].fillna(0).astype(int)

    # Compute sentiment percentages
    cluster_summary_df['negative_percentage'] = (
        cluster_summary_df['Negative'] / cluster_summary_df['total_data_points'] * 100
    ).fillna(0).round(1)

    cluster_summary_df['positive_percentage'] = (
        cluster_summary_df['Positive'] / cluster_summary_df['total_data_points'] * 100
    ).fillna(0).round(1)

    # Count requests
    request_df = (
        df[df['category'] == 'request']
        .groupby('hdbscan_id_name')['sentence']
        .count()
        .reset_index(name='request_count')
    )
    cluster_summary_df = cluster_summary_df.merge(request_df, on='hdbscan_id_name', how='left')
    cluster_summary_df['request_count'] = cluster_summary_df['request_count'].fillna(0).astype(int)

    # Exclude Noise for ranking
    non_noise_df = cluster_summary_df[cluster_summary_df['hdbscan_id_name'] != "Noise"]

    # Identify Noise Cluster Size
    noise_count = 0
    if "Noise" in cluster_summary_df['hdbscan_id_name'].values:
        noise_count = cluster_summary_df.loc[
            cluster_summary_df['hdbscan_id_name'] == "Noise", 'total_data_points'
        ].values[0]

    # --- 3) Select top clusters to display in tables ---
    top_5_neg = non_noise_df.sort_values(by='negative_percentage', ascending=False).head(5)
    top_5_pos = non_noise_df.sort_values(by='positive_percentage', ascending=False).head(5)
    top_5_req = non_noise_df.sort_values(by='request_count', ascending=False).head(5)
    top_5_overall = non_noise_df.sort_values(by='total_data_points', ascending=False).head(5)

    # --- Helper: Convert DataFrame to Markdown ---
    def table_to_md(table_df, columns, percentage_cols=None):
        """
        Converts a DataFrame into a Markdown table.
        - Rounds and adds a '%' sign to specified percentage columns.
        """
        md_table = "| " + " | ".join(columns) + " |\n"
        md_table += "|-" + "-|-".join(["-" * len(col) for col in columns]) + "-|\n"
        for _, row in table_df.iterrows():
            row_values = []
            for col in columns:
                if percentage_cols and col in percentage_cols:
                    row_values.append(f"{row[col]:.1f}%")
                else:
                    row_values.append(f"{row[col]}")
            md_table += "| " + " | ".join(row_values) + " |\n"
        return md_table

    # --- 4) Summarize top clusters in a single prompt ---

    def summarize_topN_clusters(
        df: pd.DataFrame,
        top_df: pd.DataFrame,
        cluster_group: str,
        sentiment: str
    ) -> str:
        """
        Gathers ALL statements from the top 'N' clusters in top_df.
        If there are more than 150 statements, randomly sample 150.
        Then calls the LLM *once* to summarize them all together.
        """

        # 1) Collect all cluster names
        top_cluster_names = top_df['hdbscan_id_name'].unique().tolist()

        # 2) Gather all statements from these clusters
        mask = df['hdbscan_id_name'].isin(top_cluster_names)
        selected_statements = df.loc[mask, 'sentence'].tolist()

        # 3) If more than 150, randomly sample
        if len(selected_statements) > 150:
            logger.info(f"More than 150 statements in the cluster group {cluster_group}. Sampling 150.")
            selected_statements = random.sample(selected_statements, 150)
        else:
            logger.info(f"{cluster_group} has {len(selected_statements)} player statements. Continuing...")

        # 4) Build the prompt text
        statements_text = "\n".join(f"- {s}" for s in selected_statements)

        # 5) Format the prompt
        prompt_topic = prompt_template_top5.format(
            video_game="Into the Dead",
            cluster_group=cluster_group,
            sentiment=sentiment,
            statements=statements_text
        )

        logger.info(f"Generate AI summary for top 5 {cluster_group} clusters.")
        try:
            response = api_settings["client"].chat.completions.create(
                model=api_settings["model"],
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert summarizing user statements for a video game."
                    },
                    {
                        "role": "user",
                        "content": prompt_topic
                    }
                ]
            )
            track_tokens(response)
            summary_text = response.choices[0].message.content.strip()
            logger.info(f"Tokens used so far: {prompt_tokens + completion_tokens}")
            return summary_text

        except Exception as e:
            logger.error(f"Error summarizing top {cluster_group}: {e}")
            return f"Error summarizing top {cluster_group} clusters: {e}"

    # --- 5) Construct the Markdown Report ---
    markdown_report = "# Big Picture Report\n\n"
    markdown_report += f"**Data Source:** {project}\n\n"
    markdown_report += f"**Time Range:** {df['pp_timestamp'].min()} - {df['pp_timestamp'].max()}\n\n"
    markdown_report += f"**Total Statements:** {len(df)}, Noise: {noise_count}\n\n"

    # -- 5a) Top 5 Negative Clusters
    markdown_report += "### Top 5 Negative Clusters\n\n"
    markdown_report += table_to_md(
        top_5_neg,
        ['hdbscan_id_name', 'negative_percentage'],
        percentage_cols=['negative_percentage']
    ) + "\n\n"

    neg_summary = summarize_topN_clusters(
        df,
        top_5_neg,
        cluster_group="Negative",
        sentiment="negative"
    )
    markdown_report += f"**Summary for Top 5 Negative Clusters:**\n{neg_summary}\n\n"

    # -- 5b) Top 5 Positive Clusters
    markdown_report += "### Top 5 Positive Clusters\n\n"
    markdown_report += table_to_md(
        top_5_pos,
        ['hdbscan_id_name', 'positive_percentage'],
        percentage_cols=['positive_percentage']
    ) + "\n\n"

    pos_summary = summarize_topN_clusters(
        df,
        top_5_pos,
        cluster_group="Positive",
        sentiment="positive"
    )
    markdown_report += f"**Summary for Top 5 Positive Clusters:**\n{pos_summary}\n\n"

    # -- 5c) Top 5 Request Clusters
    markdown_report += "### Top 5 Request Clusters\n\n"
    markdown_report += table_to_md(
        top_5_req,
        ['hdbscan_id_name', 'request_count']
    ) + "\n\n"

    req_summary = summarize_topN_clusters(
        df,
        top_5_req,
        cluster_group="Request",
        sentiment="wishes or requests"
    )
    markdown_report += f"**Summary for Top 5 Request Clusters:**\n{req_summary}\n\n"

    # -- 5d) Top 5 Clusters Overall
    markdown_report += "### Top 5 Clusters (Overall)\n\n"
    markdown_report += table_to_md(top_5_overall, ['hdbscan_id_name', 'total_data_points']) + "\n\n"

    overall_summary = summarize_topN_clusters(
        df,
        top_5_overall,
        cluster_group="biggest",
        sentiment="largest"
    )
    markdown_report += f"**Summary for Top 5 Overall Largest Clusters:**\n{overall_summary}\n\n"

    logger.info("Big Picture summary has been generated.")
    return markdown_report