import json
import os
from pathlib import Path

from helper.prompt_templates import prompt_template_topic, prompt_template_sentiment
from helper.utils import *


def load_existing_progress(output_path, id_column):
    """
    Loads existing progress from the output file if it exists.
    Returns the processed data as a list and the set of processed IDs.

    Args:
        output_path (str): Path to the output JSON file.
        id_column (str): The column name where IDs are stored.

    Returns:
        tuple: A list of processed data and a set of processed IDs.
    """
    if Path(output_path).exists():
        logger.info(f"Loading existing progress from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            processed_data = json.load(f)
        processed_ids = {entry[id_column] for entry in processed_data}
    else:
        logger.info(f"No existing progress found. Starting fresh.")
        processed_data = []
        processed_ids = set()
    return processed_data, processed_ids


def save_progress(processed_data, output_path):
    """
    If the process gets interrupted, all progress up until that point is saved to the output file.
    (no separate backup file will be created, every progress gets appended to the output file)
    """
    try:
        logger.info(f"Saving progress to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")
        raise


def extract_topics(entry, entry_id, prompt_template_topic, api_settings, review_fields):
    """
    All specified review fields are combined into a single review text.
    The combined review text is then used to extract topics using the API.

    Args:
        entry (dict): The review entry containing multiple fields to combine.
        entry_id (str): The ID from the users specified ID column (only used for logging).
        prompt_template_topic (PromptTemplate): The template used for topic extraction.
        api_settings (dict): API configuration from utils.py.
        review_fields (list): List of fields to combine for the review.

    Returns:
        dict: Extracted topics in JSON format.
    """
    combined_review = " ".join(entry.get(field, "").strip() for field in review_fields)
    prompt_topic = prompt_template_topic.format(review=combined_review)
    logger.info(f"Extracting topics for entry ID {entry_id}")

    try:
        response = api_settings["client"].chat.completions.create(
            model=api_settings["model"],
            messages=[
                {"role": "system", "content": "You are an expert in extracting topics from user reviews."},
                {"role": "user", "content": prompt_topic},
            ],
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        track_tokens(response)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error extracting topics for entry ID {entry_id}: {e}")
        return {"error": str(e)}


def analyze_sentiments(entry, entry_id, topics, prompt_template_sentiment, api_settings):
    """
    Takes the topics extracted from extract_topics.
    Analyzes the sentiment of each topic using the API.
    Appends the sentiment data to the topics! -> no return value.

    Args:
        entry (dict): The JSON entry being processed.
        entry_id (str): The ID of the entry (only used for logging).
        topics (dict): Topics extracted from the review.
        prompt_template_sentiment (PromptTemplate): Template for sentiment analysis.
        api_settings (dict): API configuration from utils.py.
    """
    entry["topics"] = []
    for topic in topics.get("Topics", []):
        logger.info(f"Analyzing sentiment for topic '{topic['Topic']}' (Entry ID {entry_id})")
        try:
            prompt_sentiment = prompt_template_sentiment.format(
                review=topic["Context"], topic=topic["Topic"]
            )
            response = api_settings["client"].chat.completions.create(
                model=api_settings["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for sentiment analysis."},
                    {"role": "user", "content": prompt_sentiment},
                ],
                max_tokens=1024,
            )
            track_tokens(response)
            sentiment = response.choices[0].message.content.strip()
            entry["topics"].append({
                "topic": topic["Topic"],
                "sentiment": sentiment,
                "category": topic["Category"],
                "sentence": topic["Context"]
            })
        except Exception as e:
            logger.error(f"Error analyzing sentiment for topic '{topic['Topic']}' (Entry ID {entry_id}): {e}")
            raise


def process_entry(entry, id_column, prompt_template_topic, prompt_template_sentiment, api_settings, review_fields):
    """
    Processes a single entry by extracting topics (extract_topics)
    and analyzing their sentiments (analyze_sentiments).

    Args:
        entry (dict): The JSON entry to process.
        id_column (str): The ID column name.
        prompt_template_topic (PromptTemplate): Template for topic extraction.
        prompt_template_sentiment (PromptTemplate): Template for sentiment analysis.
        api_settings (dict): API configuration.
        review_fields (list): List of fields to combine for the review.

    Returns:
        dict: Processed entry with topics and sentiments.
        (after this step the progress is saved).
    """
    global prompt_tokens, completion_tokens
    logger.info(f"Tokens used so far: Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}")

    try:
        entry_id = entry.get(id_column, "unknown")
        topics = extract_topics(entry, entry_id, prompt_template_topic, api_settings, review_fields)
        analyze_sentiments(entry, entry_id, topics, prompt_template_sentiment, api_settings)
    except Exception as e:
        logger.error(f"Error processing entry ID {entry[{id_column}]}: {e}")
    return entry


def analyse_data(translated_data, id_column, output_path, prompt_template_topic, prompt_template_sentiment,
                 api_settings, review_fields, batch_size=10):
    """
    Main function to analyse translated data with fail-safe batching and progress saving.
    Flow how one entity is processed:
    load_existing_progress -> if entry not in processed_ids:
        -> process_entry -> extract_topics -> analyze_sentiments -> save_progress
    else: skip entry

    Args:
        translated_data (list): Dataset to process.
        id_column (str): Column name where IDs are stored.
        output_path (str): Path to save analysed data.
        prompt_template_topic (PromptTemplate): Template for topic extraction.
        prompt_template_sentiment (PromptTemplate): Template for sentiment analysis.
        api_settings (dict): API configuration.
        review_fields (list): List of review fields to combine.
        batch_size (int): Number of entries to process before saving progress (default 10).
    """
    # Load existing progress if available
    processed_data, processed_ids = load_existing_progress(output_path, id_column)

    try:
        batch_counter = 0
        for entry in translated_data:
            entry_id = entry.get(id_column)
            if entry_id in processed_ids:
                logger.info(f"Skipping already processed entry ID {entry_id}")
                continue

            # Entry has not been processed yet, therefore it gets injected into the process_entry function
            processed_entry = process_entry(
                entry, id_column, prompt_template_topic, prompt_template_sentiment, api_settings, review_fields
            )
            processed_data.append(processed_entry)
            processed_ids.add(entry_id)
            batch_counter += 1

            if batch_counter >= batch_size:
                save_progress(processed_data, output_path)
                logger.info(f"Progress saved after processing {batch_counter} entries.")
                batch_counter = 0

    # Save progress in case something goes wrong
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user. Saving progress...")
        save_progress(processed_data, output_path)
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        save_progress(processed_data, output_path)
        logger.info("Processing completed. Final progress saved.")
