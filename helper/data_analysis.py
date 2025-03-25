import json
import os
import pandas as pd
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder
import helper.utils as utils
from helper.utils import logger, api_settings, read_json
from helper.prompt_templates import *

# region Translation
# Initialize language detector
    # The detector works faster with fewer languages, if major ones are missing you can add them here
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.SPANISH, Language.CHINESE, Language.GERMAN, Language.FRENCH, Language.PORTUGUESE, Language.ARABIC, Language.RUSSIAN
).build()

# Detect language and translate non-English reviews
def translate_reviews(df, file_path, id_column='recommendationid', text_column='review_text',
                    language_column='language'):
    """
    Processes reviews by checking for existing translations, detecting language, and translating non-English reviews.
    New rows are appended to the existing file, and the updated DataFrame is saved.

    Args:
        df (pd.DataFrame): Input DataFrame containing reviews.
        file_path (str): Path to save or load the existing file.
        id_column (str): Column containing unique IDs for comparison.
        text_column (str): Column containing review text.
        language_column (str): Column to store detected languages.

    Returns:
        pd.DataFrame: Updated DataFrame with all reviews (existing and new).
    """
    # Step 1: Load existing data or start fresh
    if os.path.exists(file_path):
        utils.logger.info(f"Loading existing translated reviews from: {file_path}")
        existing_df = pd.read_pickle(file_path)
        existing_ids = set(existing_df[id_column].unique())
    else:
        utils.logger.info("No existing translation file found. Starting fresh.")
        existing_df = pd.DataFrame(columns=df.columns)
        existing_ids = set()

    # Step 2: Process new rows
    new_data = []
    num_translated = 0  # Counter for translated reviews
    new_reviews_count = len(df[~df[id_column].isin(existing_ids)])
    utils.logger.info(f"Found {new_reviews_count} new reviews to check for translation.")

    for _, row in df.iterrows():
        review_id = row[id_column]
        if review_id in existing_ids:
            continue  # Skip IDs that already exist

        text = row[text_column]
        if not isinstance(text, str) or not text.strip():
            continue  # Skip rows with invalid text

        try:
            # Detect language
            detected_language = detector.detect_language_of(text)
            if detected_language is not None:
                detected_language = detected_language.name.lower()
            else:
                detected_language = 'unknown'

            # If not English, translate the text
            if detected_language != 'english':
                logger.info(f"Translating review ID: {review_id} (Detected Language: {detected_language})")
                prompt_translation = prompt_template_translation.format(text=text)
                response = api_settings["client"].chat.completions.create(
                    model= api_settings["model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for translation."},
                        {"role": "user", "content": prompt_translation},
                    ]
                )
                utils.track_tokens(response)
                translated_text = response.choices[0].message.content.strip()
                logger.info(f'Total Tokens used: Prompt: {utils.prompt_tokens}, Completion: {utils.completion_tokens}')
                row[text_column] = translated_text  # Overwrite player statement with translated text
                num_translated += 1  # Increment translation counter

            # Append the updated row
            row[language_column] = detected_language
            new_data.append(row)

        except Exception as e:
            logger.error(f"Error processing review ID: {review_id} | {e}")

    # Step 3: Combine existing and new rows
    if new_data:
        new_df = pd.DataFrame(new_data)
        if existing_df.empty:
            updated_df = new_df  # If no existing data, use new_df directly
        else:
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        updated_df.to_pickle(file_path)
        logger.info(f"Updated file saved to: {file_path}")
    else:
        logger.info("No new reviews to add. All IDs already exist.")
        updated_df = existing_df  # No new data, return existing_df as is

    # Final log summary
    logger.info(f"###### Translation completed. Total reviews translated: {num_translated} ######{os.linesep}")

    return updated_df
# endregion


# region Extract Topics and Sentiments
def extract_topics(entry, entry_id, prompt_template_topic, api_settings, review_column):
    """
    All specified review fields are combined into a single review text.
    The combined review text is then used to extract topics using the API.

    Args:
        entry (dict): The review entry containing multiple fields to combine.
        entry_id (str): The ID from the users specified ID column (only used for logging).
        prompt_template_topic (PromptTemplate): The template used for topic extraction.
        api_settings (dict): API configuration from utils.py.
        columns_of_interest (list): List of fields to combine for the review.

    Returns:
        dict: Extracted topics in JSON format.
    """

    # prompt_topic = prompt_template_topic.format(review=entry[review_column])
    prompt_topic = prompt_template_topic_zendesk.format(zendesk_ticket=entry[review_column])
    logger.info(f"Extracting topics for entry ID {entry_id}")

    try:
        response = api_settings["client"].chat.completions.create(
            model= api_settings["model"],
            messages=[
                {"role": "system", "content": "You are an expert in extracting topics from user feedback."},
                {"role": "user", "content": prompt_topic},
            ],
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        utils.track_tokens(response)
        response_json = json.loads(response.choices[0].message.content)

        # Normalize the "topics" key
        normalized_response = normalize_topics_key(response_json)
        return normalized_response

        return
    except Exception as e:
        logger.error(f"Error extracting topics for entry ID {entry_id}: {e}")
        return {"error": str(e)}

# This function currently handles the OpenAI output. It makes sure that "topic" is always written in the same way.
# TODO: This should at some point be handled with pydantic
def normalize_topics_key(response_json):
    """
    Normalize the key for topics in the outermost JSON object to "topics" (case-insensitive).
    """
    if not isinstance(response_json, dict):
        logger.error("The JSON object is not a dictionary.")

    # Find the key regardless of case
    for key in response_json.keys():
        if key.lower() == "topics":
            # Standardize the key to "topics"
            response_json["topics"] = response_json.pop(key)
            return response_json

    # If no "topics" key is found, raise an error
    logger.error("The JSON does not contain a valid 'topics' key.")


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
    for topic in topics.get("topics", []):
        logger.info(f"Analyzing sentiment for topic '{topic['Topic']}' (Entry ID {entry_id})")
        try:
            prompt_sentiment = prompt_template_sentiment.format(
                review=topic["Context"], topic=topic["Topic"]
            )
            response = api_settings["client"].chat.completions.create(
                model= api_settings["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for sentiment analysis."},
                    {"role": "user", "content": prompt_sentiment},
                ],
                max_tokens=1024,
            )
            utils.track_tokens(response)
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


def process_entry(entry, id_column, prompt_template_topic, prompt_template_sentiment, api_settings, columns_of_interest):
    """
    Processes a single entry by extracting topics (extract_topics)
    and analyzing their sentiments (analyze_sentiments).

    Args:
        entry (dict): The JSON entry to process.
        id_column (str): The ID column name.
        prompt_template_topic (PromptTemplate): Template for topic extraction.
        prompt_template_sentiment (PromptTemplate): Template for sentiment analysis.
        api_settings (dict): API configuration.
        columns_of_interest (list): List of fields to combine for the review.

    Returns:
        dict: Processed entry with topics and sentiments.
        (after this step the progress is saved).
    """

    try:
        entry_id = entry.get(id_column, "unknown")
        topics = extract_topics(entry, entry_id, prompt_template_topic, api_settings, columns_of_interest)
        analyze_sentiments(entry, entry_id, topics, prompt_template_sentiment, api_settings)
        logger.info(f'Total Tokens used: Prompt: {utils.prompt_tokens}, Completion: {utils.completion_tokens}')
    except Exception as e:
       logger.error(f"Error processing entry ID {entry[{id_column}]}: {e}")
    return entry

# endregion



