import json
import os
from pathlib import Path
from lingua import Language, LanguageDetectorBuilder
from helper.utils import *
from helper.prompt_templates import *

# Initialize language detector
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.SPANISH, Language.CHINESE, Language.GERMAN, Language.FRENCH
).build()

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

# region Translation
import os
import pandas as pd
from lingua import Language, LanguageDetectorBuilder
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize language detector
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.SPANISH, Language.CHINESE, Language.GERMAN, Language.FRENCH
).build()

# region Translation

def detect_language_in_dataframe(df, text_column='review_text', language_column='language'):
    """
    Adds a 'language' column to a DataFrame by detecting the language of a text column.

    Args:
        df (pd.DataFrame): Input DataFrame with a text column.
        text_column (str): Name of the column containing text for language detection.
        language_column (str): Name of the new column to store detected languages.

    Returns:
        pd.DataFrame: Updated DataFrame with the 'language' column added.
    """
    # Check if the 'language' column already exists
    if language_column in df.columns:
        logger.info(f"'{language_column}' column already exists. Skipping language detection.")
        return df

    logger.info(f"Starting language detection for column: '{text_column}'")

    def detect_language(text):
        """Helper function to detect language using Lingua."""
        if pd.isnull(text) or not isinstance(text, str) or text.strip() == "":
            return 'unknown'
        try:
            detected_lang = detector.detect_language_of(text)
            return detected_lang.name.lower() if detected_lang else 'unknown'
        except Exception as e:
            logger.error(f"Error detecting language for text: {text[:30]}...: {e}")
            return 'unknown'

    # Apply the function row-wise and create the 'language' column
    df[language_column] = df[text_column].apply(detect_language)
    logger.info(f"Language detection completed. Added column '{language_column}'.")

    return df


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
        logger.info(f"Loading existing reviews from: {file_path}")
        existing_df = pd.read_pickle(file_path)
        existing_ids = set(existing_df[id_column].unique())
    else:
        logger.info("No existing file found. Starting fresh.")
        existing_df = pd.DataFrame(columns=df.columns)
        existing_ids = set()

    # Step 2: Process new rows
    new_data = []
    num_translated = 0  # Counter for translated reviews
    new_reviews_count = len(df[~df[id_column].isin(existing_ids)])
    logger.info(f"Found {new_reviews_count} new reviews to process.")

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
                    model=api_settings["model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant for translation."},
                        {"role": "user", "content": prompt_translation},
                    ],
                    max_tokens=4096
                )
                translated_text = response.choices[0].message.content.strip()
                row[text_column] = translated_text  # Overwrite with translated text
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
    logger.info(f"Translation completed. Total reviews translated: {num_translated}")

    return updated_df


# endregion


# region Extract Topics and Sentiments
def extract_topics(entry, entry_id, prompt_template_topic, api_settings, columns_of_interest):
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
    combined_review = " ".join(
        str(entry.get(field, "")).strip()  # Convert field value to string and strip whitespace
        for field in columns_of_interest
        if entry.get(field) is not None  # Ensure the field value is not None
    )
    prompt_topic = prompt_template_topic.format(review=combined_review)
    logger.info(f"Extracting topics for entry ID {entry_id}")

    try:
        response = api_settings["client"].chat.completions.create(
            model=api_settings["model"],
            messages=[
                {"role": "system", "content": "You are an expert in extracting topics from user reviews."},
                {"role": "user", "content": prompt_topic},
            ],
            max_tokens=4096,
            response_format={"type": "json_object"}
        )
        track_tokens(response)
        response_json = json.loads(response.choices[0].message.content)

        # Normalize the "topics" key
        normalized_response = normalize_topics_key(response_json)
        return normalized_response

        return
    except Exception as e:
        logger.error(f"Error extracting topics for entry ID {entry_id}: {e}")
        return {"error": str(e)}

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
    global prompt_tokens, completion_tokens
    logger.info(f"Tokens used so far: Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}")

    try:
        entry_id = entry.get(id_column, "unknown")
        topics = extract_topics(entry, entry_id, prompt_template_topic, api_settings, columns_of_interest)
        analyze_sentiments(entry, entry_id, topics, prompt_template_sentiment, api_settings)
    except Exception as e:
        logger.error(f"Error processing entry ID {entry[{id_column}]}: {e}")
    return entry

# endregion


# region Save/Load Progress
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

# endregion


# Main Function
def analyse_data(translated_data, id_column, output_path, prompt_template_topic, prompt_template_sentiment,
                 api_settings, columns_of_interest, batch_size=10):
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
        columns_of_interest (list): List of columns that should be combined.
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
            processed_entry = process_entry(entry, id_column, prompt_template_topic, prompt_template_sentiment,
                                            api_settings, columns_of_interest)
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


## Example Usage
if __name__ == "__main__":
    import os
    from helper.utils import *
    from helper.prompt_templates import *
    import openai
    from dotenv import load_dotenv


    root_dir = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis\Data\DEMO'
    input_file = os.path.join(root_dir, "db_translated.json")
    output_path = os.path.join(root_dir, "db_analysed.json")

    # Load API settings
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key
    client = openai.Client()
    chat_model_name = 'gpt-4o-mini'


    configure_api(client, chat_model_name)

    id_column = "recommendationid"  # Column name for entry IDs
    columns_of_interest = ["player_response"]  # Which cols should be analyzed?
    batch_size = 10  # Fail-safe batching. The higher the number, the less often the progress is saved.

    prepared_data = read_json(input_file)

    # Run analysis
    analyse_data(
        translated_data=prepared_data,
        id_column=id_column,
        output_path=output_path,
        prompt_template_topic=prompt_template_topic,
        prompt_template_sentiment=prompt_template_sentiment,
        api_settings=api_settings,
        columns_of_interest=columns_of_interest,
        batch_size=batch_size
    )


