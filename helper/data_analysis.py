import json
import logging
from lingua import Language, LanguageDetectorBuilder
from pathlib import Path

# Imports for testing purposes
from helper.prompt_templates import prompt_template_translation, prompt_template_topic, prompt_template_topic_view
import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
client = openai.Client()


# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


# Initialize token counters
prompt_tokens = 0
completion_tokens = 0

# Initialize the language detector
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.SPANISH, Language.CHINESE, Language.GERMAN, Language.FRENCH
).build()


# Global API settings
api_settings = {"client": None, "model": None}

def configure_api(api_client, model_name):
    """
    Configures the global API client and model.
    Args:
        api_client: The initialized OpenAI client.
        model_name (str): The model name to use.
    """
    global api_settings
    api_settings["client"] = api_client
    api_settings["model"] = model_name


def track_tokens(response):
    """
    Updates the global token counters based on the response.
    """
    global prompt_tokens, completion_tokens
    prompt_tokens += response.usage.prompt_tokens
    completion_tokens += response.usage.completion_tokens

def detect_language(reason_text, wish_text):
    """
    Detects the combined language of the given text fields.
    """
    def detect_single_text_language(text):
        if isinstance(text, str) and text.strip():
            try:
                detected_language = detector.detect_language_of(text)
                return detected_language.name.lower() if detected_language else "none"
            except AttributeError:
                return "none"
        return "none"

    detected_language_reason = detect_single_text_language(reason_text)
    detected_language_wish = detect_single_text_language(wish_text)

    if detected_language_reason != "none" and detected_language_reason == detected_language_wish:
        return detected_language_reason
    elif detected_language_reason != "none" and detected_language_reason != detected_language_wish:
        return "mixed"
    elif detected_language_reason != "none":
        return detected_language_reason
    elif detected_language_wish != "none":
        return detected_language_wish
    return "unknown"

def translate_entry(entry, prompt_template_translation):
    """
    Translates the review.
    """
    reason_text = entry.get("Please tell us why you chose the rating above:", "")
    wish_text = entry.get("If you had a magic wand and you could change, add, or remove anything from the game, what would it be and why?", "")

    detected_language = detect_language(reason_text, wish_text)
    entry["language"] = detected_language

    if detected_language not in ["english", "none"]:
        logger.info(f"Translating entry ID {entry['ID']} (Language: {detected_language})")
        try:
            prompt_translation = prompt_template_translation.format(
                reason=reason_text or "N/A",
                wish=wish_text or "N/A"
            )
            response = api_settings["client"].chat.completions.create(
                model=api_settings["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for translation."},
                    {"role": "user", "content": prompt_translation},
                ],
                max_tokens=1024
            )
            track_tokens(response)
            translation_text = response.choices[0].message.content
            if "REASON:" in translation_text:
                entry["Please tell us why you chose the rating above:"] = translation_text.split("REASON:")[1].split("WISH:")[0].strip()
            if "WISH:" in translation_text:
                entry["If you had a magic wand and you could change, add, or remove anything from the game, what would it be and why?"] = translation_text.split("WISH:")[1].strip()
        except Exception as e:
            logger.error(f"Error translating entry ID {entry['ID']}: {e}")
            raise

def extract_topics(entry, prompt_template_topic):
    """
    Extracts topics from an entry's combined review.
    """
    combined_review = f"{entry.get('Please tell us why you chose the rating above:', '')} {entry.get('If you had a magic wand and you could change, add, or remove anything from the game, what would it be and why?', '')}"
    prompt_topic = prompt_template_topic.format(review=combined_review)

    logger.info(f"Extracting topics for entry ID {entry['ID']}")
    try:
        response = api_settings["client"].chat.completions.create(
            model=api_settings["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant for game review analysis."},
                {"role": "user", "content": prompt_topic},
            ],
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        track_tokens(response)
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error extracting topics for entry ID {entry['ID']}: {e}")
        raise

def analyze_sentiments(entry, topics, prompt_template_topic_view):
    """
    Performs sentiment analysis on extracted topics.
    """
    entry["topics"] = []
    for topic in topics.get("Topics", []):
        logger.info(f"Analyzing sentiment for topic '{topic['Topic']}' (Entry ID {entry['ID']})")
        prompt_sentiment = prompt_template_topic_view.format(
            review=topic["Context"],
            topic=topic["Topic"]
        )
        try:
            response = api_settings["client"].chat.completions.create(
                model=api_settings["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant expert in sentiment analysis."},
                    {"role": "user", "content": prompt_sentiment},
                ],
                max_tokens=1024
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
            logger.error(f"Error analyzing sentiment for topic '{topic['Topic']}' (Entry ID {entry['ID']}): {e}")
            raise


def process_entry(entry, prompt_template_translation, prompt_template_topic, prompt_template_topic_view, client, model):
    """
    Processes a single entry: language detection, translation, topic extraction, and sentiment analysis.
    """
    global prompt_tokens, completion_tokens
    logger.info(f"Tokens used so far: Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}")
    logger.info(f"Processing entry ID {entry['ID']}")

    try:
        translate_entry(entry, prompt_template_translation, client, model)
        topics = extract_topics(entry, prompt_template_topic, client, model)
        analyze_sentiments(entry, topics, prompt_template_topic_view, client, model)
    except Exception as e:
        logger.error(f"Error processing entry ID {entry['ID']}: {e}")
        raise

# region start Save Data in case of Interruption and load previous progress
def load_existing_progress(progress_file):
    """
    Loads the existing progress file if it exists.
    Returns the processed data as a dictionary and the set of processed IDs.
    """
    if Path(progress_file).exists():
        logger.info(f"Loading existing progress from {progress_file}")
        with open(progress_file, 'r', encoding='utf-8') as f:
            processed_data = json.load(f)
        processed_ids = {entry["ID"] for entry in processed_data}
    else:
        logger.info(f"No existing progress found. Starting fresh.")
        processed_data = []
        processed_ids = set()
    return processed_data, processed_ids

def save_progress(processed_data, progress_file):
    """
    Saves the current progress to a file.
    """
    try:
        logger.info(f"Saving progress to {progress_file}")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save progress: {e}")
        raise

def analyse_data(entries, processed_data, processed_ids, progress_file, batch_size=10):
    """
    Processes a batch of entries and saves progress periodically.
    """
    for entry in entries:
        if entry["ID"] in processed_ids:
            logger.info(f"Skipping already processed entry ID {entry['ID']}")
            continue

        try:
            process_entry(entry)
            processed_data.append(entry)
            processed_ids.add(entry["ID"])

            # Save progress after each batch
            if len(processed_data) % batch_size == 0:
                save_progress(processed_data, progress_file)
        except KeyboardInterrupt:
            logger.warning("Processing interrupted by user. Saving progress...")
            save_progress(processed_data, progress_file)
            raise
        except Exception as e:
            logger.error(f"Error processing entry ID {entry['ID']}: {e}")

    # Save final progress after completing the batch
    save_progress(processed_data, progress_file)

def process_entry(entry):
    """
    Processes a single entry: language detection, translation, topic extraction, and sentiment analysis.
    """
    logger.info(f"Processing entry ID {entry['ID']}")
    translate_entry(entry, prompt_template_translation)
    topics = extract_topics(entry, prompt_template_topic)
    analyze_sentiments(entry, topics, prompt_template_topic_view)

# endregion


if __name__ == "__main__":
    root_dir = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis'
    final_json_path = os.path.join(root_dir, "Data", "db_prepared.json")
    output_file_path = os.path.join(root_dir, "Data", "db_analysed.json")
    progress_backup_file_path = os.path.join(root_dir, "Data", "db_progress_backup.json")

    chat_model_name = 'gpt-4o-mini'

    # Load the database
    with open(final_json_path, 'r', encoding='utf-8') as f:
        db = json.load(f)

    # Load existing progress
    processed_data, processed_ids = load_existing_progress(progress_backup_file_path)

    # Process entries in batches
    try:
        analyse_data(db, processed_data, processed_ids, progress_backup_file_path, batch_size=10)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Progress saved.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

    # Save final output after all processing
    save_progress(processed_data, output_file_path)
    logger.info(f"Processing completed. Final data saved to {output_file_path}")