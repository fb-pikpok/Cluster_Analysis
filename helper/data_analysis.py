import json
import logging
from lingua import Language, LanguageDetectorBuilder

from helper.prompt_templates import prompt_template_translation,prompt_template_topic, prompt_template_topic_view

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize the language detector
detector = LanguageDetectorBuilder.from_languages(
    Language.ENGLISH, Language.SPANISH, Language.CHINESE, Language.GERMAN, Language.FRENCH
).build()

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

def translate_entry(entry, prompt_template, client, model):
    """
    Translates the reason and wish fields of an entry if needed.
    """
    reason_text = entry.get("Please tell us why you chose the rating above:", "")
    wish_text = entry.get("If you had a magic wand and you could change, add, or remove anything from the game, what would it be and why?", "")

    detected_language = detect_language(reason_text, wish_text)
    entry["language"] = detected_language

    if detected_language not in ["english", "none"]:
        logger.info(f"Translating entry ID {entry['ID']} (Language: {detected_language})")
        try:
            prompt = prompt_template.format(
                reason=reason_text or "N/A",
                wish=wish_text or "N/A"
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for translation."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024
            )
            translation_text = response.choices[0].message.content
            if "REASON:" in translation_text:
                entry["Please tell us why you chose the rating above:"] = translation_text.split("REASON:")[1].split("WISH:")[0].strip()
            if "WISH:" in translation_text:
                entry["If you had a magic wand and you could change, add, or remove anything from the game, what would it be and why?"] = translation_text.split("WISH:")[1].strip()
        except Exception as e:
            logger.error(f"Error translating entry ID {entry['ID']}: {e}")
            raise

def extract_topics(entry, prompt_template, client, model):
    """
    Extracts topics from an entry's combined review.
    """
    combined_review = f"{entry.get('Please tell us why you chose the rating above:', '')} {entry.get('If you had a magic wand and you could change, add, or remove anything from the game, what would it be and why?', '')}"
    prompt = prompt_template.format(review=combined_review)

    logger.info(f"Extracting topics for entry ID {entry['ID']}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for game review analysis."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error extracting topics for entry ID {entry['ID']}: {e}")
        raise

def analyze_sentiments(entry, topics, prompt_template, client, model):
    """
    Performs sentiment analysis on extracted topics.
    """
    entry["topics"] = []
    for topic in topics.get("Topics", []):
        logger.info(f"Analyzing sentiment for topic '{topic['Topic']}' (Entry ID {entry['ID']})")
        prompt = prompt_template.format(
            review=topic["Context"],
            topic=topic["Topic"]
        )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant expert in sentiment analysis."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024
            )
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

def process_entry(entry, id_counter, translation_template, topic_template, sentiment_template, client, model):
    """
    Processes a single entry: language detection, translation, topic extraction, and sentiment analysis.
    """
    entry["ID"] = id_counter
    logger.info(f"Processing entry ID {entry['ID']}")

    try:
        translate_entry(entry, translation_template, client, model)
        topics = extract_topics(entry, topic_template, client, model)
        analyze_sentiments(entry, topics, sentiment_template, client, model)
    except Exception as e:
        logger.error(f"Error processing entry ID {entry['ID']}: {e}")
        raise
