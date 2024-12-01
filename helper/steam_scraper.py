import requests
import logging
import re
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_reviews(appid, params={'json': 1}):
    url = 'https://store.steampowered.com/appreviews/'
    response = requests.get(url=url + appid, params=params, headers={'User-Agent': 'Mozilla/5.0'})
    return response.json()


def get_n_reviews(appid, params, n=100):

    reviews = []
    cursor = '*'
    total_retrieved = 0  # Counter for total reviews retrieved

    while n > 0:
        params['cursor'] = cursor.encode()
        params['num_per_page'] = min(100, n)

        response = get_reviews(appid, params)
        retrieved_now = len(response['reviews'])
        total_retrieved += retrieved_now
        logger.info(f"Retrieved {retrieved_now} reviews in API call. Total so far: {total_retrieved}")
        reviews += response['reviews']

        if cursor == response['cursor']:
            logger.info("No more reviews available.")
            break

        cursor = response['cursor']
        n -= retrieved_now

        if retrieved_now < 100:  # Stop if fewer reviews are retrieved than requested per page
            logger.info("Finished fetching all reviews.")
            break

    return reviews


def is_informative_review(review):
    """
    Determine if a review is informative based on word count and character content.

    Args:
        review (str): The review text.

    Returns:
        bool: True if the review is informative, False otherwise.
    """
    # Remove leading and trailing whitespace
    review = review.strip()

    # Split the review into words using Unicode word boundaries
    words = re.findall(r'\b\w+\b', review, re.UNICODE)

    # Check if the review has fewer than 3 words
    if len(words) < 3:
        return False

    # Count the number of alphabetic characters in the review
    alphabetic_count = sum(1 for char in review if char.isalpha())

    # Calculate the proportion of alphabetic characters
    proportion_alphabetic = alphabetic_count / len(review)

    # Consider the review informative if more than 50% of characters are alphabetic
    return proportion_alphabetic > 0.5

def filter_reviews(data, review_key):
    """
    Filter out uninformative reviews from the data and count the number of removed entries.

    Args:
        data (list): List of dictionaries containing reviews.
        review_key (str): The key in each dictionary where the review text is stored.

    Returns:
        tuple: A tuple containing the filtered list with only informative reviews and the count of removed entries.
    """
    original_count = len(data)
    filtered_data = [entry for entry in data if is_informative_review(entry.get(review_key, ''))]
    removed_count = original_count - len(filtered_data)
    logger.info(f"Total entries removed: {removed_count}")
    return filtered_data, removed_count

if __name__ == '__main__':
    from helper.utils import *

    # https: // store.steampowered.com / app / 455690 / Pixel_Puzzles_Junior_Jigsaw /
    appid = '455690'

    params = {
        'json': 1,
        'filter': 'all',
        'language': 'all',
        'day_range': 9223372036854775807,
        'review_type': 'all',
        'purchase_type': 'all'
    }


    reviews = get_n_reviews(appid, params, 200)
    save_to_json(reviews, f'{appid}_reviews.json')
    print(f"Total reviews: {len(reviews)}")
