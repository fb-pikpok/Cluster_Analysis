import requests
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from helper.utils import save_to_json


###########################################

def get_user_reviews(review_appid, params):
    user_review_url = f'https://store.steampowered.com/appreviews/{review_appid}'
    req_user_review = requests.get(
        user_review_url,
        params=params
    )

    if req_user_review.status_code != 200:
        print(f'Fail to get response. Status code: {req_user_review.status_code}')
        return {"success": 2}

    try:
        user_reviews = req_user_review.json()
    except:
        return {"success": 2}

    return user_reviews


def fetch_reviews_time_stamp(appid, params, start_time=None, end_time=None):
    """Fetches reviews with optional time filtering."""
    passed_start_time = not start_time  # If no start_time, assume it's passed
    passed_end_time = not end_time  # If no end_time, assume it's passed
    selected_reviews = []

    while not passed_start_time or not passed_end_time:
        reviews_response = get_user_reviews(appid, params)

        # Not successful response
        if reviews_response["success"] != 1:
            logger.error("Steam API returned not 1 as success. Unable to fetch reviews.")
            break

        if reviews_response["query_summary"]['num_reviews'] == 0:
            logger.info("No more reviews available.")
            break

        for review in reviews_response["reviews"]:
            timestamp_created = review['timestamp_created']

            # Time filtering logic
            if end_time and timestamp_created > end_time.timestamp():
                logger.info("Review out of date range")
                continue
            if start_time and timestamp_created < start_time.timestamp():
                logger.info("Review within RANGE")
                passed_start_time = True
                break

            selected_reviews.append({
                'recommendationid': review['recommendationid'],
                'author_steamid': review['author']['steamid'],
                'playtime_at_review_minutes': review['author']['playtime_at_review'],
                'playtime_forever_minutes': review['author']['playtime_forever'],
                'playtime_last_two_weeks_minutes': review['author']['playtime_last_two_weeks'],
                'last_played': review['author']['last_played'],
                'review_text': review['review'],
                'timestamp_created': timestamp_created,
                'timestamp_updated': review['timestamp_updated'],
                'voted_up': review['voted_up'],
                'votes_up': review['votes_up'],
                'votes_funny': review['votes_funny'],
                'weighted_vote_score': review['weighted_vote_score'],
                'steam_purchase': review['steam_purchase'],
                'received_for_free': review['received_for_free'],
                'written_during_early_access': review['written_during_early_access'],
            })

        # Update cursor to fetch next page
        cursor = reviews_response.get('cursor', '')
        if not cursor:
            logger.info(f"No more pages available. (last cursor reached {cursor})")
            break
        params['cursor'] = cursor

    return selected_reviews


if __name__ == "__main__":
    # https: // store.steampowered.com / app / 455690 / Pixel_Puzzles_Junior_Jigsaw /
    appid = '455690'

    params = {
            'json' : 1,
            'filter' : 'all',
            'language' : 'all',
            'day_range': 9223372036854775807,
            'start_date': 1514721661,
            'end_date': 1546257661,
            'review_type' : 'all',
            'purchase_type' : 'all'
            }

    # time interval
    end_time = datetime(2024, 1, 1, 0, 0, 0)
    start_time = datetime(2017, 1, 1, 0, 0, 0)

    # reviews = get_n_reviews(appid, 1000, params, start_time, end_time)
    # save_to_json(reviews, 'Pixel_Puzzles_Junior_Jigsaw_reviews.json')
    # print(len(reviews))

    reviews = fetch_reviews_time_stamp(appid, params, start_time, end_time)
    save_to_json(reviews, 'Pixel_Puzzles_Junior_Jigsaw_reviews.json')
    print(len(reviews))