import requests
import logging

from helper.utils import save_to_json

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from datetime import datetime


def get_user_reviews(appid, params):
    """Fetches reviews for a given app ID with specified parameters."""
    user_review_url = f'https://store.steampowered.com/appreviews/{appid}'
    req_user_review = requests.get(user_review_url, params=params)

    if req_user_review.status_code != 200:
        logger.error(f"Failed to get response. Status code: {req_user_review.status_code}")
        return {"success": 2}

    try:
        user_reviews = req_user_review.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
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

def fetch_reviews(appid, params):
    """Fetches reviews with optional time filtering."""

    selected_reviews = []

    while (True):
        reviews_response = get_user_reviews(appid, params)

        # not success?
        if reviews_response["success"] != 1:
            print("Not a success")
            break

        if reviews_response["query_summary"]["num_reviews"] == 0:
            print("no_reviews.")
            break

    # extract each review in the response of the API call
        for review in reviews_response["reviews"]:
            # for brevity, the extraction is not included
            selected_reviews.append({
                'recommendationid': review['recommendationid'],
                'author_steamid': review['author']['steamid'],
                'playtime_at_review_minutes': review['author']['playtime_at_review'],
                'playtime_forever_minutes': review['author']['playtime_forever'],
                'playtime_last_two_weeks_minutes': review['author']['playtime_last_two_weeks'],
                'last_played': review['author']['last_played'],
                'review_text': review['review'],
                'timestamp_created': review['timestamp_created'],
                'timestamp_updated': review['timestamp_updated'],
                'voted_up': review['voted_up'],
                'votes_up': review['votes_up'],
                'votes_funny': review['votes_funny'],
                'weighted_vote_score': review['weighted_vote_score'],
                'steam_purchase': review['steam_purchase'],
                'received_for_free': review['received_for_free'],
                'written_during_early_access': review['written_during_early_access'],
            })

            # go to next page
        try:
            cursor = reviews_response['cursor']  # cursor field does not exist, or = null in the last page
        except Exception as e:
            cursor = ''

        if not cursor:
            print("Reached the end of all comments.")
            break

    params["cursor"] = cursor
    print("To next page. Next page cursor Cursor:", cursor)



if __name__ == "__main__":
    # Example Usage

    # Game details
    appname = "RISK"  # the game name
    appid = 1128810  # the game appid on Steam

    # the params of the API
    params = {
        'json': 1,
        'language': 'all',
        'cursor': '*',  # set the cursor to retrieve reviews from a specific "page"
        'num_per_page': 100,
        'filter': 'all',  # filter the reviews by 'recent' or 'all'
    }

    # time interval
    end_time = datetime(2024, 11, 1, 0, 0, 0)
    start_time = datetime(2024, 9, 1, 0, 0, 0)

    # Example 1: Normal call (no time filtering)
    # print("Fetching all reviews...")
    # reviews = fetch_reviews(appid, params)
    # print(f"Fetched {len(reviews)} reviews in total.")

    # Example 2: Time-filtered call
    time_filtered_reviews = fetch_reviews_time_stamp(appid, params, start_time, end_time)
    print(f"Fetched {len(time_filtered_reviews)} reviews in the specified time range.")
    save_to_json(time_filtered_reviews, f"{appname}_reviews.json")


# What does the parameter 'recent' do?