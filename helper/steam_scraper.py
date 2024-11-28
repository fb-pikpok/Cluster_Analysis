import requests
import logging

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
