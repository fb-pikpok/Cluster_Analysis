import re
from lytikit.celery_app import celery
from lytikit.database import CeleryDatabaseTask
from datetime import datetime
import requests
import pandas as pd

from lytikit.database.redshift_connection import redshift_connect
from lytikit.helpers.Product import get_iap_sku_from_revenue_name


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
    if len(words) < 3 or len(review.encode('utf-8')) > 16000:
        return False

    # Count the number of alphabetic characters in the review
    alphabetic_count = sum(1 for char in review if char.isalpha())

    # Calculate the proportion of alphabetic characters
    proportion_alphabetic = alphabetic_count / len(review)

    # Consider the review informative if more than 50% of characters are alphabetic
    return proportion_alphabetic > 0.5


def get_app_id_and_name_from_url(url):
    """
    Get app id from steam url.
    :param str url:
    :return int:
    """
    try:
        app_id = url.split('/')[-3]
        app_name = url.split('/')[-2]
    except Exception as e:
        print(f"Error: {e}, URL most be like https://store.steampowered.com/app/546560/HalfLife_Alyx/")
        return None
    return app_id, app_name


# https://partner.steamgames.com/doc/store/getreviews
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


def save_reviews_to_database(selected_reviews):
    entries = [
        (
            review['app_id_name'], review['recommendationid'], review['playtime_at_review_minutes'],
            review['last_played'], review['review_text'], review['timestamp_updated'],
            review['voted_up'], review['votes_up'], review['votes_funny'],
            review['weighted_vote_score'], review['steam_purchase'], review['received_for_free'],
            review['written_during_early_access']
        )
        for review in selected_reviews
    ]

    # we save it in redshift
    sql = """
    INSERT INTO
    steam_review( app_id_name, recommendationid, playtime_at_review_minutes, last_played, review_text, timestamp_updated,
    voted_up, votes_up, votes_funny, weighted_vote_score, steam_purchase, received_for_free, written_during_early_access)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    with redshift_connect() as conn, conn.cursor() as cursor:
        cursor.executemany(sql, entries)

    print(f"Saved {len(entries)} reviews to database.")


def get_existing_recommendationids(review_appid, app_name):
    # get the reviews already in the database
    with redshift_connect() as conn:
        sql = f"""
        SELECT DISTINCT recommendationid
        FROM steam_review
        WHERE app_id_name = '{review_appid}_{app_name}'
        """
        r_cursor = conn.cursor()
        r_cursor.execute(sql)
        existing_recommendationids = r_cursor.fetchall() if r_cursor.rowcount > 0 else []
        existing_recommendationids = [r[0] for r in existing_recommendationids]
    return existing_recommendationids


class SteamScraper():
    """
    Scrapes Steam for reviews.
    """

    @celery.task(
        name='{name}.Format'.format(name='SteamScraper'),
        base=CeleryDatabaseTask,
        bind=True
    )
    def format(self, urls):

        # get the app id from the url
        for url in urls:
            review_appid, app_name = get_app_id_and_name_from_url(url)

            # get the reviews already in the database
            existing_recommendationids = get_existing_recommendationids(review_appid, app_name)

            # the params of the API
            params = {
                'json': 1,
                'language': 'all',
                'cursor': '*',  # set the cursor to retrieve reviews from a specific "page"
                'num_per_page': 100,
                'filter': 'updated'  # sort by updated, not filter out any reviews
            }

            print(f"App ID: {review_appid} - App Name: {app_name}")

            next_page = True
            selected_reviews = []
            seen_cursors = set()
            seen_cursors_count = count = 0

            # loop through all the comments until the end
            while next_page:
                count += 1
                reviews_response = get_user_reviews(review_appid, params)

                # not success?
                if reviews_response["success"] != 1:
                    print("Not a success")
                    print(reviews_response)

                for review in reviews_response["reviews"]:
                    recommendation_id = int(review['recommendationid'])

                    # skip the comments that are already in the database
                    if recommendation_id in existing_recommendationids:
                        continue

                    timestamp_updated = review['timestamp_updated']

                    playtime_at_review_minutes = review['author']['playtime_at_review']
                    last_played = review['author']['last_played']

                    review_text = review['review']

                    # skip useless comments
                    if not is_informative_review(review_text):
                        continue

                    voted_up = review['voted_up']
                    votes_up = review['votes_up']
                    votes_funny = review['votes_funny']
                    weighted_vote_score = review['weighted_vote_score']
                    steam_purchase = review['steam_purchase']
                    received_for_free = review['received_for_free']
                    written_during_early_access = review['written_during_early_access']

                    my_review_dict = {
                        'app_id_name': f"{review_appid}_{app_name}",
                        'recommendationid': recommendation_id,
                        'playtime_at_review_minutes': playtime_at_review_minutes,
                        'last_played': last_played,
                        'review_text': review_text,
                        'timestamp_updated': timestamp_updated,
                        'voted_up': voted_up,
                        'votes_up': votes_up,
                        'votes_funny': votes_funny,
                        'weighted_vote_score': weighted_vote_score,
                        'steam_purchase': steam_purchase,
                        'received_for_free': received_for_free,
                        'written_during_early_access': written_during_early_access,
                    }

                    selected_reviews.append(my_review_dict)

                # go to next page
                try:
                    cursor = reviews_response['cursor']  # cursor field does not exist in the last page
                    params['cursor'] = cursor
                    print('To next page. Next page cursor:', cursor)
                    if cursor in seen_cursors:
                        seen_cursors_count += 1
                    else:
                        seen_cursors.add(cursor)

                except Exception as e:
                    print(f"Reached the end of all comments of {app_name}.")
                    next_page = False

                # save every 5 iterations
                if count % 5 == 0 or not next_page:
                    save_reviews_to_database(selected_reviews)
                    if len(selected_reviews) > 0:
                        existing_recommendationids = get_existing_recommendationids(review_appid, app_name)
                    selected_reviews = []

                # break if we reach the end of the comments
                if not next_page or seen_cursors_count > 10:
                    next_page = False


if __name__ == '__main__':
    urls = [
        # 'https://store.steampowered.com/app/934700/Dead_Island_2/',
        # 'https://store.steampowered.com/app/2527390/Dead_Rising_Deluxe_Remaster/',
        # 'https://store.steampowered.com/app/1519090/Welcome_to_ParadiZe/',
        # 'https://store.steampowered.com/app/2093920/Dread_Dawn/',
        # 'https://store.steampowered.com/app/2897700/Arizona_Sunshine_Remake/',
        # 'https://store.steampowered.com/app/1881700/Undawn/',
        # 'https://store.steampowered.com/app/1963370/No_One_Survived/',
        # 'https://store.steampowered.com/app/2163330/Yet_Another_Zombie_Survivors/',
        # 'https://store.steampowered.com/app/1766060/HumanitZ/',
        # 'https://store.steampowered.com/app/2439510/Doomsday_Last_Survivors/',
        'https://store.steampowered.com/app/2351560/Apocalypse_Party/',
        'https://store.steampowered.com/app/1947500/The_Walking_Dead_Saints__Sinners__Chapter_2_Retribution/'
        'https://store.steampowered.com/app/1641960/Forever_Skies/',
        'https://store.steampowered.com/app/2244130/Ratopia/',
        'https://store.steampowered.com/app/1326470/Sons_Of_The_Forest/',
        'https://store.steampowered.com/app/2080690/Sunkenland/',
        'https://store.steampowered.com/app/2139460/Once_Human/',
        'https://store.steampowered.com/app/1458140/Pacific_Drive/',
        'https://store.steampowered.com/app/1203620/Enshrouded/',
        'https://store.steampowered.com/app/1928980/Nightingale/',
        'https://store.steampowered.com/app/2739590/Mad_Island/'  # TODO are you sure about this??? ----> NSFW
    ]

    start_time = datetime(2024, 12, 1, 0, 0, 0)
    end_time = datetime.now()

    SteamScraper().format(urls)
    print('Done')
