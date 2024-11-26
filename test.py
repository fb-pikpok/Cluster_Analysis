from datetime import datetime, timedelta
import requests
import json
from pathlib import Path


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


# https://store.steampowered.com/app/271590/Grand_Theft_Auto_V/
review_appname = "RISK"  # the game name
review_appid = 1128810  # the game appid on Steam

# the params of the API
params = {
    'json': 1,
    'language': 'english',
    'cursor': '*',  # set the cursor to retrieve reviews from a specific "page"
    'num_per_page': 100,
    'filter': 'recent'
}

# time interval
end_time = datetime(2022, 2, 1, 0, 0, 0)
start_time = datetime(2022, 1, 25, 0, 0, 0)

print(f"Start time: {start_time}")
print(f"End time: {end_time}")
print(start_time.timestamp(), end_time.timestamp())

passed_start_time = False
passed_end_time = False

selected_reviews = []

while (not passed_start_time or not passed_end_time):

    reviews_response = get_user_reviews(review_appid, params)

    # not success?
    if reviews_response["success"] != 1:
        print("Not a success")
        print(reviews_response)
        break

    if reviews_response["query_summary"]['num_reviews'] == 0:
        print("No reviews.")
        break

    for review in reviews_response["reviews"]:
        recommendation_id = review['recommendationid']

        timestamp_created = review['timestamp_created']
        timestamp_updated = review['timestamp_updated']

        # skip the comments that are beyond end_time
        if not passed_end_time:
            if timestamp_created > end_time.timestamp():
                continue
            else:
                passed_end_time = True

        # exit the loop once a comment before start_time is detected
        if not passed_start_time:
            if timestamp_created < start_time.timestamp():
                passed_start_time = True
                break

        # extract the useful data
        my_review_dict = {
            'recommendationid': review['recommendationid'],
            'author_steamid': review['author']['steamid'],
            'playtime_at_review_minutes': review['author']['playtime_at_review'],
            'playtime_forever_minutes': review['author']['playtime_forever'],
            'playtime_last_two_weeks_minutes': review['author']['playtime_last_two_weeks'],
            'last_played': review['author']['last_played'],

            'language': review['language'],
            'review_text': review['review'],
            'timestamp_created': timestamp_created,
            'timestamp_updated': timestamp_updated,

            'voted_up': review['voted_up'],
            'votes_up': review['votes_up'],
            'votes_funny': review['votes_funny'],
            'weighted_vote_score': review['weighted_vote_score'],
            'steam_purchase': review['steam_purchase'],
            'received_for_free': review['received_for_free'],
            'written_during_early_access': review['written_during_early_access'],
        }

        selected_reviews.append(my_review_dict)

    # go to next page
    try:
        cursor = reviews_response['cursor']  # cursor field does not exist in the last page
    except Exception as e:
        cursor = ''

    # no next page
    # exit the loop
    if not cursor:
        print("Reached the end of all comments.")
        break

    # set the cursor object to move to the next page to continue
    params['cursor'] = cursor
    print('To next page. Next page cursor:', cursor)

# Save the selected reviews to a JSON file
foldername = f"{review_appid}_{review_appname}"
filename = f"{review_appid}_{review_appname}_reviews_{start_time.strftime('%Y%m%d-%H%M%S')}_{end_time.strftime('%Y%m%d-%H%M%S')}.json"
output_path = Path(foldername, filename)

if not output_path.parent.exists():
    output_path.parent.mkdir(parents=True)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(selected_reviews, f, indent=2, ensure_ascii=False)

print(f"Reviews saved to {output_path}")
