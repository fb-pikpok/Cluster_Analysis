import argparse
import http.client
import json
from typing import Optional
import urllib.parse


def fetch(appid: int, cursor: Optional[str] = None) -> dict:
    all_reviews = []
    previous_cursor = None
    iterations = 0
    total_reviews = 0

    while True:
        iterations += 1
        conn = http.client.HTTPSConnection('store.steampowered.com')
        url = f'/appreviews/{appid}?json=1&filter=recent&num_per_page=100'


        if cursor:
            encoded_cursor = urllib.parse.quote(cursor)
            url += f'&cursor={encoded_cursor}'

        conn.request('GET', url)
        response = conn.getresponse()
        data = response.read()
        conn.close()

        response_data = json.loads(data)

        if response_data.get('success') != 1:
            break

        all_reviews.extend(response_data.get('reviews', []))

        if iterations == 1:
            total_reviews = response_data.get('query_summary', {}).get('total_reviews', 0)

        previous_cursor = cursor
        cursor = response_data.get('cursor')

        if not cursor or cursor == previous_cursor:
            break

        print(f"Total Reviews: {total_reviews}, Fetched Reviews: {len(all_reviews)}")

    return all_reviews


def main(appid: Optional[int] = None):
    parser = argparse.ArgumentParser(description='Get Steam Reviews by App ID')
    parser.add_argument('--appid', type=int, help='Steam App ID', default=appid)

    args = parser.parse_args()

    # Fetch the first page
    if args.appid:
        data = fetch(args.appid)
        with open(f"{args.appid}.json", 'w') as f:
            json.dump(data, f, indent=2)
            print(f"Reviews saved to {args.appid}.json")


if __name__ == '__main__':
    main(appid=455690)
