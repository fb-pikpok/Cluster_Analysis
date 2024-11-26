from datetime import datetime

# Example Unix timestamp
last_played = 1643202012

# Convert Unix timestamp to a datetime object
date = datetime.utcfromtimestamp(last_played)

# Format the datetime object to a readable string
formatted_date = date.strftime('%Y-%m-%d %H:%M:%S')

print(f"Last played date: {formatted_date}")
