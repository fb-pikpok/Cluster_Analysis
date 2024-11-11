import json
import re
import os

# Adjust paths relative to the current script location
input_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'survey_results.json')
output_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'survey_results_clean.json')


# Function to detect non-parsable Unicode characters
def is_valid_string(s):
    # Regex to match only ASCII and common English characters
    return bool(re.match(r'^[\u0000-\u007F]*$', s))

# Function to check if a JSON object is clean (all values are valid)
def is_clean_entry(entry):
    for key, value in entry.items():
        # Check if the value is a string and contains non-parsable characters
        if isinstance(value, str) and not is_valid_string(value):
            return False
    return True

# Read the JSON data
with open(input_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Filter out entries with non-parsable strings
initial_count = len(data)
clean_data = [entry for entry in data if is_clean_entry(entry)]
removed_count = initial_count - len(clean_data)

# Save the cleaned data to a new JSON file
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(clean_data, json_file, indent=4)

print(f"Cleaned data has been saved to {output_file_path}")
print(f"Entries removed: {removed_count}")
