import json
import random
import os

def create_sample_json(input_file_path, output_file_path, sample_size=600):
    # Read the cleaned JSON data
    with open(input_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Ensure the sample size does not exceed the total number of entries
    sample_size = min(sample_size, len(data))

    # Randomly sample the specified number of entries
    sampled_data = random.sample(data, sample_size)

    # Save the sampled data to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(sampled_data, json_file, indent=4)

    print(f"Sample of {sample_size} entries has been saved to {output_file_path}")

# Paths for the input and output files
input_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'survey_results_clean.json')
output_file_path = os.path.join(os.path.dirname(__file__), '..', 'Data', 'sample_size.json')

# Call the function to create the sample JSON
create_sample_json(input_file_path, output_file_path)
