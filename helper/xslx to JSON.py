import pandas as pd
import json

# Load the Excel file
excel_file_path = 'C:\\Users\\fbohm\\Desktop\\Projects\\DataScience\\cluster_analysis\\Data\\Into the Dead Our Darkest Days_1511resposnes.xlsx'
dataframe = pd.read_excel(excel_file_path)

# Convert the DataFrame to a list of dictionaries (one dictionary per row)
data_as_dict = dataframe.to_dict(orient='records')

# Define the output JSON file path
json_file_path = 'C:\\Users\\fbohm\\Desktop\\Projects\\DataScience\\cluster_analysis\\Data\\survey_results.json'

# Write the JSON data to a file
with open(json_file_path, 'w') as json_file:
    json.dump(data_as_dict, json_file, indent=4)

print(f"Data successfully converted to JSON and saved to {json_file_path}")
