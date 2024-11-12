import numpy as np
import pandas as pd
import json
import os
os.environ["OMP_NUM_THREADS"] = '1'


from langchain.prompts import PromptTemplate
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from umap import umap_ as UMAP
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


from dotenv import load_dotenv
load_dotenv()

import os
import openai

openai_api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = openai_api_key
client = openai.Client()

chat_model_name = 'gpt-3.5-turbo'
embedding_model_name = 'sentence-transformers/all-mpnet-base-v2'

n_pick = 150  # the number of reviews picked for each game
s_root = r'C:\Users\fbohm\Desktop\Projects\DataScience\cluster_analysis/'
s_db_json = 'review_db.json'
s_db_embed_json = 'review_db_embed.json'
s_db_table_json = 'review_db_table.json'
s_db_table_xlsx = 'review_db_table.xlsx'
s_db_table_pca_json = 'review_db_table_pca.json'
s_db_table_pca_xlsx = 'review_db_table_pca.xlsx'
s_kmeans_centers = 'kmeans_centers.json'
b_override = False



prompt_template_topic = PromptTemplate.from_template(
'''Please list the most important topics and their respective original context in the review of a game in a json format with "Topic", "Category", "Context" arguments.  No more than 10 topics.
Topics should be game features.  A feature in the game should be a noun rather than a verb or an adjective.
Each topic should be categorized as a "fact" or a "request".

[h0]==================================================================[\h0]
REVIEW: 

"The weapon durability in this game is frustrating; my sword breaks after just a few swings. The combat itself is fun, but I wish the durability lasted longer. Also, the audio effects are very immersive during battles."

TOPICS:

[
    {{
        "Topic": "Weapon Durability",
        "Category": "request",
        "Context": "My sword breaks after just a few swings. I wish the durability lasted longer."
    }},
    {{
        "Topic": "Combat and Fighting",
        "Category": "fact",
        "Context": "The combat itself is fun."
    }},
    {{
        "Topic": "Audio",
        "Category": "fact",
        "Context": "The audio effects are very immersive during battles."
    }}
]

[h0]==================================================================[\h0]
REVIEW: 

"Playing during the night adds a thrilling layer to the game. The lack of a proper save feature makes it hard to enjoy it though. Also, there are way too many random encounters that make progress difficult."

TOPICS:

[
    {{
        "Topic": "Night",
        "Category": "fact",
        "Context": "Playing during the night adds a thrilling layer to the game."
    }},
    {{
        "Topic": "Save Feature",
        "Category": "request",
        "Context": "The lack of a proper save feature makes it hard to enjoy fully."
    }},
    {{
        "Topic": "Randomness",
        "Category": "request",
        "Context": "There are way too many random encounters that make progress difficult."
    }}
]

[h0]==================================================================[\h0]
REVIEW: 

"{review}"

TOPICS:

'''
)

with open(s_root + 'Data/survey_results_trans.json', 'r') as f:
    db = json.load(f)

entry = db[0]

# Extract important information from the 2nd and 3rd keys
review_text = entry["Please tell us why you chose the rating above:"]
additional_feedback = entry["If you had a magic wand and you could change, add, or remove anything from the game, what would it be and why?"]

# Combine both into a single review input for the prompt
combined_review = f"{review_text} {additional_feedback}"

# Format the prompt for the LLM
prompt_topic = prompt_template_topic.format(review=combined_review)

# Make the OpenAI API call
response = client.chat.completions.create(
    model=chat_model_name,
    response_format={ "type": "json_object" },
    messages=[
        {"role": "system", "content": "You are a helpful assistant expertised in game review analysis. Respond in JSON format."},
        {"role": "user", "content": prompt_topic},
    ],
    max_tokens=1024,
)

# Print the response content
print(response.choices[0].message.content)