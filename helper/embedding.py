from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
import logging
from helper.utils import api_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embedding = api_settings['client'].embeddings.create(input=[text], model=model).data[0].embedding
    return embedding

# Load offline model when necessary
# embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
#
#
def get_offline_embedding(text):
    text = text.encode(encoding="ASCII", errors="ignore").decode()
    return embed_model.get_text_embedding(text)


def process_embedding(data, embed_key):
    for i in range(0, len(data)):
        if i % 10 == 0:
            logger.info(f"Processing entry {i}")

        for d_topic in data[i]["topics"]:
            if isinstance(d_topic, dict):
                d_topic["embedding"] = get_embedding(d_topic[embed_key], model="text-embedding-3-small")
                # d_topic["embedding"] = get_offline_embedding(d_topic[embed_key])
    return data


# Flatten
def flatten_data(data):
    flattened = []
    for entry in data:
        base_copy = dict(entry)
        topics = base_copy.pop("topics", [])

        for topic in topics:
            new_entry = dict(base_copy)
            new_entry.update(topic)
            flattened.append(new_entry)
    return flattened
