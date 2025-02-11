import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import json
import pandas as pd

# Suppose these come from your existing code
# from helper.utils import configure_api, read_json, save_to_json, logger
# from helper.prompt_templates import prompt_template_topic, prompt_template_sentiment
# from helper.data_analysis import process_entry

# General modules
import os
import openai
import pandas as pd
from dotenv import load_dotenv
from helper.utils import *

# Setup API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key
client = openai.Client()

# Specify models
chat_model_name = 'gpt-4o-mini'
openai_embedding_model = "text-embedding-3-small"
local_embedding_model = "all-MiniLM-L6-v2"

configure_api(client, chat_model_name)

# Specify paths for storing (backup) data
root_dir = r'S:\SID\Analytics\Working Files\Individual\Florian\Projects\DataScience\cluster_analysis\Data\RivalStars'
data_source = 'Steam'

path_db_prepared = os.path.join(root_dir, data_source, "db_prepared.json")          #backup
path_db_translated = os.path.join(root_dir, data_source, "db_translated.json")      #backup
path_db_analysed = os.path.join(root_dir, data_source, "db_analysed.json")          #backup
path_db_embedded = os.path.join(root_dir, data_source, "db_embedded.json")          #backup
path_db_clustered = os.path.join(root_dir, data_source, "db_clustered.json")        #backup
path_db_final = os.path.join(root_dir, data_source, "db_final.json")                #final file


MAX_CONCURRENT = 25  # You can tune this based on your rate limits

import os
import json
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor

# === Your existing imports ===
from helper.utils import (
    configure_api, read_json, save_to_json,
    logger  # assume you have a logger in place
)
from helper.prompt_templates import (
    prompt_template_topic, prompt_template_sentiment
)
from helper.data_analysis import (
    process_entry  # The synchronous function from your code
)



# You already define these globally in your code,
# but let's confirm them here for clarity:
id_column = "recommendationid"
columns_of_interest = ["review_text"]



# How many requests you want to process in parallel
MAX_CONCURRENCY = 5  # Tweak based on rate limits

async def process_entry_async(
    entry,
    id_col,
    prompt_topic,
    prompt_sentiment,
    api_cfg,
    cols_of_interest
):
    """
    Runs your existing synchronous 'process_entry' in a thread so it's non-blocking.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,  # uses default ThreadPoolExecutor
        process_entry,
        entry,
        id_col,
        prompt_topic,
        prompt_sentiment,
        api_cfg,
        cols_of_interest
    )

async def worker(i, entry, semaphore):
    """
    Single 'worker' task that:
    1) Acquires a semaphore so we don't exceed concurrency.
    2) Calls 'process_entry_async' in a thread.
    3) Returns (index, processed_entry).
    """
    async with semaphore:
        processed_entry = await process_entry_async(
            entry,
            id_column,
            prompt_template_topic,
            prompt_template_sentiment,
            api_settings,
            columns_of_interest
        )
        return i, processed_entry

async def main():
    # 1. Configure API (your existing code)
    configure_api(client, chat_model_name)

    # 2. Load data
    data = pd.read_pickle(path_db_translated)
    data_prepared = data.to_dict(orient='records')

    # 3. Load already processed entries (if any)
    all_entries = []
    processed_ids = set()
    if os.path.exists(path_db_analysed):
        all_entries = read_json(path_db_analysed)
        processed_ids = {entry[id_column] for entry in all_entries}

    # 4. Create tasks for unprocessed entries
    tasks = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    for i, entry in enumerate(data_prepared):
        current_id = entry[id_column]

        if current_id in processed_ids:
            logger.info(f"Skipping entry {i} (ID: {current_id}) - already processed.")
            continue

        # schedule a worker task
        tasks.append(asyncio.create_task(worker(i, entry, semaphore)))

    # 5. Process tasks as they complete, track partial progress
    completed_count = 0
    total_tasks = len(tasks)
    for task in asyncio.as_completed(tasks):
        i, processed_entry = await task
        all_entries.append(processed_entry)

        completed_count += 1
        if completed_count % 10 == 0:
            # save intermediate results
            save_to_json(all_entries, path_db_analysed)
            logger.info(
                f"Progress saved at index {i}. "
                f"Completed {completed_count}/{total_tasks}."
            )

    # 6. Final save
    save_to_json(all_entries, path_db_analysed)
    logger.info("All entries processed and final results saved.")

if __name__ == "__main__":
    # Run everything asynchronously
    asyncio.run(main())

