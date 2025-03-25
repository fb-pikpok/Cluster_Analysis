<br />
<div align="center">


  <h3 align="center">Insights Tool</h3>

</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#the idea">The idea</a></li>
        <li><a href="#why-would-you-want-that">Why would you want that?</a></li>
        <li><a href="#how-it-works">How it works</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This projects is experimental. The idea is that that everyone who runs this project is able to analyse different sources
of natural language data and extract insights from it.


### The Idea
People are confronted with natural language text from users all the time. This text can be in the form of reviews, comments, emails, etc.
This tool allows you to effectively analyse this text and extract the topics and sentiments from it. 
The results will be presented in an interactive dashboard that allows you to explore the data in more detail.
Frequently addressed topics will be grouped together utilising cluster algorithms.


### Why would you want that?
* **Time efficiency:** Reading through 5000 steam reviews takes weeks - with this tool it takes ~3h
* **No oversights:** If a topic is important you will find it with this tool, compared when you read through the data manually where you have a high chance of missing it when the data set is large enough
* **Data driven:** You no llonger have to rely on a feeling what people are talking about, you can see and explore it
* **Cost efficiency:** Yes the OpenAI API costs money, but an for the 5000 steam reviews it costs ~$0.4 - Manual labour would cost way more


### How it works:
The main features contain seven steps:
1. **Extract relevant Data:** E.g. Specify the col where the review text is stored
2. **Translate non english responses** (Optional if needed)
3. **Topic extraction:** Utilise OpenAI API to extract topics from the text (The prompt template can be adjusted depending on the use case)
4. **Sentiment analysis:** Utilise OpenAI API to extract the sentiment from the topics that were extracted in the previous step
5. **Embedding:** Embedd the topics or the quote (you can choose between OpenAI or a local model)
6. **Clustering:** Cluster the topics together that are similar to each other
7. **Naming:** Use the OpenAI API to name the clusters (Optional but recommended)



### Things to note:
LLMs can make mistakes and they will. This project made the individual tasks for the LLM as simple as possible to reduce the chance of mistakes. 
However, sentences can be classified as the wrong sentiment or the topic extracted could be misleading. Prompts can introduce bias and lead to hallucinations.
Therefore, take the results with a grain of salt and fact check especially when the results are surprising to you.
That being said, the LLMs are very good at extracting topics and sentiments from text and with more data individual errors will be averaged out or be treated as outliers by the clustering.



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these steps:

### Installation


1. Request an API Key for [OpenAI API Key for OpenAI](https://platform.openai.com/api-keys)
2. Clone the repo
   ```sh
   git clone https://github.com/fb-pikpok/Cluster_Analysis.git
   ```
3. Install required packages & the community package of langchain
      ```sh
      pip install -r requirements.txt
      ```
      ```sh
      pip install -U langchain-community
      ```
4. Rename `.RENAMEenv` to `.env` and enter your API key (Optional: Redshift credentials)

5. Adjust your paths: This is important because API calls can take seveal hours and we need to save the progress every now and then. Additionally storing data in intermediate steps allows us to keep track of what we have already analysed and avoid processing the same thing twice.
      - `1.datasource_and_analyse.ipynb`
         - root_dir = select the main folder where data should be stored
         - project = type in the name of the project (this will create a folder in your root_dir where all data for the project will be stored)
      - `2.storage_and_query.ipynb`
         - raw_data = specify the path for the newly created `db_embedded.json` (this file is in the path you specified in the previous step)
         - persist_dir = We store the embedded data in a chroma database. To be able to use it we need to specify a path where the database should be stored
         - collection_name = ChromaDB's wording for the name of the database
      - `3.ai_summary.ipynb`
         - root_dir = same as in the first step
         - project = same as in the first step
         - input_path = where did you store the data that was created in step 2 using `st_chroma.py`? (best practice is to store it in the same folder and name it `db_final.json`)
6. Test connection to OpenAI API and the dashboard with streamlit
      ```sh
      python helper\\testcases.py
      ```
   ```sh
    streamlit run st_dashboard.py
    ```
   


<!-- Work Flow -->
## Work Flow & How to use the tool

Once you setup the paths and made sure the API key(s) are working you can start using the tool. The current work flow is not ideal. You will have to use different Jupyter Notebooks and store data in a Vector Database and sometimes as a JSON in your directory. This is because the tool will eventually be a automated pipeline that will run on a server and not on your local machine. However, for now this is the best way to use the tool: 

1. Navigate to `1.datasource_and_analyse.ipynb` and run the first cell. This will fire up the paths and the API client. 
   Next search for the cell that supports your data source and run it. The cell will ask you to specify the path to the data source 
   and the column where the text is stored. Once you run the cell the data will be stored in a JSON file in your specified directory. 
   This JSON file will be used in the next step. NOTE: This execution can take **several hours** depending on the size of the data source.

2. Navigate to `2.storage_and_query.ipynb` and run all cells consecutively. This will embed the data and store it in a Vector Database. 
   This step is done so we can perform a semantic search in our `st_chroma.py` file. This should be pretty fast.

3. Open `st_chroma.py` and adjust the paths and the collection name to the ones you are using in the previous step

4. In your terminal or a command prompt within the environment you are using run the following command:
   ```sh
   streamlit run st_chroma.py
   ```
   This will open a new tab in your browser where you can prepare the data for the dashboard.
   - Klick on **Run Query** to connect to the database you just created and retrive all the data. (You can also filter e.g. perform a semantic search)
   - Select the cluster parameters (some hints for good results are given in the dashboard, but experiment with it - nothing can go wrong :)
   - Klick on **Cluster Data**. This can take a couple of seconds when running for the first time. If you are not happy you can always rerun the clustering with new parameters. 

   Rule of thumb: **If the shapes of the data roughly match the way they are colored you are good!**

5. Click on **Name Clusters**. (The more clusters you have the longer this tep takes)
6. Click on **Download JSON** -> This will be the final output that you can use in the dashboard and for the AI summary. 

7. In a browser navigate to our [PikPok Dashboard](https://topicsentimentextractiontool.streamlit.app/) there you can upload the newly created JSON file and explore the data
8. Alternative or in Addition: Navigate to `3.ai_summary.ipynb` and run the cells. This will give you a report of the data you just created. 
   It will contain summaries of individual clusters, the biggest issues and the most important topics.


## Roadmap

- [x] Add Steamreviews support
- [x] Add Survey Data
- [x] Works with transcripts
- [x] Combine multiple sources
- [ ] Add Facebook Data

