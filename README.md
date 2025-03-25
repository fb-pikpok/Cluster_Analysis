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

[![Product Name Screen Shot][product-screenshot]](https://example.com)

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

5. Adjust your paths in the Notebook you are using e.g. `HRC.ipynb`
6. Test connection to OpenAI API and the dashboard with streamlit
      ```sh
      python test_openai.py
      ```
   ```sh
    streamlit run app.py
    ```


<!-- ROADMAP -->
## Roadmap

- [x] Add Steamreviews support
- [x] Add Survey Data
- [x] Works with transcripts
- [x] Combine multiple sources
- [ ] Add Facebook Data

