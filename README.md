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

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

Run the requirements.txt file to install the necessary packages
*
  ```sh
  pip install -r requirements.txt
  ```

Install the Community package of langchain
*
    ```sh
    pip install -U langchain-community
    ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

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


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Steamreviews support
- [x] Add Survey Data
- [x] Works with transcripts
- [ ] Add Facebook Data
- [ ] Combine multiple sources




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
