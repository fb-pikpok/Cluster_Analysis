from langchain.prompts import PromptTemplate


""" 
Adjust the template below to match your task.

Note: If you change the structure not only the semantics, you may need to adjust the code in the main script as well.

"""


prompt_template_translation = PromptTemplate.from_template(
'''You are a professional translator. Translate the following text into English if it is not already in English.

[h0]==================================================================[\h0]
TEXT: "兄弟们，我把星空退款的钱拿来买这个了，我做的对吗"

TRANSLATION: "Brothers, I used the refund money from the stars to buy this. Did I do the right thing?"

[h0]==================================================================[\h0]
TEXT: "Me toma demasiado tiempo entrenar caballos."

TRANSLATION: "It takes too much time to train horses."

[h0]==================================================================[\h0]
TEXT: "Grinding just to get good tack, grain etc. Itâ€™s very time consuming sadly"

TRANSLATION: "Grinding just to get good tack, grain etc. It's very time consuming sadly"

[h0]==================================================================[\h0]
TEXT: "{text}"

TRANSLATION:
'''
)

prompt_template_topic = PromptTemplate.from_template(
'''Please list the most important topics and their respective original context in the review of a game in a JSON format with "Topic", "Category", and "Context" arguments. No more than 10 topics.
Topics should focus on specific game features or aspects. A feature in the game should be a noun rather than a verb or an adjective.
Each topic should be categorized as a "fact" or a "request".
Respond in JSON format.

[h0]==================================================================[\h0]
REVIEW: 

"The customization options for characters are so limited, and it's frustrating not to have more outfit choices. Also, why can't I rename my horse after I buy it? However, I do enjoy the free roam mode—riding through open fields feels relaxing and immersive."

TOPICS:

{{"Topics":
    [
        {{
            "Topic": "Character Customization",
            "Category": "request",
            "Context": "The customization options for characters are so limited, and it's frustrating not to have more outfit choices."
        }},
        {{
            "Topic": "Horse Renaming",
            "Category": "request",
            "Context": "It's frustrating not to be able to rename my horse after I buy it."
        }},
        {{
            "Topic": "Free Roam",
            "Category": "fact",
            "Context": "Riding through open fields feels relaxing and immersive."
        }}
    ]
}}

[h0]==================================================================[\h0]
REVIEW: 

"Too much useless nonsense."

TOPICS:

{{"Topics":
    [
        {{"Topic": "Game Content",
          "Category": "request",
          "Context": "Too much useless nonsense."
        }}
    ]
}}

[h0]==================================================================[\h0]
REVIEW: 

"This game has great mechanics, but the breeding system feels random and unfair. I've bred so many horses, yet the coats and stats don't seem to follow any logical pattern. On the other hand, I appreciate how detailed the horse animations are—it makes the game come alive."

TOPICS:

{{"Topics":
    [
        {{
            "Topic": "Game Mechanics",
            "Category": "fact",
            "Context": "This game has great mechanics"
        }},
        {{
            "Topic": "Breeding System",
            "Category": "request",
            "Context": "The breeding system feels random and unfair. Coats and stats don't seem to follow any logical pattern."
        }},
        {{
            "Topic": "Horse Animations",
            "Category": "fact",
            "Context": "The horse animations are detailed and make the game come alive."
        }}
    ]
}}

[h0]==================================================================[\h0]
REVIEW: 

"{review}"

TOPICS:

'''
)

prompt_template_sentiment = PromptTemplate.from_template(
'''What's the sentiment of the review with regard to the topic?
Always answer with 'Positive' or 'Negative' or 'Inconclusive'.

REVIEW: My first D&D experience and I'm enjoying it a lot.
TOPIC: D&D
SENTIMENT: Positive 

REVIEW: This game lacks a proper ending or epilog
TOPIC: epilogue
SENTIMENT: Negative

REVIEW: Posted: August 8
TOPIC: release date
SENTIMENT: Inconclusive 

REVIEW: {review}
TOPIC: {topic}
SENTIMENT: '''
)

prompt_template_cluster_naming = (
'''Based on the following topics, generate a concise name (5 words or fewer) that best describes the general theme of this cluster.

TOPICS: {topics}
CLUSTER NAME: '''
)