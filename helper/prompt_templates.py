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
'''
Please analyze the review and extract the most important topics relevant to frustrations in the game in a JSON format with "Topic", "Category", and "Context" arguments. Avoid repeating the central feature "Show Jumping" in every topic unless it is strictly necessary. 
Focus on specific aspects or features that players find frustrating. No more than 10 topics.

- "Topic" should focus on a specific game feature or aspect.
- "Category" should be either "fact" or "request".
- "Context" should be a direct excerpt from the review that explains the topic.

Respond in JSON format.

[h0]==================================================================[\h0]
REVIEW: 

"The camera during show jumping feels awkward—it’s hard to follow the horse smoothly. Also, the hitboxes for fences seem inconsistent. Sometimes, it feels like the horse clears a fence but still gets penalized."

TOPICS:

{{
    "Topics": [
        {{
            "Topic": "Camera Movement",
            "Category": "request",
            "Context": "The camera during show jumping feels awkward—it’s hard to follow the horse smoothly."
        }},
        {{
            "Topic": "Fence Hitboxes",
            "Category": "request",
            "Context": "The hitboxes for fences seem inconsistent. Sometimes, it feels like the horse clears a fence but still gets penalized."
        }}
    ]
}}

[h0]==================================================================[\h0]
REVIEW: 

"There’s too much grind to unlock better horses. It feels like you have to repeat the same thing endlessly. Also, it’s annoying how the controls for lining up the horse before a jump are so imprecise."

TOPICS:

{{
    "Topics": [
        {{
            "Topic": "Progression Grind",
            "Category": "request",
            "Context": "There’s too much grind to unlock better horses. It feels like you have to repeat the same thing endlessly."
        }},
        {{
            "Topic": "Alignment Controls",
            "Category": "request",
            "Context": "It’s annoying how the controls for lining up the horse before a jump are so imprecise."
        }}
    ]
}}

[h0]==================================================================[\h0]
REVIEW: 

"The physics during collisions feel off—it’s like the horse freezes instead of reacting naturally. But overall, I like how realistic the jump animations are."

TOPICS:

{{
    "Topics": [
        {{
            "Topic": "Collision Physics",
            "Category": "request",
            "Context": "The physics during collisions feel off—it’s like the horse freezes instead of reacting naturally."
        }},
        {{
            "Topic": "Animations",
            "Category": "fact",
            "Context": "I like how realistic the jump animations are."
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