from langchain.prompts import PromptTemplate

prompt_template_translation = PromptTemplate.from_template(
'''Please translate each section into English if it is not. The sections are separated by labels "REASON" and "WISH".

[h0]==================================================================[\h0]
REASON: "兄弟们，我把星空退款的钱拿来买这个了，我做的对吗"
WISH: "加动态模糊和垂直同步选项"

TRANSLATION:

REASON: "Brothers, I used the refund money from the stars to buy this. Did I do the right thing?"
WISH: "Add dynamic blur and vertical sync options."


[h0]==================================================================[\h0]
REASON: "My first D&D experience and I'm enjoying it a lot."
WISH: "I would like more guidance in the game."

TRANSLATION:

REASON: "My first D&D experience and I'm enjoying it a lot."
WISH: "I would like more guidance in the game."

[h0]==================================================================[\h0]
REASON: "{reason}"
WISH: "{wish}"

TRANSLATION:

'''
)

prompt_template_topic = PromptTemplate.from_template(
'''Please list the most important topics and their respective original context in the review of a game in a json format with "Topic", "Category", "Context" arguments.  No more than 10 topics.
Topics should be game features.  A feature in the game should be a noun rather than a verb or an adjective.
Each topic should be categorized as a "fact" or a "request".
Respond in JSON format.

[h0]==================================================================[\h0]
REVIEW: 

"The weapon durability in this game is frustrating; my sword breaks after just a few swings. The combat itself is fun, but I wish the durability lasted longer. Also, the audio effects are very immersive during battles."

TOPICS:

{{"Topics":
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
}}

[h0]==================================================================[\h0]
REVIEW: 

"Playing during the night adds a thrilling layer to the game. The lack of a proper save feature makes it hard to enjoy it though. Also, there are way too many random encounters that make progress difficult."

TOPICS:

{{"Topics":
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
}}

[h0]==================================================================[\h0]
REVIEW: 

"{review}"

TOPICS:

'''
)

prompt_template_topic_view = PromptTemplate.from_template(
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