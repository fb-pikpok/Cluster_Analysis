from langchain.prompts import PromptTemplate

"""
Templates of previous successful analyses. 
"""


# region Jonos DRS survey
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
            "Topic": "Random Encounters",
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

prompt_template_cluster_naming = (
'''Based on the following topics, generate a concise name (5 words or fewer) that best describes the general theme of this cluster.

TOPICS: {topics}
CLUSTER NAME: '''
)
# endregion


# region HRC general

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

# endregion


# region Steam reviews

prompt_template_topic_steam = PromptTemplate.from_template(
'''
Please list the most important topics and their respective original context in the review of a game in a json format with "Topic", "Category", "Context" arguments.
Topics should be game features.  A feature in the game should be a noun rather than a verb or an adjective.
Each topic should be categorized as a "fact" or a "request".
No more than 10 topics.
Respond in JSON format.


[h0]==================================================================[\h0]
REVIEW: 

"My first D&D experience and I'm enjoying it a lot. However, I would prefer a little more guidance as the mechanics are quite overwhelming at first (especially the combat mechanics)."

TOPICS:

[
    {{
        "Topic": "D:D",
        "Category": "fact",
        "Context": "My first D&D experience and I'm enjoying it a lot.",
    }},
    {{
        "Topic": "combat mechanic",
        "Category": "fact",
        "Context": "the mechanics are quite overwhelming at first (especially the combat mechanics)",
    }}
]


[h0]==================================================================[\h0]
REVIEW: 

"I have only one point of criticism:
This game lacks a proper ending or epilog; something like The Witcher's Blood & Wine DLC. After adventuring hundreds of hours, after growing fond of and close to my companions, I got kicked so hard out of the game - it hurt!
In fact I immediately started a 2nd playthrough, because I wasn't yet ready to let go. Please Larian, give us an epilog or at least one last night at the camp with all my allies, companions and friends for the Definitive Edition."

TOPICS:

[
    {{
        "Topic": "epilogue",
        "Category": "fact",
        "Context": "This game lacks a proper ending or epilog",
    }},
    {{
        "Topic": "companion",
        "Category": "request",
        "Context": "Please Larian, give us an epilog or at least one last night at the camp with all my allies, companions and friends for the Definitive Edition",
    }}
]


[h0]==================================================================[\h0]
REVIEW: 

"{review}"

TOPICS:

'''
)