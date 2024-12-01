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
Please list the most important topics and their respective original context in the review of a game in a JSON format with "Topic", "Category", and "Context" arguments.
The Topic should focus on a specific game feature or aspect. The Category should be either "fact" or "request". The Context should be a direct excerpt from the review that explains the topic.
Players have been ask "Is there anything you currently find frustrating in the Show Jumping?"
Avoid repeating the central feature "Show Jumping" in every topic unless it is strictly necessary.
No more than 10 topics.
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