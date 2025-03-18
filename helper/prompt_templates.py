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
Please list the most important topics and their respective original context in the review of a videogame in a JSON format with "Topic", "Category", and "Context" arguments.
The Topic should focus on a specific game feature or aspect. The Category should be either "fact", "request" or "bug". The Context should be a direct excerpt from the review that explains the topic.
No more than 10 topics.
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


"The physics during collisions often bug out which leads to missed shots. The wasteland biom sucks completely, please fix the errors with the tree colors in there. 


TOPICS:


{{
   "Topics": [
       {{
           "Topic": "Collision",
           "Category": "bug",
           "Context": "The physics during collisions often bug out which leads to missed shots."
       }},
       {{
           "Topic": "Wasteland Biom",
           "Category": "fact",
           "Context": "The wasteland biom sucks completely."
       }},
       {{
           "Topic": "Wasteland tree colors",
           "Category": "bug",
           "Context": "The wasteland biom sucks completely, please fix the errors with the tree colors in there."
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


prompt_template_influencer = PromptTemplate.from_template(
'''
Please list the most important topics and their respective original context in the excerpt of a video transcript in a JSON format with "Topic", "Category", and "Context" arguments.
The Topic should focus on a specific game feature or aspect. The Category should be either "fact" or "request". The Context should be a direct quote from the transcript that explains the topic.


[h0]==================================================================[\h0]
TRANSCRIPT:

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
TRANSCRIPT:

"{transcript}"

TOPICS:

'''
)

prompt_template_summary = PromptTemplate.from_template(
'''
cluster_name
You are analysing player statements for our video game {video_game}. This is the cluster where you should analyse the player statements:
{sentiment_distribution}
Provide a concise summary where you name any major themes or issues that stand out.
List the key insights, concerns or praises in this cluster as **bullet points** below.

Here are the player statements in that cluster:

{statements}
'''
)

prompt_template_summary_short = PromptTemplate.from_template(
    """
You are analyzing player statements for our video game {video_game}.
Below are the player statements from the cluster: {cluster_name}.

Please provide a short summary (2-3 sentences) that captures the main topics of those statements:
{statements}
"""
)

prompt_template_pain_points = PromptTemplate.from_template(
    """
You are analyzing player statements for our video game {video_game}.
Below are the player statements from the cluster: {cluster_name}.

Your task is to identify the biggest **pain points** (complaints or issues) mentioned by players.
Provide them as **bullet points**. Only list frequent or significant pain points (maximum 3).
If you cant identify any pain points just write "No issues mentioned".
{statements}
"""
)

prompt_template_highlights = PromptTemplate.from_template(
    """
You are analyzing player statements for our video game {video_game}.
Below are the player statements from the cluster: {cluster_name}.

Your task is to identify the **biggest strengths** (praises or well perceived aspects) mentioned by players.
Provide them as **bullet points**. Only list frequent or significant strengths (maximum 3).
If you cant identify any highlights just write "No strengths mentioned".
{statements}
"""
)

prompt_template_top5 = PromptTemplate.from_template(
    """
You are analyzing player statements for our video game {video_game}.
Below are the player statements from the {cluster_group} clusters.

Your task is to identify the the main {sentiment} topics in the statements. Give a brief summary about them in 2-3 sentences.
Here are the player statements:

{statements}
"""
)




prompt_template_cluster_naming = (
'''Based on the following topics, generate a concise name (5 words or fewer) that best describes the general theme of this cluster.

TOPICS: {topics}
CLUSTER NAME: '''
)