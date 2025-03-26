from langchain.prompts import PromptTemplate


""" 
Adjust the template below to match your task.

Note: If you change the structure not only the semantics, you may need to adjust the code in the main script as well.

"""

# region General Pipeline Prompts
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

prompt_template_topic_zendesk = PromptTemplate.from_template(
'''
Please list the most important topics and their respective original context from the Zendesk ticket in a JSON format with "Topic", "Category", and "Context" arguments.
Ignore disclaimers, signatures, or forwarded message headers unless they contain an actual game-related fact or request.
"Topic" should focus on a specific feature or aspect. Group single user statements as one topic if they address only one concern; split only if there are multiple distinct issues.
"Category" must be either "fact" or "request".
"Context" is a direct quote or excerpt from the ticket that explains the topic.
Return no more than 10 topics. Respond in valid JSON format only.

[h0]==================================================================[\h0]
TICKET:

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
TICKET:

"Linda E, MS, CCS
Sent from my iPad
1234@me.com

THE ONLY THING NECESSARY FOR EVIL TO TRIUMPH 
IS THAT GOOD MEN DO NOTHING.

Begin forwarded message:

> From: Linda Peoples <1234@icloud.com>
> Subject: Correct map needed
>
> This is not a request or wish - this is a necessity.
> You need to put the correct maps at the bottom left hand corner of each flat race.
> THIS needs to be corrected before any traits, or colors, or other fluff is added - this is paramount to the correctness of the game.
> PLEASE FIX IT.
> Sincerely,
> Linda

To unsubscribe from this group and stop receiving emails from it, send an email to support+unsubscribe@test.com.
image0.jpeg

TOPICS:

{{  
    "Topics": [ 
       {{ 
           "Topic": "Map Accuracy", 
           "Category": "request", 
           "Context": "You need to put the correct maps at the bottom left hand corner of each flat race." 
       }} 
   ] 
}}


[h0]==================================================================[\h0]
TICKET:

"Description: Sound goes in and out and sometimes comes back but most of the time I have to close the app and restart it, very annoying and makes me not like the game as much. 

Question or inquiry:  

In-Game Username: Laura456

Order ID:  

Steps to Reproduce: Happens randomly but almost every single time I play the game and even if I reload the game it still happens. Sound stops while playing at random times. Nothing specifically happens to produce the outcome, simply playing the game at all it happens. 
"

TOPICS:

{{
    "Topics": [
        {{
            "Topic": "Sound loss",
            "Category": "fact",
            "Context": "Sound goes in and out and sometimes comes back but most of the time I have to close the app and restart it... Happens randomly but almost every single time I play the game and even if I reload the game it still happens"
        }}
    ]
}}


[h0]==================================================================[\h0]
TICKET:


"{zendesk_ticket}"


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

# endregion

# region Prompt templates for the Cluster Report

# Used for the cluster report (individual clusters)
prompt_template_summary_short = PromptTemplate.from_template(
    """
You are analyzing player statements for our video game {video_game}.
Below are the player statements from the cluster: {cluster_name}.

Please provide a short summary (2-3 sentences) that captures the main topics of those statements:
{statements}
"""
)
# Used for the Big Picture Report (top 5 clusters)
prompt_template_top5 = PromptTemplate.from_template(
    """
You are analyzing player statements for our video game {video_game}.
Below are the player statements from the {cluster_group} clusters.

Your task is to identify the the main {sentiment} topics in the statements. Give a brief summary about them in 2-3 sentences.
Here are the player statements:

{statements}
"""
)

# endregion

# region experimental
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

# endregion