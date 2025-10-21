import openai
import numpy as np
from pydantic import BaseModel
from typing import List


SYSTEM_PROMPT="""
You are a helpful assistant to a data scientist working in NLP.
The data scientist has created a topic model over a collection of documents.
The topics are represented by characteristic words and documents that are provided to you.
Please provide a single label for each topic and a one sentence description of the topic in the structured output.
"""



class TopicSchema(BaseModel):
    topic_labels: List[str]
    topic_description: List[str]

class OpenAI():
    
    def __init__(self, api_key, model="gpt-4o-mini-2024-07-18"):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key)
        self.model = model

    
    def prompt(self, user_prompt, system_prompt, schema):
        completion = self.client.beta.chat.completions.parse(
        model=self.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=schema,
    )
        try: 
            return completion.choices[0].message.parsed
        except:
            return completion.choices[0].message.content


    def get_topics(self, topic_words,top_docs):
        assert(len(topic_words) == len(top_docs))
        user_prompt = ""
        system_prompt = SYSTEM_PROMPT
        _schema = TopicSchema
        
        for i in range(len(topic_words)):
            user_prompt += f"Topic {i+1} top words: {topic_words[i]} \n"
            system_prompt += f"Topic {i+1} top documents: {top_docs[i]} \n"
        
        return self.prompt(user_prompt, system_prompt, _schema)
    