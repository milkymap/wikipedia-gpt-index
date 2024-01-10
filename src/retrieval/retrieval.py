
import openai 

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from src.schema.retrieval import Role, Message

from pydantic import BaseModel

from typing import List, Dict, Iterable 

class LLModel:
    def __init__(self, model_name:str, openai_api_key:str, description:str, forbidden_topics:List[str]):
        self.forbidden_topics = forbidden_topics
        self.description = description
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=openai_api_key
        )
    
    def _build_system_setting(self, context:str) -> Message:
        return Message(
            role=Role.SYSTEM,
            content=f"""
                ROLE:
                    you are an assistant that will help the user to explore this document : {self.description}
                    your role is to help the user to apply retrieval augmented generation based on the following context:
                    here is the context : ###
                    {context}
                    ###
                USER/ASSISTANT FLOW:
                    |user --> greeting --> assistant --> reply greeting
                    |user --> ask information about your role --> assistant --> present yourself
                    |user --> ask a summary of the page --> call the corresponding tools (function_calling) 
                    |user --> ask questions about given the context --> try the build the response if there is one else tell the user that you was not able to build the response 
                    |user --> talk about forbidden topics --> assistant --> tell the user that your scope is limited to this document
                IMPORTANT:
                    FORBIDDEN-TOPICS={self.forbidden_topics}
            """
        )

    def _summary_fn_schema(self): 
        return {
            'type': 'function',
            'function': {
                'name': 'get_wikipedia_page_summary',
                'description': 'return the summary of the page',
                'parameters': {
                    'type': 'object', 
                    'properties': {
                    }
                }
            }
        }

    def _build_messages(self, query:str, memory:List[Message]) -> List[Dict[str, str]]:
        user_message = Message(
            role=Role.USER,
            content=query
        )    
        return [ mem.model_dump() for mem in memory ] + [ user_message.model_dump() ]

    def analyse(self, query:str, context:str, memory:List[Message]) -> Iterable[ChatCompletionChunk]:
        system_message = self._build_system_setting(context=context)
        messages = [ system_message.model_dump() ] + self._build_messages(query, memory)
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=[self._summary_fn_schema()],
            stream=True
        )
        return completion


    