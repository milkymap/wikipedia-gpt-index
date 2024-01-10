
import numpy as np 

from src.embedding import Embedding
from src.index import Index
from src.retrieval import LLModel

from typing import List 

from src.schema.retrieval import Message, Role

from src.log import logger 

class Conversation:
    def __init__(self, embedder:Embedding, index:Index, language_model:LLModel):
        self.embedder = embedder
        self.index = index  
        self.language_model = language_model

    def run_loop(self, memory_size:int, chunk_size:int, context_size:int):
        memory:List[Message] = []
        fn_name = '' 
        fn_args = ''

        while True:
            try:
                memory = memory[-memory_size:]  # keep tracking last 5 messages 

                query = input('query:')
                query_embedding = self.embedder.text_embedding(text=query, chunk_size=chunk_size)
                query_embedding = np.array(query_embedding)
                candidates = self.index.semantic_search(query_embedding, k=context_size)

                context = "\n###\n".join(candidates)
                
                stream = self.language_model.analyse(
                    query=query,
                    context=context,
                    memory=memory 
                )

                accumulator:List[str] = []
                for chunk in stream:
                    tool_calls = chunk.choices[0].delta.tool_calls
                    chunk_content = chunk.choices[0].delta.content
                    
                    if tool_calls is not None:
                        function = tool_calls[0].function
                        if function is not None:
                            fn_name = fn_name + (function.name or '')
                            fn_args = fn_args + (function.arguments or '')
                    
                    if chunk.choices[0].finish_reason == 'tool_calls':
                        print('call', fn_name, fn_args)
                        fn_name, fn_args = '', ''

                    if chunk_content is None:
                        continue

                    print(chunk_content, end='', flush=True)
                    accumulator.append(chunk_content)
                
                print('')

                assistant_response = ''.join(accumulator)
                memory += [
                    Message(
                        role=Role.USER,
                        content=query
                    ),
                    Message(
                        role=Role.ASSISTANT,
                        content=assistant_response
                    )
                ]

            except KeyboardInterrupt:
                break 
            except Exception as e:
                logger.error(e)
                break 