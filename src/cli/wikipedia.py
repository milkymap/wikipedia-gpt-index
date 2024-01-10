
import operator as op 

import numpy as np 

import click 
import wikipedia
from urllib.parse import urlparse

from typing import List 

from src.schema.wikipedia import WikiResource
from src.schema.retrieval import Message, Role

from src.embedding import Embedding

from src.index import Index
from src.retrieval import LLModel

from src.log import logger 

@click.command()
@click.option('--url', type=str, help='url to wikipedia page', required=True)
@click.option('--embedding_model_name', type=str, required=True)
@click.option('--tokenizer_model_name', type=str, default='gpt-3.5-turbo')
@click.option('--chunk_size', default=128)

@click.option('--llm_name', type=str, default='gpt-3.5-turbo')
@click.option('--description', type=str, required=True)
@click.option('--forbidden_topics', type=str, help='list of topics separated by #')
@click.pass_context
def build_index_from_wikipedia(ctx:click.core.Context, url:str, embedding_model_name:str, tokenizer_model_name:str, chunk_size:int ,llm_name:str, description:str, forbidden_topics:str):

    openai_api_key, telegram_token, cache_folder, device = op.itemgetter(
        'openai_api_key', 'telegram_token', 
        'cache_folder', 'device'
    )(ctx.obj)

    forbidden_topics = forbidden_topics.split('#')

    logger.info('get wikipedia resources')
    res = urlparse(url=url)
    language = res.netloc.split('.')[0]
    page_title = res.path.split('/')[-1]

    wikipedia.set_lang(prefix=language)
    page = wikipedia.page(title=page_title)
    page_resource = WikiResource(links=page.links, images=page.images, content=page.content)

    logger.info('compute embeddings')
    embedder = Embedding(embedding_model_name, cache_folder, device, tokenizer_model_name)
    chunks = embedder.tokenize(page_resource.content, chunk_size=chunk_size)
    knowledge = embedder.corpus_embedding(corpus=chunks, chunk_size=chunk_size)
    
    logger.info('initialize index and llm')
    index = Index(knowledge=knowledge)
    language_model = LLModel(
        model_name=llm_name,
        description=description,
        openai_api_key=openai_api_key,
        forbidden_topics=forbidden_topics
    )
    logger.info('rag:++++++++++')

    # move this into a class : llm_interaction
    memory:List[Message] = []
    while True:
        try:
            memory = memory[-5:]  # keep tracking last 5 messages 

            query = input('query:')
            query_embedding = embedder.text_embedding(text=query, chunk_size=chunk_size)
            query_embedding = np.array(query_embedding)
            candidates = index.semantic_search(query_embedding, k=5)

            context = "\n###\n".join(candidates)
            print(context)

            stream = language_model.analyse(
                query=query,
                context=context,
                memory=memory 
            )

            accumulator:List[str] = []
            for chunk in stream:
                chunk_content = chunk.choices[0].delta.content or ''
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
    