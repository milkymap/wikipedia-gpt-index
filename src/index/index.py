import zmq 
import multiprocessing as mp

from urllib.parse import urlparse
from typing import List, Tuple 

import wikipedia

from src.log import logger 
from src.server import BOTServer

from src.schema.wikipedia import WikiResource
from src.embedding import Embedding

class ZMQIndex:
    def __init__(self, port:int):
        self.endpoint = f'tcp://*:{port}'

    def _start_background_bot(self, openai_api_key:str, telegram_token:str):
        with BOTServer(token=telegram_token, api_key=openai_api_key) as bot:
            bot.listen()
    
    def _process_url(self, url:str) -> Tuple[str, str]:
        res = urlparse(url=url)
        language = res.netloc.split('.')[0]
        page_title = res.path.split('/')[-1]

        return language, page_title

    def _download_wikipedia_resources(self, language:str, page_title:str) -> WikiResource:
        wikipedia.set_lang(prefix=language)
        page = wikipedia.page(title=page_title)
        return WikiResource(
            links=page.links,
            images=page.images,
            content=page.content
        )
    
    def _compute_embeddings(self, page_resource:WikiResource, model_name:str, cache_folder:str, device:str, tokenizer_model_name:str, chunk_size:int=128):
        embedder = Embedding(model_name, cache_folder, device, tokenizer_model_name)
        chunks = embedder.tokenize(page_resource.content, chunk_size=chunk_size)
        knowledge = embedder.corpus_embedding(corpus=chunks, chunk_size=chunk_size)
        
    def launch_server(self, openai_api_key:str, telegram_token:str):
        ctx = zmq.Context
        router_socket:zmq.Socket = ctx.socket(zmq.ROUTER)
        router_socket.bind(self.endpoint)

        poller = zmq.Poller()
        poller.register(router_socket, zmq.POLLIN)

        process_ = mp.Process(target=self._start_background_bot, args=(openai_api_key, telegram_token))
        process_.start()

        background_embedding_processes:List[mp.Process] = []

        while True:
            try:
                socket_states_hmap = dict(poller.poll(timeout=100))
                incoming_event = socket_states_hmap.get(router_socket, zmq.POLLIN)
                if incoming_event != zmq.POLLIN:
                    continue

                client_id, _, client_message = router_socket.recv_multipart()
                url2wikipedia = client_message.decode()
                language, page_title = self._process_url(url=url2wikipedia)
                page_resource = self._download_wikipedia_resources(language=language, page_title=page_title)
                worker = mp.Process(
                    target=self._compute_embeddings,
                    args=[page_resource]
                )
                background_embedding_processes.append(worker)
                background_embedding_processes[-1].start()
            except KeyboardInterrupt:
                break 
            except Exception as e:
                logger.error(e)
                break 
        
        if process_.is_alive():
            process_.terminate()
            process_.join()
        
        router_socket.close(linger=0)
        ctx.term()

