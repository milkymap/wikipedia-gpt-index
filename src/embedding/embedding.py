import numpy as np 


from typing import List, Tuple, Dict 

from tiktoken import encoding_for_model, Encoding
from sentence_transformers import SentenceTransformer

from src.schema.knowledge import KnowledgeItem, Knowledge

class Embedding:
    def __init__(self, model_name:str, cache_folder:str, device:str, tokenizer_model_name:str):
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            cache_folder=cache_folder,
            device=device
        )
        self.codec = encoding_for_model(model_name=tokenizer_model_name)
    
    def vectorize(self, chunks:List[str]) -> List[float]:
        vectors = self.model.encode(sentences=chunks)
        if len(vectors) == 1:
            return vectors[0].tolist()
        fingerprint = np.mean(vectors)
        return fingerprint.tolist()
    
    def tokenize(self, text:str, chunk_size:int) -> List[str]:
        tokens:List[int] = self.codec.encode(text=text)
        nb_tokens = len(tokens)
        accumulator:List[str] = []
        for index in range(0, nb_tokens, chunk_size):
            partition = tokens[index:index+chunk_size]
            chunk = self.codec.decode(tokens=partition)
            accumulator.append(chunk)
        
        return accumulator
    
    def text_embedding(self, text:str, chunk_size:int=128) -> List[float]:
        chunks = self.tokenize(text, chunk_size=chunk_size)
        fingerprint = self.vectorize(chunks=chunks)
        return fingerprint
    
    def corpus_embedding(self, corpus:List[str], chunk_size:int) -> Knowledge:
        items:List[KnowledgeItem] = []
        for text in corpus:
            fingerprint = self.text_embedding(text=text, chunk_size=chunk_size)
            items.append(
                KnowledgeItem(
                    text=text,
                    fingerprint=fingerprint
                )
            )
        return Knowledge(items=items)


