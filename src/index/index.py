import numpy as np 

from src.log import logger 

from typing import List 
from src.schema.knowledge import Knowledge, KnowledgeItem

class Index:
    def __init__(self, knowledge:Knowledge): 
        self.texts = list(map(lambda item: item.text, knowledge.items))
        self.fingerprints = list(map(lambda item: item.fingerprint, knowledge.items))
        self.fingerprints = np.vstack(self.fingerprints)
    
    def _score(self, X:np.ndarray, corpus:np.ndarray) -> np.ndarray:
        dot_score = X @ corpus.T 
        norms = np.linalg.norm(X) * np.linalg.norm(corpus, axis=1)
        return dot_score / (norms + 1e-8)

    def _top_k(self, X:np.ndarray, corpus:np.ndarray, k:int) -> List[int]:
        scores = self._score(X, corpus)
        return np.argsort(-scores)[:k].tolist()

    def semantic_search(self, X:np.ndarray, k:int) -> List[str]:
        indices = self._top_k(X, self.fingerprints, k)
        candidates:List[str] = []
        for i in indices:
            candidates.append(self.texts[i])
        return candidates