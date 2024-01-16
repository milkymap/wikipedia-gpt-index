
from pydantic import BaseModel

from typing import List 

class KnowledgeItem(BaseModel):
    text:str 
    fingerprint:List[float]

class Knowledge(BaseModel):
    items:List[KnowledgeItem]