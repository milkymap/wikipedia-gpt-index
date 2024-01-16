
from typing import List 
from pydantic import BaseModel

class WikiResource(BaseModel):
    summary:str 
    links:List[str]
    images:List[str]
    content:str 
