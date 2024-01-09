
from typing import List 
from pydantic import BaseModel

class WikiResource(BaseModel):
    links:List[str]
    images:List[str]
    content:str 
