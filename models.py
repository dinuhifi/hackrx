from pydantic import BaseModel, HttpUrl
from typing import List, Dict

class APIRequest(BaseModel):
    documents: str
    questions: List[str]
    
class AnswerResponse(BaseModel):
    answers: List[str]