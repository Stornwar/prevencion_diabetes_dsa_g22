from pydantic import BaseModel
from typing import List, Optional

class PredictionResults(BaseModel):
    predictions: List[int]
    version: str
    errors: Optional[List[str]] = None

class Health(BaseModel):
    name: str
    api_version: str
    model_version: str
