from pydantic import BaseModel
from typing import Dict, List, Optional

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float
    all_scores: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    uptime: str
    model_version: str

class RetrainResponse(BaseModel):
    status: str
    message: str
    new_model_path: Optional[str] = None
