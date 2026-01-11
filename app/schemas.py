from pydantic import BaseModel, Field
from typing import Dict, Any

class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dict. Keys must match training columns.")

class PredictResponse(BaseModel):
    churn_probability: float
    will_churn: bool
