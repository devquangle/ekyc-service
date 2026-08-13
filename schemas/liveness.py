from typing import List
from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    livenessVerified: bool = False
    livenessScore: float = 0.0
    threshold: float = 0.80
    checksPassed: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
