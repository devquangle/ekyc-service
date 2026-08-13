from typing import List
from pydantic import BaseModel, Field


class FaceVerifyResponse(BaseModel):
    faceVerified: bool = False
    similarityScore: float = 0.0
    threshold: float = 0.45
    errors: List[str] = Field(default_factory=list)
