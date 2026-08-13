from typing import List, Optional
from pydantic import BaseModel, Field
from schemas.card import CardProcessResponse
from schemas.face import FaceVerifyResponse
from schemas.liveness import LivenessResponse


class FullEkycResponse(BaseModel):
    requestId: str = Field(..., description="Unique UUID for tracing execution")
    status: str = "SUCCESS"  # SUCCESS or FAILED
    ekycResult: str = "EKYC_NOT_VERIFIED"  # EKYC_VERIFIED or EKYC_NOT_VERIFIED
    executionTimeMs: float = 0.0
    cardResult: Optional[CardProcessResponse] = None
    faceResult: Optional[FaceVerifyResponse] = None
    livenessResult: Optional[LivenessResponse] = None
    failureReasons: List[str] = Field(default_factory=list)
