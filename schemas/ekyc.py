import uuid
from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict
from schemas.card import CardProcessResponse
from schemas.face import FaceVerifyResponse
from schemas.liveness import LivenessResponse
from schemas.enums import EkycOutcome, EkycExecutionStatus


class FullEkycResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    requestId: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique UUID for tracing execution")
    status: Union[EkycExecutionStatus, str] = Field(default=EkycExecutionStatus.SUCCESS, description="Overall execution status: SUCCESS or FAILED")
    ekycResult: Union[EkycOutcome, str] = Field(default=EkycOutcome.EKYC_NOT_VERIFIED, description="Final verification outcome: EKYC_VERIFIED or EKYC_NOT_VERIFIED")
    executionTimeMs: float = Field(default=0.0, description="Total pipeline execution time in milliseconds")
    cardResult: Optional[CardProcessResponse] = Field(default=None, description="Card OCR and Cross-Validation result")
    faceResult: Optional[FaceVerifyResponse] = Field(default=None, description="Face Matching result")
    livenessResult: Optional[LivenessResponse] = Field(default=None, description="Video Liveness Detection result")
    failureReasons: List[str] = Field(default_factory=list, description="Aggregated list of failure reason codes")
