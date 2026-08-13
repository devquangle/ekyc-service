from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBoxInfo(BaseModel):
    detected: bool = False
    bbox: List[int] = Field(default_factory=list)  # [x1, y1, x2, y2]
    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    width: int = 0
    height: int = 0
    detectionScore: float = 0.0


class FaceQualityMetrics(BaseModel):
    blurScore: float = 0.0
    brightness: float = 0.0
    faceSize: int = 0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


class FaceVerifyResponse(BaseModel):
    faceVerified: bool = False
    similarityScore: float = 0.0
    threshold: float = 0.60
    decision: str = "MISMATCH"  # MATCH | BORDERLINE | MISMATCH
    margin: float = 0.0
    errors: List[str] = Field(default_factory=list)
    cardFaceInfo: Optional[BoundingBoxInfo] = None
    selfieFaceInfo: Optional[BoundingBoxInfo] = None
    cardFaceQuality: Optional[FaceQualityMetrics] = None
    selfieFaceQuality: Optional[FaceQualityMetrics] = None

