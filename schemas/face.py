from typing import List, Optional
from pydantic import BaseModel, Field


class BoundingBoxInfo(BaseModel):
    detected: bool = Field(default=False, description="True if face detected")
    bbox: List[int] = Field(default_factory=list, description="Bounding Box [x1, y1, x2, y2]")
    x1: int = Field(default=0, description="Top-left X coordinate")
    y1: int = Field(default=0, description="Top-left Y coordinate")
    x2: int = Field(default=0, description="Bottom-right X coordinate")
    y2: int = Field(default=0, description="Bottom-right Y coordinate")
    width: int = Field(default=0, description="Bounding Box width")
    height: int = Field(default=0, description="Bounding Box height")
    detectionScore: float = Field(default=0.0, description="Face detection confidence score")


class FaceQualityMetrics(BaseModel):
    blurScore: float = Field(default=0.0, description="Laplacian blur score (higher is sharper)")
    brightness: float = Field(default=0.0, description="Mean grayscale brightness")
    faceSize: int = Field(default=0, description="Face crop width dimension")
    yaw: float = Field(default=0.0, description="Head pose Yaw angle")
    pitch: float = Field(default=0.0, description="Head pose Pitch angle")
    roll: float = Field(default=0.0, description="Head pose Roll angle")


class FaceVerifyResponse(BaseModel):
    faceVerified: bool = Field(default=False, description="True if faces match above threshold")
    similarityScore: float = Field(default=0.0, description="Cosine similarity score [-1.0, 1.0]")
    threshold: float = Field(default=0.60, description="Matching decision threshold")
    decision: str = Field(default="MISMATCH", description="Decision category: MATCH, BORDERLINE, or MISMATCH")
    margin: float = Field(default=0.0, description="Difference between similarityScore and threshold")
    errors: List[str] = Field(default_factory=list, description="List of verification error/warning codes")
    cardFaceInfo: Optional[BoundingBoxInfo] = Field(default=None, description="Card face detection bounding box metrics")
    selfieFaceInfo: Optional[BoundingBoxInfo] = Field(default=None, description="Selfie face detection bounding box metrics")
    cardFaceQuality: Optional[FaceQualityMetrics] = Field(default=None, description="Card face image quality metrics")
    selfieFaceQuality: Optional[FaceQualityMetrics] = Field(default=None, description="Selfie face image quality metrics")
