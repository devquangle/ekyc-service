from typing import List, Optional, Union, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator
from schemas.enums import VerificationDecision


class BoundingBoxInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    detected: bool = Field(default=False, description="True if face detected")
    bbox: List[int] = Field(default_factory=list, description="Bounding Box [x1, y1, x2, y2]")
    x1: int = Field(default=0, description="Top-left X coordinate")
    y1: int = Field(default=0, description="Top-left Y coordinate")
    x2: int = Field(default=0, description="Bottom-right X coordinate")
    y2: int = Field(default=0, description="Bottom-right Y coordinate")
    width: int = Field(default=0, description="Bounding Box width")
    height: int = Field(default=0, description="Bounding Box height")
    detectionScore: float = Field(default=0.0, description="Face detection confidence score")

    @model_validator(mode="before")
    @classmethod
    def validate_and_sync_bbox_dict(cls, data: Any) -> Any:
        if isinstance(data, dict):
            x1 = data.get("x1", 0)
            y1 = data.get("y1", 0)
            x2 = data.get("x2", 0)
            y2 = data.get("y2", 0)
            w = data.get("width", 0)
            h = data.get("height", 0)
            bbox = data.get("bbox") or []

            # If bbox is given, fill x1, y1, x2, y2
            if bbox and len(bbox) == 4 and (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                x1, y1, x2, y2 = bbox
            if x2 > x1 and w == 0:
                w = int(x2 - x1)
            if y2 > y1 and h == 0:
                h = int(y2 - y1)
            if w > 0 and x2 == 0:
                x2 = int(x1 + w)
            if h > 0 and y2 == 0:
                y2 = int(y1 + h)
            if not bbox and (x2 > x1 or y2 > y1):
                bbox = [int(x1), int(y1), int(x2), int(y2)]

            data["x1"] = int(x1)
            data["y1"] = int(y1)
            data["x2"] = int(x2)
            data["y2"] = int(y2)
            data["width"] = int(w)
            data["height"] = int(h)
            data["bbox"] = [int(v) for v in bbox]
        return data


class FaceQualityMetrics(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    blurScore: float = Field(default=0.0, description="Laplacian blur score (higher is sharper)")
    brightness: float = Field(default=0.0, description="Mean grayscale brightness")
    faceSize: int = Field(default=0, description="Face crop width dimension")
    yaw: float = Field(default=0.0, description="Head pose Yaw angle")
    pitch: float = Field(default=0.0, description="Head pose Pitch angle")
    roll: float = Field(default=0.0, description="Head pose Roll angle")


class FaceVerifyResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    faceVerified: bool = Field(default=False, description="True if faces match above threshold")
    similarityScore: float = Field(default=0.0, description="Cosine similarity score [-1.0, 1.0]")
    threshold: float = Field(default=0.60, description="Matching decision threshold")
    decision: Union[VerificationDecision, str] = Field(
        default=VerificationDecision.MISMATCH,
        description="Decision category: MATCH, BORDERLINE, or MISMATCH"
    )
    margin: float = Field(default=0.0, description="Difference between similarityScore and threshold")
    errors: List[str] = Field(default_factory=list, description="List of verification error/warning codes")
    cardFaceInfo: Optional[BoundingBoxInfo] = Field(default=None, description="Card face detection bounding box metrics")
    selfieFaceInfo: Optional[BoundingBoxInfo] = Field(default=None, description="Selfie face detection bounding box metrics")
    cardFaceQuality: Optional[FaceQualityMetrics] = Field(default=None, description="Card face image quality metrics")
    selfieFaceQuality: Optional[FaceQualityMetrics] = Field(default=None, description="Selfie face image quality metrics")
