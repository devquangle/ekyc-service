import numpy as np
from typing import List, Optional
from pydantic import BaseModel, Field
from ocr.detector import TextDetector, OCRText
from utils.logger import logger


class OcrLine(BaseModel):
    text: str = Field(..., description="Recognized line text content")
    confidence: float = Field(..., description="OCR detection/recognition confidence score")
    boundingBox: List[List[int]] = Field(..., description="4-point polygon bounding box [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]")


class OcrEngine:
    """
    High-Performance PaddleOCR Engine Wrapper for Vietnamese ID Card text recognition.
    Delegates to TextDetector with singleton model reuse.
    """

    def __init__(self):
        self.detector = TextDetector()

    def detect_tokens(self, image: np.ndarray) -> List[OCRText]:
        """
        Detects and recognizes word/phrase OCR tokens with bounding boxes and confidence scores.
        """
        if image is None or image.size == 0:
            return []
        return self.detector.detect(image)

    def detect_and_recognize(self, image: np.ndarray) -> List[OcrLine]:
        """
        Detects text lines and formats output into standard OcrLine models.
        """
        tokens = self.detect_tokens(image)
        lines: List[OcrLine] = []
        for t in tokens:
            bbox_int = [[int(round(pt[0])), int(round(pt[1]))] for pt in t.bbox] if t.bbox else []
            lines.append(
                OcrLine(
                    text=t.text,
                    confidence=float(t.confidence),
                    boundingBox=bbox_int
                )
            )
        return lines
