import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from ocr.detector import TextDetector, OCRText
from utils.logger import logger


class OcrLine(BaseModel):
    text: str
    confidence: float
    boundingBox: List[List[int]]


class OcrEngine:
    """
    PaddleOCR Wrapper delegating detection to ocr.detector.TextDetector while
    maintaining backward compatibility.
    """

    def __init__(self):
        self.detector = TextDetector()

    def detect_tokens(self, image: np.ndarray) -> List[OCRText]:
        return self.detector.detect(image)

    def detect_and_recognize(self, image: np.ndarray) -> List[OcrLine]:
        tokens = self.detect_tokens(image)
        lines = []
        for t in tokens:
            bbox_int = [[int(pt[0]), int(pt[1])] for pt in t.bbox]
            lines.append(
                OcrLine(
                    text=t.text,
                    confidence=t.confidence,
                    boundingBox=bbox_int
                )
            )
        return lines
