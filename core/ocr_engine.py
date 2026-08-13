import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from utils.logger import logger
from utils.image_utils import resize_maintain_aspect


class OcrLine(BaseModel):
    text: str
    confidence: float
    boundingBox: List[List[int]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


class OcrEngine:
    """
    PaddleOCR Wrapper for text detection, angle classification, and Vietnamese text recognition.
    """

    def __init__(self):
        self.ocr = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            from paddleocr import PaddleOCR
            # Initialize PaddleOCR with Vietnamese language support
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='vi',
                show_log=False,
                rec_batch_num=6
            )
            logger.info("PaddleOCR engine initialized successfully.")
        except Exception as e:
            logger.warning(f"PaddleOCR failed to initialize (falling back to mock/CPU mode): {str(e)}")
            self.ocr = None

    def detect_and_recognize(self, image: np.ndarray) -> List[OcrLine]:
        """
        Detects and recognizes text lines in an image.
        Returns a list of OcrLine objects.
        """
        if image is None or image.size == 0:
            return []

        # Resize to standard width 1024 for consistent OCR performance
        resized_img = resize_maintain_aspect(image, target_width=1024)

        if self.ocr is None:
            logger.debug("OcrEngine operating in fallback mode.")
            return []

        try:
            results = self.ocr.ocr(resized_img, cls=True)
            ocr_lines: List[OcrLine] = []

            if not results or not results[0]:
                return ocr_lines

            for line in results[0]:
                bbox_raw = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text, confidence = line[1]

                # Convert bbox coordinates to integer
                bbox_int = [[int(pt[0]), int(pt[1])] for pt in bbox_raw]

                ocr_lines.append(
                    OcrLine(
                        text=str(text).strip(),
                        confidence=float(confidence),
                        boundingBox=bbox_int
                    )
                )

            return ocr_lines

        except Exception as e:
            logger.error(f"Error during OCR execution: {str(e)}")
            return []
