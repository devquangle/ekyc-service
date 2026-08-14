import numpy as np
from typing import List, Optional
from pydantic import BaseModel, Field
from utils.logger import logger
from utils.image_utils import resize_maintain_aspect


from utils.text_normalizer import VietnameseTextCorrector, UnicodeNormalizer


class OCRText(BaseModel):
    text: str
    confidence: float
    bbox: List[List[float]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    raw_text: Optional[str] = None
    center_x: float = 0.0
    center_y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def model_post_init(self, __context):
        if not self.raw_text:
            self.raw_text = self.text
        # Ensure Unicode NFC normalization on self.text
        if self.text:
            _, ocr_val, _, _ = VietnameseTextCorrector.normalize_pipeline(self.text)
            self.text = ocr_val or self.text.strip()
        if self.bbox and len(self.bbox) >= 4:
            xs = [pt[0] for pt in self.bbox]
            ys = [pt[1] for pt in self.bbox]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            self.width = float(max_x - min_x)
            self.height = float(max_y - min_y)
            self.center_x = float(min_x + self.width / 2.0)
            self.center_y = float(min_y + self.height / 2.0)


class TextDetector:
    """
    PaddleOCR detector returning structured OCRText objects with bounding boxes,
    centers, width, height, and confidence rescaled to original image dimensions.
    """

    def __init__(self):
        self.ocr = None
        self._initialize_model()

    def _initialize_model(self):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='vi',
                show_log=False,
                rec_batch_num=6
            )
            logger.info("PaddleOCR engine initialized successfully.")
        except Exception as e:
            logger.warning(f"PaddleOCR failed to initialize (operating in fallback mode): {str(e)}")
            self.ocr = None

    def detect(self, image: np.ndarray, target_width: int = 1024) -> List[OCRText]:
        if image is None or image.size == 0:
            return []

        orig_h, orig_w = image.shape[:2]
        resized_img = resize_maintain_aspect(image, target_width=target_width)
        res_h, res_w = resized_img.shape[:2]

        scale_x = orig_w / float(res_w) if res_w > 0 else 1.0
        scale_y = orig_h / float(res_h) if res_h > 0 else 1.0

        if self.ocr is None:
            logger.debug("TextDetector operating in fallback mode (No OCR engine).")
            return []

        try:
            results = self.ocr.ocr(resized_img, cls=True)
            ocr_tokens: List[OCRText] = []

            if not results or not results[0]:
                return ocr_tokens

            for line in results[0]:
                bbox_raw = line[0]
                text, confidence = line[1]
                bbox_float = [[float(pt[0]) * scale_x, float(pt[1]) * scale_y] for pt in bbox_raw]

                raw_str = str(text)
                _, ocr_val, _, _ = VietnameseTextCorrector.normalize_pipeline(raw_str)

                token = OCRText(
                    text=ocr_val or raw_str.strip(),
                    raw_text=raw_str,
                    confidence=float(confidence),
                    bbox=bbox_float
                )
                ocr_tokens.append(token)

            return ocr_tokens
        except Exception as e:
            logger.error(f"Error during TextDetector execution: {str(e)}")
            return []
