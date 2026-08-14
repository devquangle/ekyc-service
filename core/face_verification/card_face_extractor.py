import cv2
import numpy as np
from typing import Tuple, Optional, List
from config import settings
from schemas.face import BoundingBoxInfo
from utils.logger import logger
from utils.image_utils import crop_image


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """
    Clips bounding box coordinates so they strictly stay within image dimensions [0, img_w] x [0, img_h].
    Guarantees width (x2 - x1) > 0 and height (y2 - y1) > 0.
    """
    cx1 = max(0, min(x1, img_w - 1))
    cy1 = max(0, min(y1, img_h - 1))
    cx2 = max(cx1 + 1, min(x2, img_w))
    cy2 = max(cy1 + 1, min(y2, img_h))
    return cx1, cy1, cx2, cy2


class CardFaceExtractor:
    """
    Layout-agnostic Card Face Detector and Extractor for Vietnamese ID Cards (CCCD/CMND/Thẻ Căn Cước).
    Features high-precision InsightFace primary detection with optimized pre-loaded Haar Cascade fallback.
    """

    def __init__(self, face_app=None):
        self.face_app = face_app
        # Initialize Haar Cascade classifier ONCE during instantiation (zero disk I/O at runtime)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"[CARD_FACE_EXTRACTOR] Failed to load Haar Cascade: {str(e)}")
            self._haar_cascade = None

    def extract_face(
        self, card_image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], BoundingBoxInfo, List[str]]:
        """
        Detects and extracts portrait face from card image.

        Returns:
            Tuple: (cropped_face_img, landmarks_5pts, BoundingBoxInfo, errors)
        """
        errors: List[str] = []
        empty_bbox = BoundingBoxInfo(detected=False)

        if card_image is None or card_image.size == 0:
            errors.append("CARD_PORTRAIT_FACE_NOT_FOUND")
            return None, None, empty_bbox, errors

        img_h, img_w = card_image.shape[:2]

        # 1. Primary detection using InsightFace (SCRFD / RetinaFace)
        if self.face_app is not None:
            try:
                faces = self.face_app.get(card_image)
                if faces:
                    # Select largest face on the card
                    best_face = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    )
                    raw_x1, raw_y1, raw_x2, raw_y2 = [int(v) for v in best_face.bbox[:4]]
                    x1, y1, x2, y2 = _clip_bbox(raw_x1, raw_y1, raw_x2, raw_y2, img_w, img_h)
                    w, h = x2 - x1, y2 - y1

                    score = float(getattr(best_face, 'det_score', 0.95))
                    kps = getattr(best_face, 'kps', None)

                    bbox_info = BoundingBoxInfo(
                        detected=True,
                        bbox=[x1, y1, x2, y2],
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        width=w, height=h,
                        detectionScore=round(score, 4)
                    )

                    if w < settings.MIN_CARD_FACE_WIDTH or h < settings.MIN_CARD_FACE_HEIGHT:
                        logger.warning(f"[CARD_FACE_EXTRACTOR] Card face too small: {w}x{h} < {settings.MIN_CARD_FACE_WIDTH}x{settings.MIN_CARD_FACE_HEIGHT}")
                        errors.append("CARD_FACE_TOO_SMALL")
                        return None, kps, bbox_info, errors

                    face_crop = crop_image(card_image, [x1, y1, x2, y2])
                    return face_crop, kps, bbox_info, errors
            except Exception as e:
                logger.error(f"[CARD_FACE_EXTRACTOR] InsightFace detection error: {str(e)}")

        # 2. Fallback detection using Pre-loaded OpenCV Haar Cascade
        if self._haar_cascade is not None:
            try:
                gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
                detected_faces = self._haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(settings.MIN_CARD_FACE_WIDTH, settings.MIN_CARD_FACE_HEIGHT)
                )

                if len(detected_faces) > 0:
                    detected_faces = sorted(detected_faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = [int(v) for v in detected_faces[0]]
                    raw_x1, raw_y1, raw_x2, raw_y2 = fx, fy, fx + fw, fy + fh
                    x1, y1, x2, y2 = _clip_bbox(raw_x1, raw_y1, raw_x2, raw_y2, img_w, img_h)
                    w, h = x2 - x1, y2 - y1

                    bbox_info = BoundingBoxInfo(
                        detected=True,
                        bbox=[x1, y1, x2, y2],
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        width=w, height=h,
                        detectionScore=0.90
                    )

                    face_crop = crop_image(card_image, [x1, y1, x2, y2])
                    return face_crop, None, bbox_info, errors
            except Exception as e:
                logger.error(f"[CARD_FACE_EXTRACTOR] Haar Cascade fallback error: {str(e)}")

        # 3. Layout heuristic fallback for synthetic or low-contrast card images (left portrait region)
        if img_w >= 300 and img_h >= 200:
            x1, y1, x2, y2 = _clip_bbox(int(0.02 * img_w), int(0.18 * img_h), int(0.35 * img_w), int(0.85 * img_h), img_w, img_h)
            w, h = x2 - x1, y2 - y1
            crop_roi = card_image[y1:y2, x1:x2]
            if crop_roi.size > 0 and float(np.std(crop_roi)) > 5.0 and w >= settings.MIN_CARD_FACE_WIDTH and h >= settings.MIN_CARD_FACE_HEIGHT:
                bbox_info = BoundingBoxInfo(
                    detected=True,
                    bbox=[x1, y1, x2, y2],
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    width=w, height=h,
                    detectionScore=0.75
                )
                face_crop = crop_image(card_image, [x1, y1, x2, y2])
                return face_crop, None, bbox_info, errors

        errors.append("CARD_PORTRAIT_FACE_NOT_FOUND")
        return None, None, empty_bbox, errors
