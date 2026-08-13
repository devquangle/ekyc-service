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
    Ensures width (x2 - x1) > 0 and height (y2 - y1) > 0.
    """
    cx1 = max(0, min(x1, img_w - 1))
    cy1 = max(0, min(y1, img_h - 1))
    cx2 = max(cx1 + 1, min(x2, img_w))
    cy2 = max(cy1 + 1, min(y2, img_h))
    return cx1, cy1, cx2, cy2


class SelfieFaceExtractor:
    """
    Selfie face detector and extractor.
    Enforces exactly 1 detected face, minimum size check, and bounding box metrics.
    """

    def __init__(self, face_app=None):
        self.face_app = face_app

    def extract_face(self, selfie_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], BoundingBoxInfo, List[str]]:
        """
        Detects and extracts face from selfie image.
        Returns: (cropped_face_img, landmarks, BoundingBoxInfo, errors)
        """
        errors: List[str] = []
        empty_bbox = BoundingBoxInfo(detected=False)

        if selfie_image is None or selfie_image.size == 0:
            errors.append("SELFIE_FACE_NOT_FOUND")
            return None, None, empty_bbox, errors

        img_h, img_w = selfie_image.shape[:2]

        # 1. InsightFace primary detector
        if self.face_app is not None:
            try:
                faces = self.face_app.get(selfie_image)
                if not faces:
                    errors.append("SELFIE_FACE_NOT_FOUND")
                    return None, None, empty_bbox, errors

                if len(faces) > 1:
                    logger.warning(f"Multiple faces detected in selfie ({len(faces)} faces found).")
                    errors.append("MULTIPLE_FACES_DETECTED")
                    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    raw_x1, raw_y1, raw_x2, raw_y2 = [int(v) for v in best_face.bbox[:4]]
                    x1, y1, x2, y2 = _clip_bbox(raw_x1, raw_y1, raw_x2, raw_y2, img_w, img_h)
                    w, h = x2 - x1, y2 - y1

                    score = float(getattr(best_face, 'det_score', 0.95))
                    kps = getattr(best_face, 'kps', None)
                    bbox_info = BoundingBoxInfo(
                        detected=True, bbox=[x1, y1, x2, y2], x1=x1, y1=y1, x2=x2, y2=y2, width=w, height=h, detectionScore=score
                    )
                    return None, kps, bbox_info, errors

                face = faces[0]
                raw_x1, raw_y1, raw_x2, raw_y2 = [int(v) for v in face.bbox[:4]]
                x1, y1, x2, y2 = _clip_bbox(raw_x1, raw_y1, raw_x2, raw_y2, img_w, img_h)
                w, h = x2 - x1, y2 - y1

                score = float(getattr(face, 'det_score', 0.95))
                kps = getattr(face, 'kps', None)

                bbox_info = BoundingBoxInfo(
                    detected=True, bbox=[x1, y1, x2, y2], x1=x1, y1=y1, x2=x2, y2=y2, width=w, height=h, detectionScore=score
                )

                if w < settings.MIN_SELFIE_FACE_WIDTH or h < settings.MIN_SELFIE_FACE_HEIGHT:
                    logger.warning(f"Selfie face crop too small: {w}x{h} < {settings.MIN_SELFIE_FACE_WIDTH}x{settings.MIN_SELFIE_FACE_HEIGHT}")
                    errors.append("SELFIE_FACE_TOO_SMALL")
                    return None, kps, bbox_info, errors

                face_crop = crop_image(selfie_image, [x1, y1, x2, y2])
                return face_crop, kps, bbox_info, errors
            except Exception as e:
                logger.error(f"InsightFace selfie face extraction error: {str(e)}")

        # 2. Fallback OpenCV Haar Cascade
        try:
            gray = cv2.cvtColor(selfie_image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            detected_faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            if len(detected_faces) == 0:
                errors.append("SELFIE_FACE_NOT_FOUND")
                return None, None, empty_bbox, errors

            if len(detected_faces) > 1:
                logger.warning(f"Multiple faces detected in selfie ({len(detected_faces)} faces found).")
                errors.append("MULTIPLE_FACES_DETECTED")
                fx, fy, fw, fh = [int(v) for v in detected_faces[0]]
                raw_x1, raw_y1, raw_x2, raw_y2 = fx, fy, fx + fw, fy + fh
                x1, y1, x2, y2 = _clip_bbox(raw_x1, raw_y1, raw_x2, raw_y2, img_w, img_h)
                w, h = x2 - x1, y2 - y1

                bbox_info = BoundingBoxInfo(
                    detected=True, bbox=[x1, y1, x2, y2], x1=x1, y1=y1, x2=x2, y2=y2, width=w, height=h, detectionScore=0.90
                )
                return None, None, bbox_info, errors

            fx, fy, fw, fh = [int(v) for v in detected_faces[0]]
            raw_x1, raw_y1, raw_x2, raw_y2 = fx, fy, fx + fw, fy + fh
            x1, y1, x2, y2 = _clip_bbox(raw_x1, raw_y1, raw_x2, raw_y2, img_w, img_h)
            w, h = x2 - x1, y2 - y1

            bbox_info = BoundingBoxInfo(
                detected=True, bbox=[x1, y1, x2, y2], x1=x1, y1=y1, x2=x2, y2=y2, width=w, height=h, detectionScore=0.90
            )

            if w < settings.MIN_SELFIE_FACE_WIDTH or h < settings.MIN_SELFIE_FACE_HEIGHT:
                errors.append("SELFIE_FACE_TOO_SMALL")
                return None, None, bbox_info, errors

            face_crop = crop_image(selfie_image, [x1, y1, x2, y2])
            return face_crop, None, bbox_info, errors
        except Exception as e:
            logger.error(f"Fallback Haar Cascade selfie face extraction error: {str(e)}")

        # Fallback if image itself is face size
        if img_w >= settings.MIN_SELFIE_FACE_WIDTH and img_h >= settings.MIN_SELFIE_FACE_HEIGHT:
            bbox_info = BoundingBoxInfo(
                detected=True, bbox=[0, 0, img_w, img_h], x1=0, y1=0, x2=img_w, y2=img_h, width=img_w, height=img_h, detectionScore=0.80
            )
            return selfie_image, None, bbox_info, errors

        errors.append("SELFIE_FACE_NOT_FOUND")
        return None, None, empty_bbox, errors
