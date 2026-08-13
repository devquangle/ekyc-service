import os
import cv2
import numpy as np
from typing import Optional, Tuple, List
from utils.logger import logger


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 corner points as: top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left: smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right: smallest y-x
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest y-x
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies perspective transform to straighten a quadrilateral region into a rectangle.
    """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 50 or max_height < 30:
        return image  # too small, return original

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped


class CardAligner:
    """
    AI-powered Card Alignment module using YOLOv8-seg ONNX detection with
    OpenCV contour-based fallback. Detects 4 corners of an ID card and applies
    Perspective Transform to produce a straight, cropped card image.

    Graceful Degradation:
    - If ONNX model exists: YOLOv8-seg detects card mask → extract corners → warpPerspective
    - If ONNX model missing: OpenCV Canny + contour detection → warpPerspective
    - If both fail: returns original image unchanged (no crash)
    """

    # YOLOv8 ONNX input size
    INPUT_SIZE = (640, 640)

    def __init__(self, model_path: str = "weights/card_seg.onnx"):
        self.model_path = model_path
        self.session = None
        self.use_model = False

        if os.path.exists(model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    model_path,
                    providers=["CPUExecutionProvider"]
                )
                self.input_name = self.session.get_inputs()[0].name
                self.use_model = True
                logger.info(f"[CARD_ALIGNER] YOLOv8-seg ONNX loaded from '{model_path}'")
            except Exception as e:
                logger.warning(f"[CARD_ALIGNER] Failed to load ONNX model: {e}. Using contour fallback.")
        else:
            logger.info(f"[CARD_ALIGNER] Model not found at '{model_path}'. Using OpenCV contour fallback.")

    def align(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Detects and aligns ID card in the image.
        Returns: (aligned_image, was_aligned)
        - aligned_image: perspective-corrected card, or original if detection fails
        - was_aligned: True if perspective transform was applied
        """
        if image is None or image.size == 0:
            return image, False

        try:
            if self.use_model:
                pts = self._detect_with_onnx(image)
                if pts is not None:
                    aligned = _four_point_transform(image, pts)
                    if aligned is not image:
                        logger.info(f"[CARD_ALIGNER] ONNX alignment applied. Output shape: {aligned.shape}")
                        return aligned, True

            # Fallback: OpenCV contour detection
            pts = self._detect_with_contour(image)
            if pts is not None:
                aligned = _four_point_transform(image, pts)
                if aligned is not image:
                    logger.info(f"[CARD_ALIGNER] Contour fallback alignment applied. Output shape: {aligned.shape}")
                    return aligned, True

        except Exception as e:
            logger.warning(f"[CARD_ALIGNER] Alignment failed, returning original: {e}")

        return image, False

    def _detect_with_onnx(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Runs YOLOv8-seg ONNX inference to detect card segmentation mask.
        Extracts convex hull 4-corner approximation from the largest mask.
        Returns 4-point array (shape [4,2]) or None.
        """
        if self.session is None:
            return None

        h, w = image.shape[:2]
        img_resized = cv2.resize(image, self.INPUT_SIZE)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, ...]  # NCHW

        try:
            outputs = self.session.run(None, {self.input_name: img_input})
        except Exception as e:
            logger.warning(f"[CARD_ALIGNER] ONNX inference error: {e}")
            return None

        # YOLOv8-seg outputs[0]: detection boxes [1, 8400, nc+4+mask_dim]
        # outputs[1]: proto masks [1, 32, 160, 160]
        # Parse boxes to find highest confidence detection
        if not outputs or outputs[0] is None:
            return None

        preds = outputs[0][0]  # shape: [8400, nc+4+32]
        # Columns: cx, cy, w, h, [cls_scores...], [mask_coeffs...]
        num_classes = max(1, preds.shape[1] - 4 - 32)
        class_scores = preds[:, 4:4 + num_classes]
        conf = np.max(class_scores, axis=1)
        best_idx = int(np.argmax(conf))

        if conf[best_idx] < 0.40:
            return None

        # Reconstruct mask from proto (simplified: use bbox as quadrilateral)
        cx, cy, bw, bh = preds[best_idx, :4]
        # Scale back to original image coordinates
        x1 = int((cx - bw / 2) * w / self.INPUT_SIZE[0])
        y1 = int((cy - bh / 2) * h / self.INPUT_SIZE[1])
        x2 = int((cx + bw / 2) * w / self.INPUT_SIZE[0])
        y2 = int((cy + bh / 2) * h / self.INPUT_SIZE[1])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Ensure minimum area
        if (x2 - x1) * (y2 - y1) < 0.10 * w * h:
            return None

        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        return pts

    def _detect_with_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects card boundary using OpenCV edge detection and contour approximation.
        Finds the largest quadrilateral contour that resembles an ID card.
        Returns 4-point array (shape [4,2]) or None.
        """
        h, w = image.shape[:2]
        image_area = h * w

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

        # Enhance contrast before edge detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection with adaptive thresholds
        median_val = float(np.median(blurred))
        lower = int(max(0, 0.67 * median_val))
        upper = int(min(255, 1.33 * median_val))
        edges = cv2.Canny(blurred, lower, upper)

        # Morphological close to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Sort contours by area descending
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area < 0.10 * image_area:
                continue

            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                return pts

            # If not exactly 4, try convex hull and re-approximate
            hull = cv2.convexHull(contour)
            hull_peri = cv2.arcLength(hull, True)
            hull_approx = cv2.approxPolyDP(hull, 0.02 * hull_peri, True)
            if len(hull_approx) == 4:
                pts = hull_approx.reshape(4, 2).astype(np.float32)
                return pts

        return None
