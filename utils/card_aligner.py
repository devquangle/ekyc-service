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


def _get_quad_aspect_and_area(pts: np.ndarray) -> Tuple[float, float]:
    """
    Computes aspect ratio and approximate area of 4-point quadrilateral.
    """
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    width = max(float(np.linalg.norm(br - bl)), float(np.linalg.norm(tr - tl)))
    height = max(float(np.linalg.norm(tr - br)), float(np.linalg.norm(tl - bl)))
    if width <= 0 or height <= 0:
        return 0.0, 0.0
    aspect = max(width, height) / min(width, height)
    area = width * height
    return aspect, area


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Applies perspective transform to straighten a quadrilateral region into a rectangle.
    Validates ID-1 card aspect ratio (~1.586) to prevent severe geometric distortions.
    """
    if image is None or pts is None or len(pts) != 4:
        return image

    aspect, area = _get_quad_aspect_and_area(pts)
    h, w = image.shape[:2]

    # Standard ID-1 aspect ratio is 85.6mm / 53.98mm = 1.586
    # Must be between 1.35 and 1.85, and area at least 15% of frame
    if aspect < 1.35 or aspect > 1.85 or area < 0.15 * (w * h):
        logger.debug(f"[CARD_ALIGNER] Skipping warp: abnormal aspect ratio {aspect:.2f} or area {area:.0f}")
        return image

    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 100 or max_height < 60:
        return image

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1],
    ], dtype=np.float32)

    try:
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (max_width, max_height), flags=cv2.INTER_CUBIC)
        # If output is vertical (height > width), rotate 90 degrees to horizontal
        if warped.shape[0] > warped.shape[1]:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        return warped
    except Exception as e:
        logger.warning(f"[CARD_ALIGNER] warpPerspective exception: {str(e)}")
        return image


class CardAligner:
    """
    AI-powered Card Alignment module using YOLOv8-seg ONNX detection with
    OpenCV contour-based fallback. Detects 4 corners of an ID card and applies
    Perspective Transform to produce a straight, cropped card image.
    """

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
        """
        if image is None or image.size == 0:
            return image, False

        try:
            if self.use_model:
                pts = self._detect_with_onnx(image)
                if pts is not None:
                    aligned = _four_point_transform(image, pts)
                    if aligned is not image and aligned.size > 0:
                        logger.info(f"[CARD_ALIGNER] ONNX alignment applied. Output shape: {aligned.shape}")
                        return aligned, True

            # Fallback: OpenCV contour detection
            pts = self._detect_with_contour(image)
            if pts is not None:
                aligned = _four_point_transform(image, pts)
                if aligned is not image and aligned.size > 0:
                    logger.info(f"[CARD_ALIGNER] Contour fallback alignment applied. Output shape: {aligned.shape}")
                    return aligned, True

        except Exception as e:
            logger.warning(f"[CARD_ALIGNER] Alignment failed, returning original: {e}")

        return image, False

    def _detect_with_onnx(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Runs YOLOv8-seg ONNX inference to detect card segmentation mask.
        Extracts quadrilateral corners from the segmentation mask or rotated rectangle.
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

        if not outputs or outputs[0] is None:
            return None

        preds = outputs[0][0]  # shape: [8400, nc+4+32]
        num_classes = max(1, preds.shape[1] - 4 - 32)
        class_scores = preds[:, 4:4 + num_classes]
        conf = np.max(class_scores, axis=1)

        best_idx = int(np.argmax(conf))
        if conf[best_idx] < 0.40:
            return None

        # Try mask reconstruction if proto masks (outputs[1]) are available
        if len(outputs) > 1 and outputs[1] is not None:
            try:
                mask_coeffs = preds[best_idx, 4 + num_classes:]  # [32]
                proto = outputs[1][0]  # [32, 160, 160]
                c, mh, mw = proto.shape
                proto_flat = proto.reshape(c, mh * mw)
                mask_flat = np.dot(mask_coeffs, proto_flat)
                mask = 1.0 / (1.0 + np.exp(-mask_flat))  # sigmoid
                mask = mask.reshape(mh, mw)
                mask_bin = (mask > 0.5).astype(np.uint8) * 255
                mask_full = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

                contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(max_c) > 0.15 * (w * h):
                        rot_rect = cv2.minAreaRect(max_c)
                        box_pts = cv2.boxPoints(rot_rect).astype(np.float32)
                        return box_pts
            except Exception as e:
                logger.debug(f"[CARD_ALIGNER] Proto mask parsing fallback: {e}")

        # Fallback to bounding box prediction
        cx, cy, bw, bh = preds[best_idx, :4]
        x1 = int((cx - bw / 2) * w / self.INPUT_SIZE[0])
        y1 = int((cy - bh / 2) * h / self.INPUT_SIZE[1])
        x2 = int((cx + bw / 2) * w / self.INPUT_SIZE[0])
        y2 = int((cy + bh / 2) * h / self.INPUT_SIZE[1])

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if (x2 - x1) * (y2 - y1) < 0.15 * w * h:
            return None

        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        return pts

    def _detect_with_contour(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects card boundary using OpenCV edge detection and contour approximation.
        Finds the largest quadrilateral contour matching ID-1 card aspect ratio.
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
            if area < 0.20 * image_area:
                continue

            # 1. Try minAreaRect
            rot_rect = cv2.minAreaRect(contour)
            box_w, box_h = rot_rect[1]
            if box_w > 0 and box_h > 0:
                rect_area = box_w * box_h
                if rect_area >= 0.20 * image_area:
                    ratio = max(box_w, box_h) / min(box_w, box_h)
                    if 1.35 <= ratio <= 1.85:
                        box_pts = cv2.boxPoints(rot_rect).astype(np.float32)
                        return box_pts

            # 2. Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                aspect, q_area = _get_quad_aspect_and_area(pts)
                if q_area >= 0.20 * image_area and 1.35 <= aspect <= 1.85:
                    return pts

        return None
