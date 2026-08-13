import cv2
import numpy as np
from typing import Tuple, Optional, List
from config import settings
from utils.logger import logger


def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decodes raw image byte array into an OpenCV BGR numpy array.
    """
    if not image_bytes:
        return None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode image bytes: {str(e)}")
        return None


def check_image_quality(image: np.ndarray) -> Tuple[bool, bool, bool]:
    """
    Evaluates image blur using Laplacian Variance, glare using HSV thresholding,
    and cropped status using margin boundary detection.
    Returns: (is_blur, has_glare, is_cropped)
    """
    if image is None or image.size == 0:
        return True, False, True

    # 1. Blur detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blur = laplacian_var < settings.IMAGE_BLUR_THRESHOLD

    # 2. Glare detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else image
    v_channel = hsv[:, :, 2] if len(hsv.shape) == 3 else hsv
    glare_pixels = np.sum(v_channel > 250)
    total_pixels = v_channel.size
    glare_ratio = glare_pixels / float(total_pixels)
    has_glare = glare_ratio > 0.05

    # 3. Cropped detection (margin boundary threshold)
    h, w = gray.shape[:2]
    border_margin = int(min(h, w) * 0.02)
    top_edge = gray[0:border_margin, :]
    bottom_edge = gray[h - border_margin:h, :]
    left_edge = gray[:, 0:border_margin]
    right_edge = gray[:, w - border_margin:w]

    # High contrast gradients touching extreme border indicate image crop
    is_cropped = bool(
        np.std(top_edge) > 60 or np.std(bottom_edge) > 60 or np.std(left_edge) > 60 or np.std(right_edge) > 60
    )

    return is_blur, has_glare, is_cropped


def crop_image(image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
    """
    Crops a region of interest [x1, y1, x2, y2] safely from image.
    """
    if image is None or not bbox or len(bbox) < 4:
        return None

    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    cropped = image[y1:y2, x1:x2]
    return cropped if cropped.size > 0 else None


def resize_maintain_aspect(image: np.ndarray, target_width: int = 1024) -> np.ndarray:
    """
    Resizes image maintaining aspect ratio to standard target width.
    """
    if image is None:
        return image
    h, w = image.shape[:2]
    if w == target_width or w == 0:
        return image
    scale = target_width / float(w)
    target_height = int(h * scale)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
