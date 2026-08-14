import os
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from utils.logger import logger


def crop_face_roi_with_scale(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    scale: float = 2.7,
    target_size: Tuple[int, int] = (80, 80)
) -> Optional[np.ndarray]:
    """
    Crops an expanded Face ROI around the face bounding box for MiniFASNet anti-spoofing models.
    Preserves context (2.7x or 4.0x face scale) and handles boundary padding with BORDER_REFLECT.

    Args:
        image: Original BGR frame.
        bbox: Face bounding box (x1, y1, x2, y2).
        scale: Expansion scale factor (e.g. 2.7 for Model 1, 4.0 for Model 2).
        target_size: Desired model input size (80, 80).

    Returns:
        Expanded and resized face ROI of shape (target_size[1], target_size[0], 3).
    """
    if image is None or image.size == 0 or not bbox:
        return None

    img_h, img_w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    # Expand box by scale
    scaled_w = w * scale
    scaled_h = h * scale

    nx1 = int(round(cx - scaled_w / 2.0))
    ny1 = int(round(cy - scaled_h / 2.0))
    nx2 = int(round(cx + scaled_w / 2.0))
    ny2 = int(round(cy + scaled_h / 2.0))

    # Calculate padding if expanded bbox extends beyond frame boundaries
    pad_left = max(0, -nx1)
    pad_top = max(0, -ny1)
    pad_right = max(0, nx2 - img_w)
    pad_bottom = max(0, ny2 - img_h)

    # Pad image if needed
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded_img = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_REFLECT
        )
        crop_x1 = nx1 + pad_left
        crop_y1 = ny1 + pad_top
        crop_x2 = nx2 + pad_left
        crop_y2 = ny2 + pad_top
        face_roi = padded_img[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        face_roi = image[ny1:ny2, nx1:nx2]

    if face_roi.size == 0:
        return None

    resized_roi = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
    return resized_roi


class AntiSpoofEngine:
    """
    Deep Learning Anti-Spoofing Engine using MiniFASNet ONNX models (Silent-Face-Anti-Spoofing).
    Accurately detects Screen Replay, Print Attacks, and 3D Masks from expanded Face ROIs.

    Ensemble strategy:
    - Multi-scale MiniFASNet models (Scale 2.7 & Scale 4.0).
    - Face detection & expansion before tensor normalization.
    - Softmax real-face classification (Index 1 = Real Face probability).
    - Multi-metric texture fallback (FFT + Laplacian + LBP + Glare analysis).
    """

    INPUT_SIZE = (80, 80)
    DEFAULT_SCALES = [2.7, 4.0]

    def __init__(self, model_path_1: str, model_path_2: str):
        self.sessions: List[Tuple[Any, float]] = []

        # Load MiniFASNet ONNX sessions with their corresponding expansion scales
        self._load_model(model_path_1, scale=2.7)
        self._load_model(model_path_2, scale=4.0)

        # Pre-load OpenCV Haar Cascade detector once for standalone face localization
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"[ANTI_SPOOF] Failed to initialize Haar Cascade: {e}")
            self._haar_cascade = None

        if self.sessions:
            logger.info(f"[ANTI_SPOOF] Initialized {len(self.sessions)} MiniFASNet ONNX model(s).")
        else:
            logger.info("[ANTI_SPOOF] No ONNX models found. Operating on enhanced FFT+LBP+Laplacian fallback.")

    def _load_model(self, model_path: str, scale: float) -> None:
        if not model_path or not os.path.exists(model_path):
            logger.info(f"[ANTI_SPOOF] Model not found: '{model_path}' — skipping.")
            return
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.sessions.append((sess, scale))
            logger.info(f"[ANTI_SPOOF] Loaded MiniFASNet model (scale={scale}): '{model_path}'")
        except Exception as e:
            logger.warning(f"[ANTI_SPOOF] Failed to load ONNX model '{model_path}': {e}")

    def predict(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> float:
        """
        Predicts anti-spoofing liveness score for a single video frame.

        Args:
            frame: Full BGR frame or face crop.
            face_bbox: Optional bounding box (x1, y1, x2, y2). If None, automatic detection is performed.

        Returns:
            Liveness confidence score in [0.0, 1.0] (higher score = genuine live person).
        """
        if frame is None or frame.size == 0:
            return 0.0

        # Auto-detect face bbox if not provided
        resolved_bbox = face_bbox
        if resolved_bbox is None and self._haar_cascade is not None:
            resolved_bbox = self._detect_largest_face(frame)

        # 1. Deep Learning MiniFASNet Prediction
        if self.sessions and resolved_bbox is not None:
            return self._ensemble_predict(frame, resolved_bbox)

        # 2. Multi-Metric Texture Fallback
        target_roi = frame
        if resolved_bbox is not None:
            x1, y1, x2, y2 = resolved_bbox
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size > 0:
                target_roi = crop

        return self._fallback_texture_score(target_roi)

    def _detect_largest_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            faces = self._haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                return int(fx), int(fy), int(fx + fw), int(fy + fh)
        except Exception:
            pass
        return None

    def _ensemble_predict(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> float:
        """
        Runs all loaded MiniFASNet models with their respective scale factors and returns average score.
        """
        scores: List[float] = []
        for sess, scale in self.sessions:
            try:
                score = self._predict_single(sess, frame, face_bbox, scale)
                scores.append(score)
            except Exception as e:
                logger.warning(f"[ANTI_SPOOF] Model inference error (scale={scale}): {e}")

        if not scores:
            return self._fallback_texture_score(frame)

        return float(np.mean(scores))

    def _predict_single(
        self,
        session: Any,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
        scale: float
    ) -> float:
        """
        Preprocesses expanded face ROI and executes MiniFASNet ONNX inference.
        Returns real-face probability (Softmax Index 1).
        """
        roi = crop_face_roi_with_scale(frame, face_bbox, scale=scale, target_size=self.INPUT_SIZE)
        if roi is None:
            roi = cv2.resize(frame, self.INPUT_SIZE)

        # Preprocess: BGR -> RGB -> Float32 [0.0, 1.0] -> NCHW (1, 3, 80, 80)
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB) if len(roi.shape) == 3 else cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})

        if not outputs or outputs[0] is None:
            return 0.5

        logits = np.squeeze(outputs[0])  # Shape: (3,) -> [background, real, fake]

        # Numerical stable Softmax
        exp_logits = np.exp(logits - np.max(logits))
        softmax = exp_logits / (np.sum(exp_logits) + 1e-8)

        # Index 1 corresponds to Real Face probability in standard MiniFASNet
        real_prob = float(softmax[1]) if len(softmax) > 1 else float(softmax[0])
        return min(1.0, max(0.0, real_prob))

    def _fallback_texture_score(self, face_img: np.ndarray) -> float:
        """
        Enhanced multi-metric texture analysis when no ONNX model is available.
        Combines FFT high-frequency distribution, Laplacian micro-sharpness, LBP richness, and glare penalty.
        """
        if face_img is None or face_img.size == 0:
            return 0.0

        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img.copy()

        if float(np.std(gray)) < 5.0:
            return 0.0

        # Metric 1: 2D FFT Frequency Variance
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        freq_var = float(np.var(magnitude_spectrum))
        fft_score = min(1.0, max(0.0, freq_var / 300.0))

        # Metric 2: Laplacian Variance (High frequency edge micro-texture)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(laplacian))
        lap_score = min(1.0, max(0.0, lap_var / 500.0))

        # Metric 3: Local Binary Pattern (LBP) Uniformity & Entropy
        lbp_score = self._compute_lbp_score(gray)

        # Metric 4: Specular Glare / Screen Highlight Penalty
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        bright_ratio = float(np.sum(bright_mask > 0)) / max(1, gray.size)
        glare_penalty = min(1.0, bright_ratio * 10.0)
        glare_score = 1.0 - glare_penalty

        # Weighted Ensemble Texture Score
        score = (0.35 * fft_score + 0.30 * lap_score + 0.25 * lbp_score + 0.10 * glare_score)
        return min(1.0, max(0.0, float(score)))

    def _compute_lbp_score(self, gray: np.ndarray) -> float:
        try:
            radius = 1
            h, w = gray.shape
            if h <= 2 * radius or w <= 2 * radius:
                return 0.5

            center = gray[radius:h - radius, radius:w - radius]
            neighbors = [
                gray[0:h - 2 * radius, 0:w - 2 * radius],
                gray[0:h - 2 * radius, radius:w - radius],
                gray[0:h - 2 * radius, 2 * radius:w],
                gray[radius:h - radius, 2 * radius:w],
                gray[2 * radius:h, 2 * radius:w],
                gray[2 * radius:h, radius:w - radius],
                gray[2 * radius:h, 0:w - 2 * radius],
                gray[radius:h - radius, 0:w - 2 * radius],
            ]

            lbp_center = np.zeros(center.shape, dtype=np.uint8)
            for i, neighbor in enumerate(neighbors):
                lbp_center += ((neighbor >= center).astype(np.uint8)) << i

            hist, _ = np.histogram(lbp_center, bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-8)

            entropy = -float(np.sum(hist * np.log2(hist + 1e-8)))
            lbp_score = min(1.0, max(0.0, entropy / 8.0))
            return lbp_score
        except Exception:
            return 0.5
