import os
import cv2
import numpy as np
from typing import Optional, List, Tuple
from utils.logger import logger


class AntiSpoofEngine:
    """
    Deep Learning Anti-Spoofing Engine using MiniFASNet ONNX models (Silent-Face-Anti-Spoofing).
    Distinguishes Real Faces from Presentation Attacks (Screen replay, Print paper, 3D masks).

    MiniFASNet Standard Inference Protocol:
    - Input: Face ROI expanded by scale factor (2.7x - 4.0x), resized to (80, 80).
    - Tensor: (1, 3, 80, 80) float32 in [0, 1] RGB, NCHW.
    - Output: Softmax (1, 3) -> [background, real, fake]. Index 1 represents Real Face Probability.
    """

    INPUT_SIZE = (80, 80)

    def __init__(self, model_path_1: str, model_path_2: str):
        self.sessions = []
        self.model_scales = []

        # Pre-load face cascade detector once for ROI localization
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.warning(f"[ANTI_SPOOF] Failed to load face cascade: {str(e)}")
            self._face_cascade = None

        # Load MiniFASNet Model 1 (scale 2.7x)
        self._load_model(model_path_1, scale=2.7)
        # Load MiniFASNet Model 2 (scale 4.0x)
        self._load_model(model_path_2, scale=4.0)

        if self.sessions:
            logger.info(f"[ANTI_SPOOF] Loaded {len(self.sessions)} MiniFASNet ONNX model(s).")
        else:
            logger.info("[ANTI_SPOOF] No ONNX models found. Using enhanced multi-metric texture fallback.")

    def _load_model(self, model_path: str, scale: float = 2.7) -> None:
        if not model_path or not os.path.exists(model_path):
            return
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.sessions.append(sess)
            self.model_scales.append(scale)
            logger.info(f"[ANTI_SPOOF] Loaded model: '{model_path}' with scale {scale}x")
        except Exception as e:
            logger.warning(f"[ANTI_SPOOF] Failed to load '{model_path}': {str(e)}")

    def predict(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> float:
        """
        Predicts anti-spoofing real-face probability score for a video frame.

        Args:
            frame: Input video frame (BGR).
            face_bbox: Optional face bounding box (x1, y1, x2, y2). If None, auto-detected.

        Returns:
            float in [0.0, 1.0] representing real person confidence.
        """
        if frame is None or frame.size == 0:
            return 0.0

        # Locate face bbox if not provided
        if face_bbox is None and self._face_cascade is not None:
            face_bbox = self._detect_face_bbox(frame)

        if self.sessions:
            return self._ensemble_predict(frame, face_bbox)

        # Fallback to multi-metric texture analysis on face ROI or full frame
        roi = self._crop_expanded_roi(frame, face_bbox, scale=1.2) if face_bbox else frame
        return self._fallback_texture_score(roi)

    def _detect_face_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
            if len(faces) > 0:
                # Select largest face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                fx, fy, fw, fh = faces[0]
                return int(fx), int(fy), int(fx + fw), int(fy + fh)
        except Exception as e:
            logger.debug(f"[ANTI_SPOOF] Face detection exception: {str(e)}")
        return None

    def _crop_expanded_roi(
        self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], scale: float = 2.7
    ) -> np.ndarray:
        """
        Crops face ROI expanded by given scale factor according to MiniFASNet specifications.
        """
        if bbox is None:
            return frame

        img_h, img_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale

        # Keep square aspect ratio for MiniFASNet
        max_dim = max(w, h)
        nx1 = max(0, int(cx - max_dim / 2.0))
        ny1 = max(0, int(cy - max_dim / 2.0))
        nx2 = min(img_w, int(cx + max_dim / 2.0))
        ny2 = min(img_h, int(cy + max_dim / 2.0))

        if nx2 <= nx1 or ny2 <= ny1:
            return frame

        crop = frame[ny1:ny2, nx1:nx2]
        return crop if crop.size > 0 else frame

    def _ensemble_predict(self, frame: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> float:
        scores = []
        for sess, scale in zip(self.sessions, self.model_scales):
            try:
                roi = self._crop_expanded_roi(frame, face_bbox, scale=scale)
                score = self._predict_single(sess, roi)
                scores.append(score)
            except Exception as e:
                logger.warning(f"[ANTI_SPOOF] MiniFASNet inference error: {str(e)}")

        if not scores:
            roi = self._crop_expanded_roi(frame, face_bbox, scale=1.2) if face_bbox else frame
            return self._fallback_texture_score(roi)

        return float(np.mean(scores))

    def _predict_single(self, session, face_roi: np.ndarray) -> float:
        """
        Preprocesses face ROI and runs single MiniFASNet ONNX inference.
        Returns Softmax index 1 (real probability).
        """
        resized = cv2.resize(face_roi, self.INPUT_SIZE, interpolation=cv2.INTER_AREA)
        if len(resized.shape) == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 80, 80)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})

        if not outputs or outputs[0] is None:
            return 0.0

        logits = outputs[0][0]  # shape: (3,) -> [background, real, fake]

        # Numerically stable Softmax
        exp_logits = np.exp(logits - np.max(logits))
        softmax = exp_logits / (np.sum(exp_logits) + 1e-8)

        # Index 1 = Real Face Probability
        real_prob = float(softmax[1]) if len(softmax) > 1 else float(softmax[0])
        return min(1.0, max(0.0, real_prob))

    def _fallback_texture_score(self, roi: np.ndarray) -> float:
        """
        Multi-metric texture analysis when ONNX models are absent:
        1. FFT Frequency Spectrum Variance (High frequency micro-texture).
        2. Laplacian Edge Sharpness.
        3. Local Binary Pattern (LBP) Texture Uniformity.
        4. Specular Glare / Screen Moiré Penalty.
        """
        if roi is None or roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi.copy()

        if float(np.std(gray)) < 5.0:
            return 0.0

        # 1. FFT Frequency Variance
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20.0 * np.log(np.abs(fshift) + 1e-8)
        freq_var = float(np.var(magnitude_spectrum))
        fft_score = min(1.0, max(0.0, freq_var / 300.0))

        # 2. Laplacian Sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(laplacian))
        lap_score = min(1.0, max(0.0, lap_var / 500.0))

        # 3. LBP Texture Uniformity
        lbp_score = self._compute_lbp_score(gray)

        # 4. Specular Screen Reflection Penalty
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        bright_ratio = float(np.sum(bright_mask > 0)) / max(1, gray.size)
        glare_score = max(0.0, 1.0 - min(1.0, bright_ratio * 10.0))

        score = (0.35 * fft_score + 0.30 * lap_score + 0.25 * lbp_score + 0.10 * glare_score)
        return min(1.0, max(0.0, float(score)))

    def _compute_lbp_score(self, gray: np.ndarray) -> float:
        try:
            h, w = gray.shape
            if h < 10 or w < 10:
                return 0.5

            center = gray[1:h-1, 1:w-1]
            lbp = np.zeros_like(center, dtype=np.uint8)

            neighbors = [
                gray[0:h-2, 0:w-2], gray[0:h-2, 1:w-1], gray[0:h-2, 2:w],
                gray[1:h-1, 2:w], gray[2:h, 2:w], gray[2:h, 1:w-1],
                gray[2:h, 0:w-2], gray[1:h-1, 0:w-2]
            ]

            for i, n in enumerate(neighbors):
                lbp |= ((n >= center).astype(np.uint8) << i)

            hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
            non_zero_hist = hist[hist > 0]
            entropy = -float(np.sum(non_zero_hist * np.log2(non_zero_hist)))
            return min(1.0, max(0.0, entropy / 7.0))
        except Exception:
            return 0.5
