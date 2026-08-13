import os
import cv2
import numpy as np
from typing import Optional
from utils.logger import logger


class AntiSpoofEngine:
    """
    Deep Learning Anti-Spoofing Engine using MiniFASNet ONNX models (Silent-Face-Anti-Spoofing).
    Distinguishes: Real face vs Screen replay vs Print attack vs 3D mask.

    Ensemble strategy:
    - If both MiniFASNet ONNX models loaded: weighted average of 2 model predictions
    - If only 1 model loaded: single model score
    - If no models: Enhanced FFT + LBP + Laplacian texture fallback (better than FFT-only)

    MiniFASNet model inputs: (1, 3, 80, 80) float32 normalized [0,1] RGB
    MiniFASNet model outputs: (1, 3) softmax — [background, real, fake]
                               index 1 = real probability
    """

    INPUT_SIZE = (80, 80)

    def __init__(self, model_path_1: str, model_path_2: str):
        self.sessions = []
        self._load_model(model_path_1)
        self._load_model(model_path_2)

        if self.sessions:
            logger.info(f"[ANTI_SPOOF] Loaded {len(self.sessions)} MiniFASNet ONNX model(s).")
        else:
            logger.info("[ANTI_SPOOF] No ONNX models found. Using enhanced FFT+LBP texture fallback.")

    def _load_model(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            logger.info(f"[ANTI_SPOOF] Model not found: '{model_path}' — skipping.")
            return
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.sessions.append(sess)
            logger.info(f"[ANTI_SPOOF] Loaded: '{model_path}'")
        except Exception as e:
            logger.warning(f"[ANTI_SPOOF] Failed to load '{model_path}': {e}")

    def predict(self, frame: np.ndarray) -> float:
        """
        Predicts liveness score for a single video frame.
        Returns: float in [0.0, 1.0] — higher = more likely real person.
        """
        if frame is None or frame.size == 0:
            return 0.0

        if self.sessions:
            return self._ensemble_predict(frame)

        return self._fallback_texture_score(frame)

    def _ensemble_predict(self, frame: np.ndarray) -> float:
        """
        Runs all loaded MiniFASNet models and returns averaged real-face probability.
        """
        scores = []
        for sess in self.sessions:
            try:
                score = self._predict_single(sess, frame)
                scores.append(score)
            except Exception as e:
                logger.warning(f"[ANTI_SPOOF] Inference error: {e}")

        if not scores:
            return self._fallback_texture_score(frame)

        return float(np.mean(scores))

    def _predict_single(self, session, frame: np.ndarray) -> float:
        """
        Preprocesses frame and runs a single MiniFASNet ONNX inference.
        Returns real-face probability (softmax index 1).
        """
        # Preprocess: resize, BGR→RGB, normalize [0,1], NHWC→NCHW
        resized = cv2.resize(frame, self.INPUT_SIZE)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if len(resized.shape) == 3 else cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        input_tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 80, 80)

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_tensor})

        if not outputs or outputs[0] is None:
            return 0.5

        logits = outputs[0][0]  # shape: (3,) — [background, real, fake]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        softmax = exp_logits / (exp_logits.sum() + 1e-8)

        # Index 1 = real probability
        real_prob = float(softmax[1]) if len(softmax) > 1 else float(softmax[0])
        return min(1.0, max(0.0, real_prob))

    def _fallback_texture_score(self, frame: np.ndarray) -> float:
        """
        Enhanced multi-metric texture analysis when no ONNX model is available.
        Combines FFT frequency variance, Laplacian sharpness, and LBP texture richness.
        Much more robust than the original single FFT heuristic.
        """
        if frame is None or frame.size == 0:
            return 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame.copy()

        if np.std(gray) < 5.0:
            return 0.0

        # --- Metric 1: FFT Frequency Variance ---
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        freq_var = float(np.var(magnitude_spectrum))
        fft_score = min(1.0, max(0.0, freq_var / 300.0))

        # --- Metric 2: Laplacian Sharpness (real faces have more micro-texture detail) ---
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(laplacian))
        lap_score = min(1.0, max(0.0, lap_var / 500.0))

        # --- Metric 3: Local Binary Pattern (LBP) Texture Richness ---
        lbp_score = self._compute_lbp_score(gray)

        # --- Metric 4: Specular Highlight Detection (screens have uniform bright spots) ---
        _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        bright_ratio = float(np.sum(bright_mask > 0)) / max(1, gray.size)
        # Many bright pixels = likely screen reflection → lower score
        glare_penalty = min(1.0, bright_ratio * 10.0)
        glare_score = 1.0 - glare_penalty

        # Weighted ensemble of metrics
        score = (0.35 * fft_score + 0.30 * lap_score + 0.25 * lbp_score + 0.10 * glare_score)
        return min(1.0, max(0.0, score))

    def _compute_lbp_score(self, gray: np.ndarray) -> float:
        """
        Computes Local Binary Pattern uniformity as a texture richness metric.
        Real faces have richer, more varied LBP distributions than printed/screen faces.
        """
        try:
            radius = 1
            n_points = 8
            h, w = gray.shape

            # Compute LBP manually using OpenCV (no scikit-image required)
            lbp = np.zeros_like(gray, dtype=np.uint8)
            center = gray[radius:h-radius, radius:w-radius]

            neighbors = [
                gray[0:h-2*radius, 0:w-2*radius],       # top-left
                gray[0:h-2*radius, radius:w-radius],     # top
                gray[0:h-2*radius, 2*radius:w],          # top-right
                gray[radius:h-radius, 2*radius:w],       # right
                gray[2*radius:h, 2*radius:w],            # bottom-right
                gray[2*radius:h, radius:w-radius],       # bottom
                gray[2*radius:h, 0:w-2*radius],          # bottom-left
                gray[radius:h-radius, 0:w-2*radius],     # left
            ]

            lbp_center = np.zeros(center.shape, dtype=np.uint8)
            for i, neighbor in enumerate(neighbors):
                lbp_center += ((neighbor >= center).astype(np.uint8)) << i

            hist, _ = np.histogram(lbp_center, bins=256, range=(0, 256))
            hist = hist.astype(np.float32)
            hist /= (hist.sum() + 1e-8)

            # Entropy of LBP histogram as richness measure
            entropy = -float(np.sum(hist * np.log2(hist + 1e-8)))
            # Max entropy for 256 bins = log2(256) = 8.0
            lbp_score = min(1.0, entropy / 8.0)
            return lbp_score
        except Exception:
            return 0.5
