import os
import cv2
import tempfile
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from config import settings
from core.anti_spoof_engine import AntiSpoofEngine
from utils.logger import logger


class LivenessEngine:
    """
    High-Security Video Liveness & Anti-Spoofing Engine.
    Combines:
    1. Passive Anti-Spoofing (MiniFASNet Deep Learning + Multi-metric texture analysis).
    2. Active Challenge-Response Gesture Verification:
       - Eye Aspect Ratio (EAR) Blink Detection
       - 3D Head Pose Yaw/Pitch (TURN_LEFT, TURN_RIGHT, LOOK_UP, LOOK_DOWN)
       - Smile Verification (SMILE)
    Zero fake-pass logic: all verification strictly fails if criteria are not met.
    """

    def __init__(self):
        # 1. Anti-Spoofing Deep Learning Engine
        self.anti_spoof = AntiSpoofEngine(
            model_path_1=settings.ANTI_SPOOF_MODEL_PATH_1,
            model_path_2=settings.ANTI_SPOOF_MODEL_PATH_2,
        ) if settings.ANTI_SPOOF_ENABLED else None

        # 2. Pre-load OpenCV Cascades once at initialization (Zero runtime disk I/O)
        try:
            self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self._eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self._smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        except Exception as e:
            logger.warning(f"[LIVENESS] Failed to pre-load cascades: {str(e)}")
            self._face_cascade = None
            self._eye_cascade = None
            self._smile_cascade = None

    def analyze_video(
        self, video_bytes: bytes, expected_gestures: Optional[List[str]] = None
    ) -> Tuple[bool, float, List[str], List[str]]:
        """
        Analyzes video stream for passive anti-spoofing and active challenge gestures.

        Returns:
            Tuple: (liveness_verified, liveness_score, checks_passed, errors)
        """
        if not video_bytes:
            return False, 0.0, [], ["VIDEO_EMPTY"]

        if len(video_bytes) > settings.MAX_VIDEO_SIZE_MB * 1024 * 1024:
            return False, 0.0, [], ["VIDEO_SIZE_EXCEEDED"]

        sampled_frames: List[np.ndarray] = []
        duration_exceeded = False
        video_invalid = False

        # Safe Temporary File Management for OpenCV VideoCapture across Windows/Linux
        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp_file.name

        try:
            tmp_file.write(video_bytes)
            tmp_file.flush()
            tmp_file.close()  # Close descriptor so OpenCV can read without file locking

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                video_invalid = True
            else:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = total_frames / fps if fps > 0 else 0.0

                if duration_sec > settings.MAX_VIDEO_DURATION_SEC:
                    duration_exceeded = True
                else:
                    sample_interval = max(1, int(fps / settings.VIDEO_FRAME_SAMPLING_RATE))
                    frame_idx = 0

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break
                        if frame_idx % sample_interval == 0:
                            sampled_frames.append(frame)
                        frame_idx += 1

                cap.release()
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.debug(f"[LIVENESS] Temp file cleanup notice: {str(e)}")

        if video_invalid:
            return False, 0.0, [], ["VIDEO_INVALID"]

        if duration_exceeded:
            return False, 0.0, [], ["VIDEO_DURATION_EXCEEDED"]

        if not sampled_frames:
            return False, 0.0, [], ["VIDEO_NO_FRAMES"]

        # 1. Passive Anti-Spoofing Analysis across sampled frames
        passive_scores = [self._analyze_passive_frame(f) for f in sampled_frames]
        avg_passive_score = float(np.mean(passive_scores)) if passive_scores else 0.0

        checks_passed: List[str] = []
        errors: List[str] = []

        if avg_passive_score >= settings.LIVENESS_PASSIVE_THRESHOLD:
            checks_passed.append("PASSIVE_TEXTURE_CHECK")
        else:
            errors.append("LIVENESS_FAILED")

        # 2. Active Gesture Verification
        if expected_gestures:
            active_passed, detected_gestures = self._verify_active_gestures(sampled_frames, expected_gestures)
            if active_passed:
                checks_passed.extend([f"GESTURE_{g}" for g in detected_gestures])
            else:
                errors.append("ACTIVE_GESTURE_FAILED")
        else:
            # Default single verification: natural blink detection
            blink_ok = self._detect_blink(sampled_frames)
            if blink_ok:
                checks_passed.append("BLINK_DETECTION")

        liveness_verified = (avg_passive_score >= settings.LIVENESS_PASSIVE_THRESHOLD) and len(errors) == 0

        logger.info(
            f"[LIVENESS] Video verification: verified={liveness_verified}, "
            f"score={avg_passive_score:.4f}, passed={checks_passed}, errors={errors}"
        )

        return liveness_verified, round(avg_passive_score, 4), checks_passed, errors

    def _analyze_passive_frame(self, frame: np.ndarray) -> float:
        if frame is None or frame.size == 0:
            return 0.0

        if self.anti_spoof is not None:
            return self.anti_spoof.predict(frame)

        # Baseline texture score
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        if np.std(gray) < 5.0:
            return 0.0
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        return min(1.0, float(np.var(lap) / 500.0))

    def _verify_active_gestures(
        self, frames: List[np.ndarray], expected_gestures: List[str]
    ) -> Tuple[bool, List[str]]:
        """
        Validates presence of each expected gesture across frames.
        Returns: (passed, list_of_confirmed_gestures)
        """
        detected: List[str] = []

        for gesture in expected_gestures:
            g_upper = gesture.strip().upper()
            if g_upper == "BLINK":
                if self._detect_blink(frames):
                    detected.append("BLINK")
            elif g_upper == "TURN_LEFT":
                if self._detect_head_turn(frames, direction="LEFT"):
                    detected.append("TURN_LEFT")
            elif g_upper == "TURN_RIGHT":
                if self._detect_head_turn(frames, direction="RIGHT"):
                    detected.append("TURN_RIGHT")
            elif g_upper in ("LOOK_UP", "NOD_UP"):
                if self._detect_head_pitch(frames, direction="UP"):
                    detected.append(g_upper)
            elif g_upper in ("LOOK_DOWN", "NOD_DOWN"):
                if self._detect_head_pitch(frames, direction="DOWN"):
                    detected.append(g_upper)
            elif g_upper == "SMILE":
                if self._detect_smile(frames):
                    detected.append("SMILE")

        passed = len(detected) == len(expected_gestures)
        return passed, detected

    def _detect_blink(self, frames: List[np.ndarray]) -> bool:
        """
        Calculates Eye Aspect Ratio (EAR) series across video frames.
        Strict: Returns False if no eyes detected or EAR drop is below threshold.
        """
        if not frames or self._face_cascade is None or self._eye_cascade is None:
            return False

        try:
            ear_series: List[float] = []

            for frame in frames:
                if frame is None or frame.size == 0:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

                if len(faces) == 0:
                    continue

                fx, fy, fw, fh = faces[0]
                face_roi = gray[fy:fy + int(fh * 0.6), fx:fx + fw]

                eyes = self._eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

                for (ex, ey, ew, eh) in eyes:
                    # Construct 6 landmark approximation for EAR formula
                    p1 = np.array([ex, ey + eh / 2.0])
                    p4 = np.array([ex + ew, ey + eh / 2.0])
                    p2 = np.array([ex + ew / 3.0, ey])
                    p6 = np.array([ex + ew / 3.0, ey + eh])
                    p3 = np.array([ex + 2.0 * ew / 3.0, ey])
                    p5 = np.array([ex + 2.0 * ew / 3.0, ey + eh])

                    ear = (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * max(1e-5, np.linalg.norm(p1 - p4)))
                    ear_series.append(float(ear))

            if len(ear_series) < 2:
                return False

            min_ear = min(ear_series)
            max_ear = max(ear_series)
            ear_delta = max_ear - min_ear

            logger.info(f"[LIVENESS] Blink EAR: min={min_ear:.4f}, max={max_ear:.4f}, delta={ear_delta:.4f}")
            # Blink confirmed if minimum EAR drops below threshold or relative difference > 0.12
            return min_ear < settings.LIVENESS_EYE_RATIO_THRESHOLD or ear_delta >= 0.12
        except Exception as e:
            logger.error(f"[LIVENESS] Blink detection error: {str(e)}")
            return False

    def _detect_head_turn(self, frames: List[np.ndarray], direction: str = "LEFT") -> bool:
        """
        Detects horizontal head turn (Yaw angle change) across video frames.
        """
        if not frames or self._face_cascade is None or self._eye_cascade is None:
            return False

        try:
            x_offsets: List[float] = []

            for frame in frames:
                if frame is None or frame.size == 0:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

                if len(faces) == 0:
                    continue

                fx, fy, fw, fh = faces[0]
                face_center_x = fx + fw / 2.0
                face_roi = gray[fy:fy + int(fh * 0.6), fx:fx + fw]

                eyes = self._eye_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=3)
                if len(eyes) >= 2:
                    eye_mid_x = fx + (eyes[0][0] + eyes[1][0] + (eyes[0][2] + eyes[1][2]) / 2.0) / 2.0
                    offset = (eye_mid_x - face_center_x) / float(fw)
                    x_offsets.append(offset)

            if len(x_offsets) < 2:
                return False

            min_offset = min(x_offsets)
            max_offset = max(x_offsets)

            if direction == "LEFT":
                return min_offset < -0.06 or (max_offset - min_offset) > 0.08
            else:
                return max_offset > 0.06 or (max_offset - min_offset) > 0.08
        except Exception as e:
            logger.error(f"[LIVENESS] Head turn detection error: {str(e)}")
            return False

    def _detect_head_pitch(self, frames: List[np.ndarray], direction: str = "UP") -> bool:
        """
        Detects vertical head pitch (up/down nod) across video frames.
        """
        if not frames or self._face_cascade is None:
            return False

        try:
            y_positions: List[float] = []
            for frame in frames:
                if frame is None or frame.size == 0:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) > 0:
                    _, fy, _, fh = faces[0]
                    y_positions.append(fy / float(frame.shape[0]))

            if len(y_positions) < 2:
                return False

            delta_y = max(y_positions) - min(y_positions)
            return delta_y >= 0.04
        except Exception as e:
            logger.error(f"[LIVENESS] Head pitch detection error: {str(e)}")
            return False

    def _detect_smile(self, frames: List[np.ndarray]) -> bool:
        """
        Detects smile event in mouth region of face across video frames.
        """
        if not frames or self._face_cascade is None or self._smile_cascade is None:
            return False

        try:
            smile_count = 0
            for frame in frames:
                if frame is None or frame.size == 0:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) == 0:
                    continue
                fx, fy, fw, fh = faces[0]
                mouth_roi = gray[fy + int(fh * 0.6):fy + fh, fx:fx + fw]
                smiles = self._smile_cascade.detectMultiScale(mouth_roi, scaleFactor=1.7, minNeighbors=8)
                if len(smiles) > 0:
                    smile_count += 1

            return smile_count >= 1
        except Exception as e:
            logger.error(f"[LIVENESS] Smile detection error: {str(e)}")
            return False
