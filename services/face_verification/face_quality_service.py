import cv2
import numpy as np
from typing import Optional
from schemas.face import FaceQualityMetrics, BoundingBoxInfo


class FaceQualityService:
    """
    Evaluates image and face quality parameters:
    blur score (Laplacian variance), brightness, face area size, and head pose angles (yaw, pitch, roll).
    """

    def analyze_quality(
        self,
        face_crop: Optional[np.ndarray],
        bbox_info: Optional[BoundingBoxInfo] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> FaceQualityMetrics:
        if face_crop is None or face_crop.size == 0:
            return FaceQualityMetrics()

        # Grayscale conversion
        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        # 1. Blur score (Laplacian variance)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Brightness (mean intensity)
        brightness = float(np.mean(gray))

        # 3. Face size
        if bbox_info and bbox_info.detected:
            face_size = bbox_info.width * bbox_info.height
        else:
            face_size = face_crop.shape[1] * face_crop.shape[0]

        # 4. Pose estimation (yaw, pitch, roll) from 5 landmarks if available
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if landmarks is not None and len(landmarks) == 5:
            yaw, pitch, roll = self._estimate_pose_from_landmarks(landmarks)

        return FaceQualityMetrics(
            blurScore=round(blur_score, 2),
            brightness=round(brightness, 2),
            faceSize=int(face_size),
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2)
        )

    def _estimate_pose_from_landmarks(self, kps: np.ndarray) -> tuple[float, float, float]:
        """
        Estimates rough yaw, pitch, roll angles (degrees) from 5 facial landmarks.
        kps: [[x_le, y_le], [x_re, y_re], [x_nose, y_nose], [x_lm, y_lm], [x_rm, y_rm]]
        """
        try:
            left_eye, right_eye, nose, left_mouth, right_mouth = kps[:5]

            # Roll angle (eye tilt)
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            roll = np.degrees(np.arctan2(dy, dx))

            # Yaw angle (horizontal asymmetry of nose relative to eyes center)
            eye_center_x = (left_eye[0] + right_eye[0]) / 2.0
            eye_dist = max(1.0, np.linalg.norm(right_eye - left_eye))
            yaw = np.degrees(np.arctan2(nose[0] - eye_center_x, eye_dist)) * 2.0

            # Pitch angle (vertical position of nose relative to eye-mouth midpoint)
            eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2.0
            face_height = max(1.0, mouth_center_y - eye_center_y)
            pitch = np.degrees(np.arctan2(nose[1] - (eye_center_y + mouth_center_y) / 2.0, face_height)) * 2.0

            return float(yaw), float(pitch), float(roll)
        except Exception:
            return 0.0, 0.0, 0.0
