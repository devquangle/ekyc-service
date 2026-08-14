import cv2
import numpy as np
from typing import Optional, Tuple
from schemas.face import FaceQualityMetrics, BoundingBoxInfo


class FaceQualityService:
    """
    Evaluates face image quality parameters:
    - Blur score (Laplacian variance: higher score indicates sharper image).
    - Mean grayscale brightness.
    - Face bounding area in pixels.
    - Head pose Euler angles (Yaw, Pitch, Roll) estimated from 5 facial landmarks.
    """

    def analyze_quality(
        self,
        face_crop: Optional[np.ndarray],
        bbox_info: Optional[BoundingBoxInfo] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> FaceQualityMetrics:
        """
        Calculates quality metrics for an extracted face image.

        Args:
            face_crop: Cropped BGR face ROI.
            bbox_info: Bounding box metadata.
            landmarks: 5 facial landmarks [[x, y], ...].

        Returns:
            FaceQualityMetrics object.
        """
        if face_crop is None or face_crop.size == 0:
            return FaceQualityMetrics()

        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        # 1. Blur score: Laplacian Variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. Mean Brightness
        brightness = float(np.mean(gray))

        # 3. Face Area Size
        if bbox_info and bbox_info.detected:
            face_size = int(bbox_info.width * bbox_info.height)
        else:
            face_size = int(face_crop.shape[1] * face_crop.shape[0])

        # 4. Head Pose Angles (Yaw, Pitch, Roll)
        yaw, pitch, roll = 0.0, 0.0, 0.0
        if landmarks is not None and len(landmarks) == 5:
            yaw, pitch, roll = self._estimate_pose_from_landmarks(landmarks)

        return FaceQualityMetrics(
            blurScore=round(blur_score, 2),
            brightness=round(brightness, 2),
            faceSize=face_size,
            yaw=round(yaw, 2),
            pitch=round(pitch, 2),
            roll=round(roll, 2)
        )

    def _estimate_pose_from_landmarks(self, kps: np.ndarray) -> Tuple[float, float, float]:
        """
        Estimates approximate 3D head pose Euler angles (Yaw, Pitch, Roll in degrees)
        from 5 standard facial keypoints (left eye, right eye, nose, left mouth, right mouth).
        """
        try:
            pts = np.array(kps, dtype=np.float32)
            left_eye, right_eye, nose, left_mouth, right_mouth = pts[:5]

            # Roll (in-plane tilt): Angle of line connecting the two eyes
            dx = float(right_eye[0] - left_eye[0])
            dy = float(right_eye[1] - left_eye[1])
            roll = float(np.degrees(np.arctan2(dy, dx)))

            # Yaw (left-right turn): Nose offset from midpoint between eyes
            eye_center_x = float((left_eye[0] + right_eye[0]) / 2.0)
            eye_dist = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
            yaw = float(np.degrees(np.arctan2(nose[0] - eye_center_x, eye_dist)) * 2.0)

            # Pitch (up-down nod): Nose position relative to eye-mouth vertical midpoint
            eye_center_y = float((left_eye[1] + right_eye[1]) / 2.0)
            mouth_center_y = float((left_mouth[1] + right_mouth[1]) / 2.0)
            face_height = max(1.0, float(mouth_center_y - eye_center_y))
            pitch = float(np.degrees(np.arctan2(nose[1] - (eye_center_y + mouth_center_y) / 2.0, face_height)) * 2.0)

            return float(yaw), float(pitch), float(roll)
        except Exception:
            return 0.0, 0.0, 0.0
