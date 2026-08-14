import cv2
import numpy as np
from typing import Optional, Tuple, List, Union
from utils.logger import logger


class FaceAlignmentService:
    """
    Standardized Face Preprocessing & 5-Point Affine Alignment Service (InsightFace Standard).
    Aligns facial landmarks (eyes, nose, mouth corners) to standard 112x112 ArcFace template.
    Robustly handles both full-frame images and cropped ROIs with relative coordinate correction.
    """

    # Standard ArcFace 112x112 5-point reference landmarks
    REFERENCE_5PTS = np.array([
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041]   # Right mouth corner
    ], dtype=np.float32)

    def align_face(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        bbox: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
        target_size: Tuple[int, int] = (112, 112)
    ) -> Optional[np.ndarray]:
        """
        Aligns and resizes face image to standardized target size (112x112).

        Args:
            image: Source face image (Full image or Cropped face ROI).
            landmarks: 5 facial landmarks [[x, y], ...].
            bbox: Optional bounding box [x1, y1, x2, y2] if image is a cropped ROI.
            target_size: Target aligned dimensions (width, height), default (112, 112).

        Returns:
            112x112 BGR aligned face image or None if input invalid.
        """
        if image is None or image.size == 0:
            return None

        # 1. 5-Point Affine Alignment
        if landmarks is not None and len(landmarks) == 5:
            try:
                src_pts = np.array(landmarks, dtype=np.float32)

                # If image is a cropped ROI and landmarks are in original full-image coordinates,
                # shift landmarks to be relative to the cropped ROI origin (x1, y1).
                img_h, img_w = image.shape[:2]
                if bbox is not None and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    # Check if landmarks are in global coordinate space outside the crop bounds
                    if np.any(src_pts[:, 0] > img_w) or np.any(src_pts[:, 1] > img_h) or np.any(src_pts[:, 0] < 0) or np.any(src_pts[:, 1] < 0):
                        src_pts[:, 0] -= float(x1)
                        src_pts[:, 1] -= float(y1)

                dst_pts = self.REFERENCE_5PTS.copy()
                if target_size != (112, 112):
                    scale_x = target_size[0] / 112.0
                    scale_y = target_size[1] / 112.0
                    dst_pts[:, 0] *= scale_x
                    dst_pts[:, 1] *= scale_y

                # Compute Similarity / Affine transformation matrix
                tfm, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
                if tfm is not None:
                    aligned = cv2.warpAffine(
                        image,
                        tfm,
                        (target_size[0], target_size[1]),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT
                    )
                    return aligned
            except Exception as e:
                logger.warning(f"[FACE_ALIGNMENT] 5-point affine alignment failed, falling back to resize: {str(e)}")

        # 2. Fallback: Direct aspect-ratio preserving resize
        try:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            return resized
        except Exception as e:
            logger.error(f"[FACE_ALIGNMENT] Face image resize failed: {str(e)}")
            return None
