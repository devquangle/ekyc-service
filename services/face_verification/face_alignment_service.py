import cv2
import numpy as np
from typing import Optional
from utils.logger import logger


class FaceAlignmentService:
    """
    Standardized Face Preprocessing & 5-point Alignment Service.
    Applies unified pipeline across Card Face and Selfie Face:
    detect -> landmarks -> alignment (affine warp) -> resize (112x112) -> normalize.
    """

    REFERENCE_5PTS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    def align_face(
        self,
        image: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        target_size: tuple[int, int] = (112, 112)
    ) -> Optional[np.ndarray]:
        """
        Aligns and resizes face image to standardized target size (112x112).
        """
        if image is None or image.size == 0:
            return None

        # 1. 5-Point Affine Warping if landmarks present
        if landmarks is not None and len(landmarks) == 5:
            try:
                src_pts = np.array(landmarks, dtype=np.float32)
                dst_pts = self.REFERENCE_5PTS.copy()
                if target_size != (112, 112):
                    dst_pts[:, 0] *= (target_size[0] / 112.0)
                    dst_pts[:, 1] *= (target_size[1] / 112.0)

                tfm, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                if tfm is not None:
                    aligned = cv2.warpAffine(
                        image, tfm, (target_size[0], target_size[1]), borderValue=0
                    )
                    return aligned
            except Exception as e:
                logger.warning(f"5-point affine alignment failed, falling back to standard resize: {str(e)}")

        # 2. Fallback: Direct resize to target_size
        try:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            return resized
        except Exception as e:
            logger.error(f"Face image resize failed: {str(e)}")
            return None
