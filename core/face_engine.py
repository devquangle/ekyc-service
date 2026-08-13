import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from config import settings
from schemas.face import FaceVerifyResponse
from core.face_verification import FaceVerificationService
from utils.logger import logger
from utils.image_utils import crop_image


class FaceEngine:
    """
    InsightFace ArcFace Engine wrapper for face detection, 5-point alignment,
    512-dimensional embedding extraction, and modular Face Verification.
    """

    def __init__(self):
        self.app = None
        self._initialize_model()
        self.verification_service = FaceVerificationService(face_app=self.app)

    def _initialize_model(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis

            ctx_id = 0 if settings.DEVICE.lower() == "cuda" else -1
            self.app = FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL_NAME,
                root=settings.MODEL_DIR,
                allowed_modules=['detection', 'recognition']
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("InsightFace engine initialized successfully.")
        except Exception as e:
            logger.warning(f"InsightFace model failed to initialize (falling back to OpenCV Haar Cascade): {str(e)}")
            self.app = None

    def verify_faces(
        self, card_image: Optional[np.ndarray], selfie_image: Optional[np.ndarray]
    ) -> FaceVerifyResponse:
        """
        Executes face verification using modular FaceVerificationService.
        """
        return self.verification_service.verify_faces(card_image, selfie_image)

    def extract_face_embedding(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Detects faces and extracts normalized 512-d embedding for the largest face.
        Returns: (embedding_vector, face_count)
        """
        if image is None or image.size == 0:
            return None, 0

        vec, dim, norm, errs = self.verification_service.embedding_service.extract_embedding(image)
        if vec is not None:
            return vec, 1
        return None, 0

    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculates Cosine Similarity between two L2-normalized 512d vectors.
        """
        sim, errs = self.verification_service.embedding_service.calculate_cosine_similarity(embedding1, embedding2)
        return sim

    def crop_portrait_from_card(self, card_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects and crops portrait photo from card image using layout-agnostic CardFaceExtractor.
        Returns cropped face portrait array or None if no face is detected.
        """
        if card_image is None or card_image.size == 0:
            return None

        crop, _, _, _ = self.verification_service.card_extractor.extract_face(card_image)
        if crop is not None:
            return crop

        logger.warning("[FACE_ENGINE] Could not extract portrait face from card image.")
        return None
