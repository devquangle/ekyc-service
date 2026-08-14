import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from config import settings
from schemas.face import FaceVerifyResponse
from core.face_verification.face_verification_service import FaceVerificationService
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
            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            use_cuda = (settings.DEVICE.lower() == "cuda" and "CUDAExecutionProvider" in available_providers)
            ctx_id = 0 if use_cuda else -1
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]

            self.app = FaceAnalysis(
                name=settings.INSIGHTFACE_MODEL_NAME,
                root=settings.MODEL_DIR,
                allowed_modules=['detection', 'recognition'],
                providers=providers
            )
            self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("[FACE_ENGINE] InsightFace engine initialized successfully.")
        except Exception as e:
            logger.warning(f"[FACE_ENGINE] InsightFace model failed to initialize (falling back to OpenCV Haar Cascade): {str(e)}")
            self.app = None

    def verify_faces(
        self, card_image: Optional[np.ndarray], selfie_image: Optional[np.ndarray]
    ) -> FaceVerifyResponse:
        """
        Executes modular 1-1 face verification between card portrait and live selfie.
        """
        return self.verification_service.verify_faces(card_image, selfie_image)

    def extract_face_embedding(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Executes standard eKYC pipeline:
        1. Detect Face & Extract 5 Landmarks (KPS).
        2. 5-point Affine Alignment to Standard 112x112 ArcFace Template.
        3. ArcFace Feature Extraction via Recognition Model (get_feat).
        4. L2 Normalization (||v||_2 = 1.0).

        Returns:
            Tuple: (L2_normalized_embedding_vector, face_count)
        """
        if image is None or image.size == 0:
            return None, 0

        # Step 1: Detect face using layout-agnostic extractor
        face_crop, kps, bbox_info, errs = self.verification_service.card_extractor.extract_face(image)
        if face_crop is None or not bbox_info.detected:
            # Try selfie extractor if card extractor found no face
            face_crop, kps, bbox_info, errs = self.verification_service.selfie_extractor.extract_face(image)

        if face_crop is None or not bbox_info.detected:
            return None, 0

        # Step 2: 5-Point Affine Alignment to Standard 112x112 Template
        aligned_face = self.verification_service.alignment_service.align_face(
            face_crop,
            landmarks=kps,
            bbox=bbox_info.bbox,
            target_size=(112, 112)
        )

        if aligned_face is None:
            return None, 0

        # Step 3 & 4: ArcFace Feature Extraction & L2 Normalization
        emb_vec, dim, norm, emb_errs = self.verification_service.embedding_service.extract_embedding(aligned_face)

        if emb_vec is not None and len(emb_errs) == 0:
            return emb_vec, 1

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
