import numpy as np
from typing import Optional, Tuple, List
from config import settings
from schemas.face import FaceVerifyResponse, BoundingBoxInfo, FaceQualityMetrics
from core.face_verification.card_face_extractor import CardFaceExtractor
from core.face_verification.selfie_face_extractor import SelfieFaceExtractor
from core.face_verification.face_quality_service import FaceQualityService
from core.face_verification.face_alignment_service import FaceAlignmentService
from core.face_verification.face_embedding_service import FaceEmbeddingService
from utils.logger import logger


class FaceVerificationService:
    """
    Main Orchestrator for 1-1 Face Verification Pipeline.
    Coordinates face extraction, quality analysis, 5-point alignment,
    ArcFace 512-d embedding extraction, cosine similarity calculation,
    and 3-tier decision verification (MATCH, BORDERLINE, MISMATCH).
    """

    def __init__(
        self,
        card_extractor: Optional[CardFaceExtractor] = None,
        selfie_extractor: Optional[SelfieFaceExtractor] = None,
        quality_service: Optional[FaceQualityService] = None,
        alignment_service: Optional[FaceAlignmentService] = None,
        embedding_service: Optional[FaceEmbeddingService] = None,
        face_app=None
    ):
        self.card_extractor = card_extractor or CardFaceExtractor(face_app=face_app)
        self.selfie_extractor = selfie_extractor or SelfieFaceExtractor(face_app=face_app)
        self.quality_service = quality_service or FaceQualityService()
        self.alignment_service = alignment_service or FaceAlignmentService()
        self.embedding_service = embedding_service or FaceEmbeddingService(face_app=face_app)

    def verify_faces(
        self,
        card_image: Optional[np.ndarray],
        selfie_image: Optional[np.ndarray]
    ) -> FaceVerifyResponse:
        """
        Executes end-to-end face verification between ID card and selfie image.
        """
        all_errors: List[str] = []

        if card_image is None or selfie_image is None or card_image.size == 0 or selfie_image.size == 0:
            return FaceVerifyResponse(
                faceVerified=False,
                similarityScore=0.0,
                threshold=settings.FACE_MATCH_THRESHOLD,
                decision="MISMATCH",
                margin=round(0.0 - settings.FACE_MATCH_THRESHOLD, 4),
                errors=["INVALID_IMAGE_FORMAT"]
            )

        # 1. Extract Face from Card Image
        card_crop, card_kps, card_bbox, card_extract_errs = self.card_extractor.extract_face(card_image)
        all_errors.extend(card_extract_errs)

        # 2. Extract Face from Selfie Image
        selfie_crop, selfie_kps, selfie_bbox, selfie_extract_errs = self.selfie_extractor.extract_face(selfie_image)
        all_errors.extend(selfie_extract_errs)

        # 3. Analyze Face Quality Metrics
        card_quality = self.quality_service.analyze_quality(card_crop, card_bbox, card_kps) if card_crop is not None else None
        selfie_quality = self.quality_service.analyze_quality(selfie_crop, selfie_bbox, selfie_kps) if selfie_crop is not None else None

        self._log_face_metrics(card_bbox, selfie_bbox, card_quality, selfie_quality)

        # 4. Fatal Extraction Error Check
        if card_crop is None or selfie_crop is None or len(all_errors) > 0:
            if card_crop is None and not any("CARD" in e for e in all_errors):
                all_errors.append("CARD_PORTRAIT_FACE_NOT_FOUND")
            if selfie_crop is None and not any("SELFIE" in e for e in all_errors):
                all_errors.append("SELFIE_FACE_NOT_FOUND")

            unique_errors = list(dict.fromkeys(all_errors))
            logger.warning(f"[FACE_VERIFICATION] Face extraction aborted: {unique_errors}")
            return FaceVerifyResponse(
                faceVerified=False,
                similarityScore=0.0,
                threshold=settings.FACE_MATCH_THRESHOLD,
                decision="MISMATCH",
                margin=round(0.0 - settings.FACE_MATCH_THRESHOLD, 4),
                errors=unique_errors,
                cardFaceInfo=card_bbox,
                selfieFaceInfo=selfie_bbox,
                cardFaceQuality=card_quality,
                selfieFaceQuality=selfie_quality
            )

        # 5. 5-Point Affine Alignment to Standard 112x112 Template (Relative Coordinate Correction)
        card_bbox_coords = card_bbox.bbox if card_bbox and card_bbox.detected else None
        selfie_bbox_coords = selfie_bbox.bbox if selfie_bbox and selfie_bbox.detected else None

        card_aligned = self.alignment_service.align_face(
            card_crop,
            landmarks=card_kps,
            bbox=card_bbox_coords,
            target_size=(112, 112)
        )
        selfie_aligned = self.alignment_service.align_face(
            selfie_crop,
            landmarks=selfie_kps,
            bbox=selfie_bbox_coords,
            target_size=(112, 112)
        )

        if card_aligned is None or selfie_aligned is None:
            all_errors.append("FACE_ALIGNMENT_FAILED")
            unique_errors = list(dict.fromkeys(all_errors))
            return FaceVerifyResponse(
                faceVerified=False,
                similarityScore=0.0,
                threshold=settings.FACE_MATCH_THRESHOLD,
                decision="MISMATCH",
                margin=round(0.0 - settings.FACE_MATCH_THRESHOLD, 4),
                errors=unique_errors,
                cardFaceInfo=card_bbox,
                selfieFaceInfo=selfie_bbox,
                cardFaceQuality=card_quality,
                selfieFaceQuality=selfie_quality
            )

        # 6. Extract L2-Normalized 512-d ArcFace Embeddings
        emb_card, dim_card, norm_card, emb_card_errs = self.embedding_service.extract_embedding(card_aligned)
        all_errors.extend(emb_card_errs)

        emb_selfie, dim_selfie, norm_selfie, emb_selfie_errs = self.embedding_service.extract_embedding(selfie_aligned)
        all_errors.extend(emb_selfie_errs)

        logger.info(
            f"[FACE_VERIFICATION] EMBEDDINGS: cardDim={dim_card}, selfieDim={dim_selfie}, "
            f"cardNorm={norm_card:.4f}, selfieNorm={norm_selfie:.4f}"
        )

        if emb_card is None or emb_selfie is None or len(all_errors) > 0:
            unique_errors = list(dict.fromkeys(all_errors))
            logger.warning(f"[FACE_VERIFICATION] Embedding extraction failed: {unique_errors}")
            return FaceVerifyResponse(
                faceVerified=False,
                similarityScore=0.0,
                threshold=settings.FACE_MATCH_THRESHOLD,
                decision="MISMATCH",
                margin=round(0.0 - settings.FACE_MATCH_THRESHOLD, 4),
                errors=unique_errors,
                cardFaceInfo=card_bbox,
                selfieFaceInfo=selfie_bbox,
                cardFaceQuality=card_quality,
                selfieFaceQuality=selfie_quality
            )

        # 7. Compute Cosine Similarity
        similarity, sim_errs = self.embedding_service.calculate_cosine_similarity(emb_card, emb_selfie)
        all_errors.extend(sim_errs)
        similarity = round(similarity, 4)

        match_threshold = settings.FACE_MATCH_THRESHOLD
        mismatch_threshold = settings.FACE_MISMATCH_THRESHOLD
        margin = round(similarity - match_threshold, 4)

        # 8. Decision Gate Evaluation
        decision, face_verified, decision_errs = self._evaluate_decision(
            similarity, mismatch_threshold, match_threshold, len(all_errors) > 0
        )
        all_errors.extend(decision_errs)
        unique_errors = list(dict.fromkeys(all_errors))

        logger.info(
            f"[FACE_VERIFICATION] MATCHING: cosineSimilarity={similarity}, "
            f"threshold={match_threshold}, margin={margin}, decision={decision}"
        )

        return FaceVerifyResponse(
            faceVerified=face_verified,
            similarityScore=similarity,
            threshold=match_threshold,
            decision=decision,
            margin=margin,
            errors=unique_errors,
            cardFaceInfo=card_bbox,
            selfieFaceInfo=selfie_bbox,
            cardFaceQuality=card_quality,
            selfieFaceQuality=selfie_quality
        )

    def _evaluate_decision(
        self,
        similarity: float,
        mismatch_threshold: float,
        match_threshold: float,
        has_errors: bool
    ) -> Tuple[str, bool, List[str]]:
        errs: List[str] = []

        if has_errors:
            return "MISMATCH", False, errs

        if similarity >= match_threshold:
            return "MATCH", True, []
        elif similarity >= mismatch_threshold:
            errs.append("FACE_SIMILARITY_BORDERLINE")
            return "BORDERLINE", False, errs
        else:
            errs.append("FACE_MISMATCH")
            return "MISMATCH", False, errs

    def _log_face_metrics(
        self,
        card_bbox: Optional[BoundingBoxInfo],
        selfie_bbox: Optional[BoundingBoxInfo],
        card_quality: Optional[FaceQualityMetrics],
        selfie_quality: Optional[FaceQualityMetrics]
    ):
        if card_bbox:
            logger.info(
                f"[FACE_METRICS] CARD_FACE: detected={card_bbox.detected}, bbox={card_bbox.bbox}, "
                f"x1={card_bbox.x1}, y1={card_bbox.y1}, x2={card_bbox.x2}, y2={card_bbox.y2}, "
                f"width={card_bbox.width}, height={card_bbox.height}, score={card_bbox.detectionScore:.4f}"
            )
        if selfie_bbox:
            logger.info(
                f"[FACE_METRICS] SELFIE_FACE: detected={selfie_bbox.detected}, bbox={selfie_bbox.bbox}, "
                f"x1={selfie_bbox.x1}, y1={selfie_bbox.y1}, x2={selfie_bbox.x2}, y2={selfie_bbox.y2}, "
                f"width={selfie_bbox.width}, height={selfie_bbox.height}, score={selfie_bbox.detectionScore:.4f}"
            )
        if card_quality:
            logger.info(
                f"[FACE_METRICS] CARD_QUALITY: blur={card_quality.blurScore}, brightness={card_quality.brightness}, "
                f"size={card_quality.faceSize}, yaw={card_quality.yaw}, pitch={card_quality.pitch}, roll={card_quality.roll}"
            )
        if selfie_quality:
            logger.info(
                f"[FACE_METRICS] SELFIE_QUALITY: blur={selfie_quality.blurScore}, brightness={selfie_quality.brightness}, "
                f"size={selfie_quality.faceSize}, yaw={selfie_quality.yaw}, pitch={selfie_quality.pitch}, roll={selfie_quality.roll}"
            )
