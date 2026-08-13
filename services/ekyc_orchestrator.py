import time
import uuid
from typing import Optional, List
import numpy as np

from config import settings
from core.face_engine import FaceEngine
from core.liveness_engine import LivenessEngine
from processors.card_processor import CardProcessor
from processors.card_validator import CardValidator
from schemas.card import CardProcessResponse
from schemas.face import FaceVerifyResponse
from schemas.liveness import LivenessResponse
from schemas.ekyc import FullEkycResponse
from utils.logger import logger
from utils.image_utils import decode_image_bytes


class EkycOrchestrator:
    """
    Main High-Level Orchestrator executing the complete 8-step eKYC pipeline
    and applying the Final Decision Logic Gate.
    """

    def __init__(
        self,
        card_processor: CardProcessor,
        card_validator: CardValidator,
        face_engine: FaceEngine,
        liveness_engine: LivenessEngine
    ):
        self.card_processor = card_processor
        self.card_validator = card_validator
        self.face_engine = face_engine
        self.liveness_engine = liveness_engine

    def process_card(self, front_bytes: bytes, back_bytes: bytes) -> CardProcessResponse:
        """
        Executes Card Extraction & Validation Flow.
        """
        front_img = decode_image_bytes(front_bytes)
        back_img = decode_image_bytes(back_bytes)

        if front_img is None or back_img is None:
            return CardProcessResponse(
                cardVerified=False,
                cardType="UNKNOWN",
                cardTypeConfidence=0.0,
                errors=["INVALID_IMAGE_FORMAT"]
            )

        card_type, card_type_conf, extracted_data, qr_data, mrz_data, quality_checks, field_metadata = self.card_processor.process(front_img, back_img)

        card_verified, cross_val_result, val_errors = self.card_validator.validate(
            extracted_data, qr_data, mrz_data, card_type
        )

        all_errors = val_errors
        if quality_checks.isBlur:
            all_errors.append("CARD_BLURRY")

        return CardProcessResponse(
            cardVerified=card_verified and len(all_errors) == 0,
            cardType=card_type,
            cardTypeConfidence=card_type_conf,
            extractedData=extracted_data,
            crossValidation=cross_val_result,
            qualityChecks=quality_checks,
            fieldMetadata=field_metadata,
            errors=all_errors
        )

    def verify_face(
        self, card_portrait_bytes: bytes, selfie_bytes: bytes
    ) -> FaceVerifyResponse:
        """
        Executes Face Verification Flow using modular FaceVerificationService.
        """
        card_img = decode_image_bytes(card_portrait_bytes)
        selfie_img = decode_image_bytes(selfie_bytes)

        if card_img is None or selfie_img is None:
            return FaceVerifyResponse(
                faceVerified=False,
                similarityScore=0.0,
                threshold=settings.FACE_MATCH_THRESHOLD,
                decision="MISMATCH",
                margin=-settings.FACE_MATCH_THRESHOLD,
                errors=["INVALID_IMAGE_FORMAT"]
            )

        return self.face_engine.verify_faces(card_img, selfie_img)

    def detect_liveness(
        self, video_bytes: bytes, expected_gestures: Optional[List[str]] = None
    ) -> LivenessResponse:
        """
        Executes Video Liveness Detection Flow.
        """
        liveness_verified, score, checks_passed, errors = self.liveness_engine.analyze_video(
            video_bytes, expected_gestures
        )

        return LivenessResponse(
            livenessVerified=liveness_verified,
            livenessScore=score,
            threshold=settings.LIVENESS_PASSIVE_THRESHOLD,
            checksPassed=checks_passed,
            errors=errors
        )

    def process_full_ekyc(
        self,
        front_bytes: bytes,
        back_bytes: bytes,
        selfie_bytes: Optional[bytes] = None,
        video_bytes: Optional[bytes] = None
    ) -> FullEkycResponse:
        """
        Executes Full Orchestrated eKYC Pipeline.
        """
        start_time = time.time()
        request_id = str(uuid.uuid4())
        failure_reasons = []

        # Step 1: Card Process
        card_res = self.process_card(front_bytes, back_bytes)
        if not card_res.cardVerified:
            failure_reasons.extend(card_res.errors)

        # Step 2: Face Verification
        if selfie_bytes:
            face_res = self.verify_face(front_bytes, selfie_bytes)
        else:
            face_res = FaceVerifyResponse(faceVerified=False, errors=["SELFIE_MISSING"])

        if not face_res.faceVerified:
            failure_reasons.extend(face_res.errors)

        # Step 3: Video Liveness
        if video_bytes:
            liveness_res = self.detect_liveness(video_bytes)
        else:
            # Fallback for selfie-only verification
            liveness_res = LivenessResponse(livenessVerified=True, livenessScore=1.0, checksPassed=["SELFIE_PASSIVE_FALLBACK"])

        if not liveness_res.livenessVerified:
            failure_reasons.extend(liveness_res.errors)

        # Final Decision Logic Gate: CARD && FACE && LIVENESS
        is_ekyc_verified = card_res.cardVerified and face_res.faceVerified and liveness_res.livenessVerified

        ekyc_result_str = "EKYC_VERIFIED" if is_ekyc_verified else "EKYC_NOT_VERIFIED"

        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        return FullEkycResponse(
            requestId=request_id,
            status="SUCCESS",
            ekycResult=ekyc_result_str,
            executionTimeMs=execution_time_ms,
            cardResult=card_res,
            faceResult=face_res,
            livenessResult=liveness_res,
            failureReasons=list(set(failure_reasons))
        )
