import os
import time
import uuid
import tempfile
from typing import Optional, List, Union
import cv2
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
from schemas.enums import CardType, VerificationDecision, EkycOutcome, EkycExecutionStatus
from utils.logger import logger
from utils.image_utils import decode_image_bytes


class EkycOrchestrator:
    """
    Enterprise-grade eKYC Orchestrator executing the complete 3-Stage Pipeline
    with strict Fail-Fast resource protection, buffer reuse, automatic Best-Frame extraction,
    and granular performance metrics.
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

    def process_card(
        self,
        front_bytes: Union[bytes, np.ndarray],
        back_bytes: Optional[Union[bytes, np.ndarray]] = None
    ) -> CardProcessResponse:
        """
        Executes Card Extraction & Validation Flow.
        Accepts raw image bytes or already-decoded np.ndarray buffers.
        """
        front_img = front_bytes if isinstance(front_bytes, np.ndarray) else (decode_image_bytes(front_bytes) if front_bytes else None)
        back_img = back_bytes if isinstance(back_bytes, np.ndarray) else (decode_image_bytes(back_bytes) if (back_bytes and len(back_bytes) > 0) else None)

        if front_img is None:
            return CardProcessResponse(
                cardVerified=False,
                cardType=CardType.UNKNOWN,
                cardTypeConfidence=0.0,
                errors=["INVALID_IMAGE_FORMAT"]
            )

        (
            card_type,
            card_type_conf,
            extracted_data,
            qr_data,
            mrz_data,
            quality_checks,
            field_metadata,
            visual_regions
        ) = self.card_processor.process(front_img, back_img)

        card_verified, cross_val_result, val_errors = self.card_validator.validate(
            extracted_data, qr_data, mrz_data, card_type
        )

        all_errors = list(val_errors)
        if quality_checks and quality_checks.isBlur:
            all_errors.append("CARD_BLURRY")

        # Distinct fatal errors from non-fatal warnings
        fatal_error_set = {
            "CARD_DATA_MISMATCH_IDENTITY_NUMBER",
            "CARD_DATA_MISMATCH_FULL_NAME",
            "CARD_EXPIRED",
            "INVALID_IMAGE_FORMAT",
            "CARD_NOT_DETECTED",
            "CARD_BLURRY",
            "OCR_FAILED"
        }
        has_fatal_error = any(err in fatal_error_set for err in all_errors)
        final_card_verified = card_verified and not has_fatal_error

        unique_errors = list(dict.fromkeys(all_errors))
        field_meta_list = list(field_metadata.values()) if isinstance(field_metadata, dict) else (field_metadata or [])

        return CardProcessResponse(
            cardVerified=final_card_verified,
            cardType=card_type,
            cardTypeConfidence=card_type_conf,
            extractedData=extracted_data,
            crossValidation=cross_val_result,
            qualityChecks=quality_checks,
            visualRegions=visual_regions,
            visual_regions=visual_regions,
            fieldMetadata=field_meta_list,
            errors=unique_errors
        )

    def verify_face(
        self,
        card_portrait_bytes: Optional[Union[bytes, np.ndarray]] = None,
        selfie_bytes: Optional[Union[bytes, np.ndarray]] = None,
        card_image_bytes: Optional[Union[bytes, np.ndarray]] = None,
        selfie_image_bytes: Optional[Union[bytes, np.ndarray]] = None,
    ) -> FaceVerifyResponse:
        """
        Executes Face Verification Flow using FaceEngine.
        Accepts raw image bytes or already-decoded np.ndarray buffers.
        """
        c_input = card_portrait_bytes if card_portrait_bytes is not None else card_image_bytes
        s_input = selfie_bytes if selfie_bytes is not None else selfie_image_bytes

        if isinstance(c_input, np.ndarray):
            card_img = c_input
        elif isinstance(c_input, (bytes, bytearray)) and len(c_input) > 0:
            card_img = decode_image_bytes(c_input)
        else:
            card_img = None

        if isinstance(s_input, np.ndarray):
            selfie_img = s_input
        elif isinstance(s_input, (bytes, bytearray)) and len(s_input) > 0:
            selfie_img = decode_image_bytes(s_input)
        else:
            selfie_img = None

        if card_img is None or selfie_img is None:
            missing_errors = []
            if card_img is None:
                missing_errors.append("CARD_IMAGE_MISSING")
            if selfie_img is None:
                missing_errors.append("SELFIE_MISSING")
            return FaceVerifyResponse(
                faceVerified=False,
                similarityScore=0.0,
                threshold=settings.FACE_MATCH_THRESHOLD,
                decision=VerificationDecision.MISMATCH,
                margin=-settings.FACE_MATCH_THRESHOLD,
                errors=missing_errors if missing_errors else ["INVALID_IMAGE_FORMAT"]
            )

        return self.face_engine.verify_faces(card_img, selfie_img)

    def detect_liveness(
        self,
        video_bytes: bytes,
        expected_gestures: Optional[List[str]] = None
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

    def extract_best_frame_from_video(self, video_bytes: bytes) -> Optional[np.ndarray]:
        """
        Extracts the highest-quality face frame from video bytes when selfie image is missing.
        Ranks frames based on face presence, face size, detection confidence, and Laplacian sharpness.
        """
        if not video_bytes or len(video_bytes) == 0:
            return None

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        best_frame = None
        best_score = -1.0

        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                logger.warning("[EKYC_ORCHESTRATOR] Unable to open video stream for frame extraction.")
                return None

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            sample_interval = max(1, int(fps / settings.VIDEO_FRAME_SAMPLING_RATE))

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0 and frame is not None and frame.size > 0:
                    score = self._compute_frame_quality_score(frame)
                    if score > best_score:
                        best_score = score
                        best_frame = frame.copy()

                frame_idx += 1

            cap.release()
        except Exception as e:
            logger.error(f"[EKYC_ORCHESTRATOR] Best frame extraction exception: {str(e)}", exc_info=True)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.warning(f"[EKYC_ORCHESTRATOR] Temp file cleanup warning: {str(e)}")

        if best_frame is not None and best_score > 0.0:
            logger.info(f"[EKYC_ORCHESTRATOR] Extracted best video frame with quality score: {best_score:.2f}")
            return best_frame
        return None

    def _compute_frame_quality_score(self, frame: np.ndarray) -> float:
        """
        Computes frame quality based on face detection and sharpness (Laplacian variance).
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # Minimum sharpness filter
            if laplacian_var < 15.0:
                return 0.0

            # 1. Try InsightFace face detection if available
            if self.face_engine and getattr(self.face_engine, "app", None):
                faces = self.face_engine.app.get(frame)
                if faces and len(faces) == 1:
                    face = faces[0]
                    det_score = float(getattr(face, "det_score", 0.9))
                    bbox = face.bbox
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    area_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
                    return laplacian_var * det_score * (1.0 + area_ratio * 2.0)
                elif faces and len(faces) > 1:
                    # Multiple faces penalty
                    return laplacian_var * 0.1

            # 2. Fallback to OpenCV Haar Cascade / Face detector
            cascade = getattr(self.liveness_engine, "_face_cascade", None)
            if cascade is not None:
                faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                if len(faces) == 1:
                    (x, y, w, h) = faces[0]
                    area_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
                    return laplacian_var * (1.0 + area_ratio * 2.0)
                elif len(faces) > 1:
                    return laplacian_var * 0.1

            # If no face detector available, return raw sharpness
            return laplacian_var
        except Exception:
            return 0.0

    def process_full_ekyc(
        self,
        front_bytes: bytes,
        back_bytes: Optional[bytes] = None,
        selfie_bytes: Optional[bytes] = None,
        video_bytes: Optional[bytes] = None
    ) -> FullEkycResponse:
        """
        Executes Complete 3-Stage eKYC Pipeline with Fail-Fast Resource Protection:
        Stage 1: Card Processing & Business Cross-Validation (Fail-Fast Gate 1)
        Stage 2: 1-1 Face Verification with Best-Frame Fallback (Fail-Fast Gate 2)
        Stage 3: Video Anti-Spoofing & Liveness Detection (Final Gate)
        """
        pipeline_start_time = time.time()
        request_id = str(uuid.uuid4())
        failure_reasons: List[str] = []

        try:
            logger.info(f"[FULL_EKYC] [{request_id}] Starting eKYC Pipeline execution.")

            # Decode front & back images once to reuse buffers across steps
            front_img = decode_image_bytes(front_bytes) if front_bytes else None
            back_img = decode_image_bytes(back_bytes) if (back_bytes and len(back_bytes) > 0) else None

            if front_img is None:
                exec_time_ms = round((time.time() - pipeline_start_time) * 1000, 2)
                return FullEkycResponse(
                    requestId=request_id,
                    status=EkycExecutionStatus.SUCCESS,
                    ekycResult=EkycOutcome.EKYC_NOT_VERIFIED,
                    executionTimeMs=exec_time_ms,
                    cardResult=CardProcessResponse(
                        cardVerified=False,
                        cardType=CardType.UNKNOWN,
                        cardTypeConfidence=0.0,
                        errors=["INVALID_IMAGE_FORMAT"]
                    ),
                    faceResult=None,
                    livenessResult=None,
                    failureReasons=["INVALID_IMAGE_FORMAT"]
                )

            # ==================== STEP 1: CARD PROCESSING ====================
            logger.info(f"[FULL_EKYC] [{request_id}] [STEP 1/3] Processing Card...")
            card_res = self.process_card(front_img, back_img)

            if not card_res.cardVerified:
                failure_reasons.extend(card_res.errors)
                unique_failures = list(dict.fromkeys(failure_reasons))
                exec_time_ms = round((time.time() - pipeline_start_time) * 1000, 2)
                logger.info(f"[FULL_EKYC] [{request_id}] [FAIL-FAST] Short-circuiting at Step 1 (Card Unverified): {unique_failures}")
                return FullEkycResponse(
                    requestId=request_id,
                    status=EkycExecutionStatus.SUCCESS,
                    ekycResult=EkycOutcome.EKYC_NOT_VERIFIED,
                    executionTimeMs=exec_time_ms,
                    cardResult=card_res,
                    faceResult=None,
                    livenessResult=None,
                    failureReasons=unique_failures
                )

            # ==================== STEP 2: FACE VERIFICATION ====================
            logger.info(f"[FULL_EKYC] [{request_id}] [STEP 2/3] Processing Face Verification...")
            selfie_img = None
            if selfie_bytes and len(selfie_bytes) > 0:
                selfie_img = decode_image_bytes(selfie_bytes)
            elif video_bytes and len(video_bytes) > 0:
                logger.info(f"[FULL_EKYC] [{request_id}] Selfie missing: extracting best frame from video...")
                selfie_img = self.extract_best_frame_from_video(video_bytes)

            if selfie_img is not None:
                # Reuse decoded front_img buffer directly (zero re-decoding)
                face_res = self.verify_face(card_portrait_bytes=front_img, selfie_bytes=selfie_img)
            else:
                face_res = FaceVerifyResponse(
                    faceVerified=False,
                    similarityScore=0.0,
                    threshold=settings.FACE_MATCH_THRESHOLD,
                    decision=VerificationDecision.MISMATCH,
                    margin=-settings.FACE_MATCH_THRESHOLD,
                    errors=["SELFIE_MISSING"]
                )

            if not face_res.faceVerified:
                failure_reasons.extend(face_res.errors)
                unique_failures = list(dict.fromkeys(failure_reasons))
                exec_time_ms = round((time.time() - pipeline_start_time) * 1000, 2)
                logger.info(f"[FULL_EKYC] [{request_id}] [FAIL-FAST] Short-circuiting at Step 2 (Face Unverified): {unique_failures}")
                return FullEkycResponse(
                    requestId=request_id,
                    status=EkycExecutionStatus.SUCCESS,
                    ekycResult=EkycOutcome.EKYC_NOT_VERIFIED,
                    executionTimeMs=exec_time_ms,
                    cardResult=card_res,
                    faceResult=face_res,
                    livenessResult=None,
                    failureReasons=unique_failures
                )

            # ==================== STEP 3: VIDEO LIVENESS ====================
            logger.info(f"[FULL_EKYC] [{request_id}] [STEP 3/3] Processing Video Liveness...")
            if video_bytes and len(video_bytes) > 0:
                liveness_res = self.detect_liveness(video_bytes)
            else:
                liveness_res = LivenessResponse(
                    livenessVerified=False,
                    livenessScore=0.0,
                    threshold=settings.LIVENESS_PASSIVE_THRESHOLD,
                    checksPassed=[],
                    errors=["VIDEO_MISSING"]
                )

            if not liveness_res.livenessVerified:
                failure_reasons.extend(liveness_res.errors)

            # ==================== FINAL DECISION GATE ====================
            is_ekyc_verified = card_res.cardVerified and face_res.faceVerified and liveness_res.livenessVerified
            ekyc_result_str = EkycOutcome.EKYC_VERIFIED if is_ekyc_verified else EkycOutcome.EKYC_NOT_VERIFIED
            exec_time_ms = round((time.time() - pipeline_start_time) * 1000, 2)
            unique_failures = list(dict.fromkeys(failure_reasons))

            logger.info(
                f"[FULL_EKYC] [{request_id}] Finished pipeline in {exec_time_ms}ms: "
                f"result={ekyc_result_str}, card={card_res.cardVerified}, "
                f"face={face_res.faceVerified}, liveness={liveness_res.livenessVerified}"
            )

            return FullEkycResponse(
                requestId=request_id,
                status=EkycExecutionStatus.SUCCESS,
                ekycResult=ekyc_result_str,
                executionTimeMs=exec_time_ms,
                cardResult=card_res,
                faceResult=face_res,
                livenessResult=liveness_res,
                failureReasons=unique_failures
            )

        except Exception as e:
            logger.error(f"[FULL_EKYC] [{request_id}] Pipeline unhandled exception: {str(e)}", exc_info=True)
            exec_time_ms = round((time.time() - pipeline_start_time) * 1000, 2)
            return FullEkycResponse(
                requestId=request_id,
                status=EkycExecutionStatus.FAILED,
                ekycResult=EkycOutcome.EKYC_NOT_VERIFIED,
                executionTimeMs=exec_time_ms,
                cardResult=None,
                faceResult=None,
                livenessResult=None,
                failureReasons=[str(e)]
            )
