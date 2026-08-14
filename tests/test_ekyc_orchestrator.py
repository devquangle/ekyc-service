import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from services.ekyc_orchestrator import EkycOrchestrator
from schemas.card import CardProcessResponse, ExtractedCardData, QualityChecks, CrossValidationResult
from schemas.face import FaceVerifyResponse
from schemas.liveness import LivenessResponse
from schemas.enums import CardType, VerificationDecision, EkycOutcome, EkycExecutionStatus


@pytest.fixture
def mock_card_processor():
    processor = MagicMock()
    return processor


@pytest.fixture
def mock_card_validator():
    validator = MagicMock()
    return validator


@pytest.fixture
def mock_face_engine():
    engine = MagicMock()
    return engine


@pytest.fixture
def mock_liveness_engine():
    engine = MagicMock()
    return engine


@pytest.fixture
def orchestrator(mock_card_processor, mock_card_validator, mock_face_engine, mock_liveness_engine):
    return EkycOrchestrator(
        card_processor=mock_card_processor,
        card_validator=mock_card_validator,
        face_engine=mock_face_engine,
        liveness_engine=mock_liveness_engine
    )


def test_card_validation_warnings_do_not_veto_card_verified(orchestrator, mock_card_processor, mock_card_validator):
    """
    Ensure non-fatal warnings (e.g. MRZ checksum warning when QR is valid) do NOT veto cardVerified.
    """
    dummy_img = np.zeros((300, 400, 3), dtype=np.uint8)
    _, dummy_bytes = cv2.imencode('.jpg', dummy_img)
    raw_bytes = dummy_bytes.tobytes()

    extracted = ExtractedCardData(identityNumber="087204000897", fullName="HUYNH QUANG LE")
    quality = QualityChecks(isBlur=False, hasGlare=False, isCropped=False)
    mock_card_processor.process.return_value = (
        CardType.CCCD_OLD, 0.85, extracted, None, None, quality, [], {}
    )

    # CardValidator marks card_verified=True, but returns a warning in val_errors
    mock_card_validator.validate.return_value = (
        True,
        CrossValidationResult(ocrMatchMrz=True),
        ["CARD_DATA_WARNING_MRZ_CHECKSUM_INVALID"]
    )

    res = orchestrator.process_card(raw_bytes)
    assert res.cardVerified is True
    assert "CARD_DATA_WARNING_MRZ_CHECKSUM_INVALID" in res.errors


def test_fail_fast_short_circuit_on_unverified_card(orchestrator, mock_card_processor, mock_card_validator, mock_face_engine, mock_liveness_engine):
    """
    Fail-Fast Gate 1: If card verification fails, pipeline stops immediately without calling face/liveness.
    """
    dummy_img = np.zeros((300, 400, 3), dtype=np.uint8)
    _, dummy_bytes = cv2.imencode('.jpg', dummy_img)
    raw_bytes = dummy_bytes.tobytes()

    mock_card_processor.process.return_value = (
        CardType.CCCD_OLD, 0.85, ExtractedCardData(), None, None, QualityChecks(isBlur=True), [], {}
    )
    mock_card_validator.validate.return_value = (
        False, CrossValidationResult(), ["CARD_BLURRY", "CARD_NOT_DETECTED"]
    )

    full_res = orchestrator.process_full_ekyc(
        front_bytes=raw_bytes,
        back_bytes=None,
        selfie_bytes=raw_bytes,
        video_bytes=b"dummy_video"
    )

    assert full_res.status == EkycExecutionStatus.SUCCESS
    assert full_res.ekycResult == EkycOutcome.EKYC_NOT_VERIFIED
    assert full_res.cardResult is not None
    assert full_res.cardResult.cardVerified is False
    assert full_res.faceResult is None
    assert full_res.livenessResult is None
    assert mock_face_engine.verify_faces.call_count == 0
    assert mock_liveness_engine.analyze_video.call_count == 0


def test_fail_fast_short_circuit_on_unverified_face(orchestrator, mock_card_processor, mock_card_validator, mock_face_engine, mock_liveness_engine):
    """
    Fail-Fast Gate 2: If face verification fails, pipeline stops immediately without calling video liveness.
    """
    dummy_img = np.zeros((300, 400, 3), dtype=np.uint8)
    _, dummy_bytes = cv2.imencode('.jpg', dummy_img)
    raw_bytes = dummy_bytes.tobytes()

    mock_card_processor.process.return_value = (
        CardType.CCCD_OLD, 0.85, ExtractedCardData(identityNumber="087204000897", fullName="HUYNH QUANG LE"),
        None, None, QualityChecks(isBlur=False), [], {}
    )
    mock_card_validator.validate.return_value = (
        True, CrossValidationResult(), []
    )
    mock_face_engine.verify_faces.return_value = FaceVerifyResponse(
        faceVerified=False,
        similarityScore=0.35,
        threshold=0.60,
        decision=VerificationDecision.MISMATCH,
        margin=-0.25,
        errors=["FACE_MISMATCH"]
    )

    full_res = orchestrator.process_full_ekyc(
        front_bytes=raw_bytes,
        back_bytes=None,
        selfie_bytes=raw_bytes,
        video_bytes=b"dummy_video"
    )

    assert full_res.status == EkycExecutionStatus.SUCCESS
    assert full_res.ekycResult == EkycOutcome.EKYC_NOT_VERIFIED
    assert full_res.cardResult.cardVerified is True
    assert full_res.faceResult.faceVerified is False
    assert full_res.livenessResult is None
    assert mock_liveness_engine.analyze_video.call_count == 0


def test_best_frame_extraction_from_video_fallback(orchestrator, mock_card_processor, mock_card_validator, mock_face_engine, mock_liveness_engine):
    """
    If selfie_bytes is None, orchestrator extracts the best frame from video.
    """
    dummy_img = np.zeros((300, 400, 3), dtype=np.uint8)
    _, dummy_bytes = cv2.imencode('.jpg', dummy_img)
    raw_bytes = dummy_bytes.tobytes()

    mock_card_processor.process.return_value = (
        CardType.CCCD_OLD, 0.85, ExtractedCardData(identityNumber="087204000897", fullName="HUYNH QUANG LE"),
        None, None, QualityChecks(isBlur=False), [], {}
    )
    mock_card_validator.validate.return_value = (
        True, CrossValidationResult(), []
    )
    mock_face_engine.verify_faces.return_value = FaceVerifyResponse(
        faceVerified=True,
        similarityScore=0.88,
        threshold=0.60,
        decision=VerificationDecision.MATCH,
        margin=0.28,
        errors=[]
    )
    mock_liveness_engine.analyze_video.return_value = (
        True, 0.95, ["PASSIVE_TEXTURE_CHECK"], []
    )

    best_frame_mock = np.ones((200, 200, 3), dtype=np.uint8) * 128

    with patch.object(orchestrator, 'extract_best_frame_from_video', return_value=best_frame_mock) as mock_extract:
        full_res = orchestrator.process_full_ekyc(
            front_bytes=raw_bytes,
            back_bytes=None,
            selfie_bytes=None,  # No selfie provided!
            video_bytes=b"dummy_video_bytes"
        )

        mock_extract.assert_called_once_with(b"dummy_video_bytes")
        assert full_res.ekycResult == EkycOutcome.EKYC_VERIFIED
        assert full_res.cardResult.cardVerified is True
        assert full_res.faceResult.faceVerified is True
        assert full_res.livenessResult.livenessVerified is True
