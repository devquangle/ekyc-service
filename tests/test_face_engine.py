import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock

from core.face_engine import FaceEngine
from schemas.face import BoundingBoxInfo, FaceVerifyResponse
from schemas.enums import VerificationDecision


@pytest.fixture(autouse=True)
def mock_face_engine_model(monkeypatch):
    monkeypatch.setattr(FaceEngine, "_initialize_model", lambda self: None)


def test_face_similarity_calculation():
    """
    Tests cosine similarity calculation between 512-dimensional embedding vectors.
    """
    engine = FaceEngine()

    # 1. Identical normalized vectors -> similarity = 1.0
    vec1 = np.random.randn(512).astype(np.float32)
    vec1 = vec1 / np.linalg.norm(vec1)

    sim = engine.calculate_similarity(vec1, vec1)
    assert abs(sim - 1.0) < 1e-4

    # 2. Orthogonal vectors -> similarity ~ 0.0
    vec2 = np.zeros(512, dtype=np.float32)
    vec2[0] = 1.0
    vec3 = np.zeros(512, dtype=np.float32)
    vec3[1] = 1.0

    sim_ortho = engine.calculate_similarity(vec2, vec3)
    assert abs(sim_ortho - 0.0) < 1e-4

    # 3. Opposite vectors -> similarity = -1.0
    sim_opp = engine.calculate_similarity(vec2, -vec2)
    assert abs(sim_opp - (-1.0)) < 1e-4


def test_crop_portrait_from_card_with_mock():
    """
    Tests crop_portrait_from_card using mocked extractor to ensure predictable unit test behavior.
    """
    engine = FaceEngine()
    dummy_card = np.full((630, 1000, 3), 200, dtype=np.uint8)

    # Mock successful face detection on card
    mock_crop = np.full((150, 120, 3), 180, dtype=np.uint8)
    engine.verification_service.card_extractor.extract_face = MagicMock(
        return_value=(mock_crop, np.zeros((5, 2)), BoundingBoxInfo(detected=True, bbox=[80, 180, 200, 330]), [])
    )

    cropped = engine.crop_portrait_from_card(dummy_card)
    assert cropped is not None
    assert cropped.shape == (150, 120, 3)


def test_crop_portrait_no_face_returns_none():
    """
    Tests that crop_portrait_from_card returns None when no face is found or image is blank.
    """
    engine = FaceEngine()
    blank_img = np.zeros((300, 400, 3), dtype=np.uint8)

    # Blank image with no face
    cropped = engine.crop_portrait_from_card(blank_img)
    assert cropped is None

    # None image
    assert engine.crop_portrait_from_card(None) is None


def test_verify_faces_delegation():
    """
    Tests that FaceEngine.verify_faces correctly delegates to FaceVerificationService.
    """
    engine = FaceEngine()
    mock_card = np.full((100, 100, 3), 120, dtype=np.uint8)
    mock_selfie = np.full((100, 100, 3), 130, dtype=np.uint8)

    expected_resp = FaceVerifyResponse(
        faceVerified=True,
        similarityScore=0.92,
        threshold=0.60,
        decision=VerificationDecision.MATCH,
        margin=0.32,
        errors=[]
    )
    engine.verification_service.verify_faces = MagicMock(return_value=expected_resp)

    res = engine.verify_faces(mock_card, mock_selfie)
    assert res.faceVerified is True
    assert res.similarityScore == 0.92
    assert res.decision == VerificationDecision.MATCH
