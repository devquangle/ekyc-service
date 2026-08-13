import numpy as np
import pytest
from config import settings
from services.face_verification import (
    CardFaceExtractor,
    SelfieFaceExtractor,
    FaceQualityService,
    FaceAlignmentService,
    FaceEmbeddingService,
    FaceVerificationService,
)
from schemas.face import BoundingBoxInfo, FaceQualityMetrics


def test_decision_logic_thresholds():
    service = FaceVerificationService()

    # 1. Similarity 0.40 => MISMATCH
    decision, verified, errs = service._evaluate_decision(0.40, 0.45, 0.60, False)
    assert decision == "MISMATCH"
    assert not verified
    assert "FACE_MISMATCH" in errs

    # 2. Similarity 0.50 => BORDERLINE
    decision, verified, errs = service._evaluate_decision(0.50, 0.45, 0.60, False)
    assert decision == "BORDERLINE"
    assert not verified
    assert "FACE_SIMILARITY_BORDERLINE" in errs

    # 3. Similarity 0.5872 => BORDERLINE
    decision, verified, errs = service._evaluate_decision(0.5872, 0.45, 0.60, False)
    assert decision == "BORDERLINE"
    assert not verified
    assert "FACE_SIMILARITY_BORDERLINE" in errs
    margin = round(0.5872 - 0.60, 4)
    assert margin == -0.0128

    # 4. Similarity 0.60 => MATCH
    decision, verified, errs = service._evaluate_decision(0.60, 0.45, 0.60, False)
    assert decision == "MATCH"
    assert verified
    assert len(errs) == 0

    # 5. Similarity 0.75 => MATCH
    decision, verified, errs = service._evaluate_decision(0.75, 0.45, 0.60, False)
    assert decision == "MATCH"
    assert verified
    assert len(errs) == 0


def test_face_too_small():
    extractor = CardFaceExtractor(face_app=None)
    # Create very small 10x10 image
    small_img = np.ones((10, 10, 3), dtype=np.uint8) * 200

    class MockFace:
        bbox = [0, 0, 10, 10]
        det_score = 0.99

    class MockApp:
        def get(self, img):
            return [MockFace()]

    mock_extractor = CardFaceExtractor(face_app=MockApp())
    crop, kps, bbox, errs = mock_extractor.extract_face(small_img)

    assert "CARD_FACE_TOO_SMALL" in errs
    assert bbox.width == 10 and bbox.height == 10


def test_no_face_detected():
    class EmptyApp:
        def get(self, img):
            return []

    extractor = SelfieFaceExtractor(face_app=EmptyApp())
    img = np.ones((200, 200, 3), dtype=np.uint8) * 100
    crop, kps, bbox, errs = extractor.extract_face(img)

    assert "SELFIE_FACE_NOT_FOUND" in errs
    assert not bbox.detected


def test_multiple_faces_detected():
    class MultiFaceApp:
        def get(self, img):
            class F1:
                bbox = [10, 10, 50, 50]
                det_score = 0.9
            class F2:
                bbox = [60, 60, 100, 100]
                det_score = 0.85
            return [F1(), F2()]

    extractor = SelfieFaceExtractor(face_app=MultiFaceApp())
    img = np.ones((200, 200, 3), dtype=np.uint8) * 100
    crop, kps, bbox, errs = extractor.extract_face(img)

    assert "MULTIPLE_FACES_DETECTED" in errs


def test_embedding_dimension_mismatch():
    emb_service = FaceEmbeddingService()
    v1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
    v2 = np.ones(256, dtype=np.float32) / np.sqrt(256)

    sim, errs = emb_service.calculate_cosine_similarity(v1, v2)
    assert "EMBEDDING_DIMENSION_MISMATCH" in errs
    assert sim == 0.0


def test_invalid_embedding_nan_inf():
    emb_service = FaceEmbeddingService()
    img = np.ones((112, 112, 3), dtype=np.uint8)

    class NanApp:
        def get(self, img):
            class F:
                embedding = np.array([np.nan] * 512, dtype=np.float32)
            return [F()]

    nan_service = FaceEmbeddingService(face_app=NanApp())
    vec, dim, norm, errs = nan_service.extract_embedding(img)

    assert "INVALID_EMBEDDING" in errs
    assert vec is None


def test_zero_norm_embedding():
    emb_service = FaceEmbeddingService()
    img = np.ones((112, 112, 3), dtype=np.uint8)

    class ZeroApp:
        def get(self, img):
            class F:
                embedding = np.zeros(512, dtype=np.float32)
            return [F()]

    zero_service = FaceEmbeddingService(face_app=ZeroApp())
    vec, dim, norm, errs = zero_service.extract_embedding(img)

    assert "ZERO_NORM_EMBEDDING" in errs
    assert vec is None


def test_full_face_verification_end_to_end_borderline_case(dummy_card_front_image, dummy_selfie_image):
    service = FaceVerificationService()
    res = service.verify_faces(dummy_card_front_image, dummy_selfie_image)

    assert res is not None
    assert hasattr(res, 'decision')
    assert hasattr(res, 'margin')
    assert res.cardFaceInfo is not None
    assert res.selfieFaceInfo is not None
    assert res.cardFaceQuality is not None
    assert res.selfieFaceQuality is not None
