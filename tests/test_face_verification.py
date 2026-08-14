import numpy as np
import pytest
from unittest.mock import MagicMock
from schemas.face import BoundingBoxInfo, FaceQualityMetrics, FaceVerifyResponse
from core.face_verification.face_alignment_service import FaceAlignmentService
from core.face_verification.face_embedding_service import FaceEmbeddingService
from core.face_verification.face_quality_service import FaceQualityService
from core.face_verification.card_face_extractor import CardFaceExtractor
from core.face_verification.selfie_face_extractor import SelfieFaceExtractor
from core.face_verification.face_verification_service import FaceVerificationService


def test_face_alignment_relative_coordinate_correction():
    """
    Tests 5-point face alignment with global landmarks on a cropped ROI
    to ensure landmarks are shifted correctly and no black/distorted crop occurs.
    """
    aligner = FaceAlignmentService()

    # Create dummy face crop 120x100
    face_crop = np.full((120, 100, 3), 180, dtype=np.uint8)

    # Global landmarks from full image where face bbox was [50, 60, 150, 180]
    bbox = [50, 60, 150, 180]
    global_kps = np.array([
        [50 + 30, 60 + 40],   # Left eye
        [50 + 70, 60 + 40],   # Right eye
        [50 + 50, 60 + 60],   # Nose
        [50 + 35, 60 + 80],   # Left mouth
        [50 + 65, 60 + 80],   # Right mouth
    ], dtype=np.float32)

    aligned = aligner.align_face(face_crop, landmarks=global_kps, bbox=bbox, target_size=(112, 112))

    assert aligned is not None
    assert aligned.shape == (112, 112, 3)
    # Ensure image is not completely black
    assert np.mean(aligned) > 50


def test_face_alignment_fallback_resize():
    """
    Tests fallback direct resize when landmarks are None.
    """
    aligner = FaceAlignmentService()
    dummy_img = np.zeros((80, 60, 3), dtype=np.uint8)
    aligned = aligner.align_face(dummy_img, landmarks=None, target_size=(112, 112))

    assert aligned is not None
    assert aligned.shape == (112, 112, 3)


def test_selfie_extractor_strictly_rejects_missing_face():
    """
    Tests that SelfieFaceExtractor strictly rejects empty/black/non-face images
    and never falls back to returning the full image.
    """
    extractor = SelfieFaceExtractor(face_app=None)
    # A plain black image has no face
    black_img = np.zeros((300, 300, 3), dtype=np.uint8)
    crop, kps, bbox_info, errors = extractor.extract_face(black_img)

    assert crop is None
    assert kps is None
    assert bbox_info.detected is False
    assert "SELFIE_FACE_NOT_FOUND" in errors


def test_face_embedding_l2_normalization():
    """
    Tests FaceEmbeddingService L2 unit normalization.
    """
    # Create mock recognition model returning raw 512-d vector
    mock_app = MagicMock()
    raw_vec = np.random.randn(512).astype(np.float32) * 5.0
    mock_rec = MagicMock()
    mock_rec.get_feat.return_value = raw_vec
    mock_app.models = {'recognition': mock_rec}

    embedding_service = FaceEmbeddingService(face_app=mock_app)
    dummy_face = np.full((112, 112, 3), 128, dtype=np.uint8)

    norm_vec, dim, norm, errors = embedding_service.extract_embedding(dummy_face)

    assert norm_vec is not None
    assert dim == 512
    assert abs(norm - 1.0) < 1e-5
    assert len(errors) == 0


def test_cosine_similarity_calculation():
    """
    Tests cosine similarity calculation for identical and orthogonal vectors.
    """
    service = FaceEmbeddingService(face_app=None)

    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    sim_identical, err1 = service.calculate_cosine_similarity(v1, v2)
    assert abs(sim_identical - 1.0) < 1e-5
    assert len(err1) == 0

    sim_orthogonal, err2 = service.calculate_cosine_similarity(v1, v3)
    assert abs(sim_orthogonal - 0.0) < 1e-5
    assert len(err2) == 0


def test_face_quality_metrics_and_pose():
    """
    Tests FaceQualityService blur score, brightness, face area, and Euler angle estimation.
    """
    quality_service = FaceQualityService()
    face_img = np.full((100, 100, 3), 150, dtype=np.uint8)
    kps = np.array([
        [30, 40],
        [70, 40],
        [50, 60],
        [35, 80],
        [65, 80]
    ], dtype=np.float32)

    metrics = quality_service.analyze_quality(face_img, landmarks=kps)

    assert metrics.brightness == 150.0
    assert metrics.faceSize == 10000
    assert isinstance(metrics.yaw, float)
    assert isinstance(metrics.pitch, float)
    assert isinstance(metrics.roll, float)


def test_face_verification_full_pipeline_mock():
    """
    Tests end-to-end FaceVerificationService orchestration with matching mock faces.
    """
    mock_app = MagicMock()
    vec = np.ones(512, dtype=np.float32)
    mock_rec = MagicMock()
    mock_rec.get_feat.return_value = vec
    mock_app.models = {'recognition': mock_rec}

    # Mock extractors returning detected face
    mock_card_ext = MagicMock()
    mock_card_ext.extract_face.return_value = (
        np.full((100, 100, 3), 120, dtype=np.uint8),
        np.array([[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]], dtype=np.float32),
        BoundingBoxInfo(detected=True, bbox=[10, 10, 110, 110], width=100, height=100, detectionScore=0.98),
        []
    )

    mock_selfie_ext = MagicMock()
    mock_selfie_ext.extract_face.return_value = (
        np.full((100, 100, 3), 120, dtype=np.uint8),
        np.array([[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]], dtype=np.float32),
        BoundingBoxInfo(detected=True, bbox=[20, 20, 120, 120], width=100, height=100, detectionScore=0.99),
        []
    )

    service = FaceVerificationService(
        card_extractor=mock_card_ext,
        selfie_extractor=mock_selfie_ext,
        face_app=mock_app
    )

    card_img = np.full((300, 400, 3), 100, dtype=np.uint8)
    selfie_img = np.full((300, 400, 3), 100, dtype=np.uint8)

    response = service.verify_faces(card_img, selfie_img)

    assert response.faceVerified is True
    assert response.decision == "MATCH"
    assert response.similarityScore >= 0.60
    assert response.cardFaceInfo.detected is True
    assert response.selfieFaceInfo.detected is True
