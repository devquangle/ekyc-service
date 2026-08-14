import cv2
import numpy as np
import pytest
from unittest.mock import MagicMock

from core.anti_spoof_engine import AntiSpoofEngine, crop_face_roi_with_scale
from core.liveness_engine import LivenessEngine
from core.face_engine import FaceEngine
from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from schemas.face import BoundingBoxInfo


def test_crop_face_roi_with_scale():
    """
    Tests expanded Face ROI cropping with scale 2.7x and 4.0x,
    ensuring correct shape (80, 80) and boundary reflection padding.
    """
    img = np.full((300, 400, 3), 120, dtype=np.uint8)
    # Face near the corner to trigger padding
    bbox = (10, 10, 80, 90)

    roi_27 = crop_face_roi_with_scale(img, bbox, scale=2.7, target_size=(80, 80))
    assert roi_27 is not None
    assert roi_27.shape == (80, 80, 3)

    roi_40 = crop_face_roi_with_scale(img, bbox, scale=4.0, target_size=(80, 80))
    assert roi_40 is not None
    assert roi_40.shape == (80, 80, 3)

    # Test None / empty input
    assert crop_face_roi_with_scale(None, bbox) is None
    assert crop_face_roi_with_scale(img, ()) is None


def test_anti_spoof_engine_fallback_texture():
    """
    Tests AntiSpoofEngine multi-metric texture fallback when models are not loaded.
    Supports both full-frame and face_bbox modes.
    """
    engine = AntiSpoofEngine("nonexistent1.onnx", "nonexistent2.onnx")

    # 1. Uniform black frame -> score 0.0
    black_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    score_black = engine.predict(black_frame)
    assert score_black == 0.0

    # 2. Frame with natural texture -> score in [0.0, 1.0]
    textured_frame = np.random.randint(50, 200, (200, 200, 3), dtype=np.uint8)
    score_textured = engine.predict(textured_frame, face_bbox=(20, 20, 150, 150))
    assert 0.0 <= score_textured <= 1.0

    # 3. None / empty frame
    assert engine.predict(None) == 0.0


def test_liveness_engine_fails_safely_on_no_face():
    """
    Tests that LivenessEngine strictly returns False when no face/eyes are detected
    and never produces fake passes.
    """
    engine = LivenessEngine()

    # Empty frames
    assert engine._detect_blink([]) is False

    # Frames without faces (plain gray image)
    dummy_frames = [np.full((200, 200, 3), 128, dtype=np.uint8) for _ in range(5)]
    assert engine._detect_blink(dummy_frames) is False


def test_liveness_engine_video_validation():
    """
    Tests LivenessEngine video byte validation.
    """
    engine = LivenessEngine()

    # 1. Empty video
    verified, score, passed, errs = engine.analyze_video(b"")
    assert verified is False
    assert "VIDEO_EMPTY" in errs

    # 2. Corrupt video bytes
    verified, score, passed, errs = engine.analyze_video(b"NotARealVideoBytes")
    assert verified is False
    assert "VIDEO_INVALID" in errs


def test_face_engine_extract_embedding_pipeline(monkeypatch):
    """
    Tests FaceEngine full pipeline: detect -> align -> embedding -> L2-norm.
    """
    monkeypatch.setattr(FaceEngine, "_initialize_model", lambda self: None)
    face_engine = FaceEngine()

    # Mock recognition model returning 512-d vector
    mock_app = MagicMock()
    mock_rec = MagicMock()
    mock_rec.get_feat.return_value = np.ones(512, dtype=np.float32)
    mock_app.models = {'recognition': mock_rec}
    face_engine.verification_service.embedding_service.face_app = mock_app

    # Mock face crop with landmarks
    mock_card_ext = MagicMock()
    mock_card_ext.extract_face.return_value = (
        np.full((100, 100, 3), 120, dtype=np.uint8),
        np.array([[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]], dtype=np.float32),
        BoundingBoxInfo(detected=True, bbox=[10, 10, 110, 110], width=100, height=100, detectionScore=0.95),
        []
    )
    face_engine.verification_service.card_extractor = mock_card_ext

    dummy_img = np.full((300, 400, 3), 100, dtype=np.uint8)
    vec, count = face_engine.extract_face_embedding(dummy_img)

    assert vec is not None
    assert count == 1
    assert len(vec) == 512
    # Verify L2 unit norm
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_qr_and_mrz_engine_wrappers():
    """
    Tests QR and MRZ engine typed wrappers.
    """
    mrz_engine = MrzEngine()
    # Test check digit
    cd = mrz_engine.compute_check_digit("041004")
    assert isinstance(cd, int)
    assert cd == 7

    qr_engine = QrEngine()
    parsed = qr_engine.parse_qr_string("086173011002|086173011002|TRAN THI UT|01011973|Nu|Tan Binh, Chau Thanh, Dong Thap|01012021")
    assert parsed is not None
    assert parsed.get("identityNumber") == "086173011002"
