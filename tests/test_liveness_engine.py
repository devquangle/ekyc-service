import cv2
import numpy as np
import pytest
from core.liveness_engine import LivenessEngine


def test_passive_frame_analysis():
    """
    Tests passive frame texture and blur analysis.
    """
    engine = LivenessEngine()

    # Generate synthetic textured frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    score = engine._analyze_passive_frame(frame)

    assert 0.0 <= score <= 1.0


def test_empty_video_analysis():
    """
    Tests that empty video payload fails safely with VIDEO_EMPTY error code.
    """
    engine = LivenessEngine()
    liveness_verified, score, checks, errors = engine.analyze_video(b"")

    assert liveness_verified is False
    assert "VIDEO_EMPTY" in errors


def test_invalid_video_bytes():
    """
    Tests that corrupted/invalid video bytes fail gracefully with VIDEO_INVALID error code.
    """
    engine = LivenessEngine()
    liveness_verified, score, checks, errors = engine.analyze_video(b"CorruptedInvalidVideoBytesHeader")

    assert liveness_verified is False
    assert "VIDEO_INVALID" in errors
