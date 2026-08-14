import cv2
import numpy as np
from core.liveness_engine import LivenessEngine


def test_passive_frame_analysis():
    engine = LivenessEngine()

    # Generate synthetic textured frame
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    score = engine._analyze_passive_frame(frame)

    assert 0.0 <= score <= 1.0


def test_empty_video_analysis():
    engine = LivenessEngine()
    liveness_verified, score, checks, errors = engine.analyze_video(b"")

    assert liveness_verified is False
    assert "VIDEO_EMPTY" in errors
