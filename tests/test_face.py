import numpy as np
from core.face_engine import FaceEngine


def test_face_similarity_calculation():
    engine = FaceEngine()

    # Identical vectors -> similarity 1.0
    vec1 = np.random.randn(512).astype(np.float32)
    vec1 = vec1 / np.linalg.norm(vec1)

    sim = engine.calculate_similarity(vec1, vec1)
    assert abs(sim - 1.0) < 1e-4

    # Orthogonal vectors -> similarity ~ 0.0
    vec2 = np.zeros(512, dtype=np.float32)
    vec2[0] = 1.0
    vec3 = np.zeros(512, dtype=np.float32)
    vec3[1] = 1.0

    sim_ortho = engine.calculate_similarity(vec2, vec3)
    assert abs(sim_ortho - 0.0) < 1e-4


def test_crop_portrait_from_card(dummy_card_front_image):
    engine = FaceEngine()
    cropped = engine.crop_portrait_from_card(dummy_card_front_image)
    assert cropped is not None
    assert cropped.shape[0] > 0 and cropped.shape[1] > 0
