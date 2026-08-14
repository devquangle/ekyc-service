import os
import io
import cv2
import pytest
from fastapi.testclient import TestClient
from main import app


def get_image_path(filename: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "image", filename)


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_client_card_process_multipart(client):
    front_path = get_image_path("cccd_c_mt.jpg")
    back_path = get_image_path("cccd_c_ms.jpg")

    if not os.path.exists(front_path) or not os.path.exists(back_path):
        pytest.skip("Test images not found")

    with open(front_path, "rb") as f_front, open(back_path, "rb") as f_back:
        files = {
            "front_image": ("front.jpg", f_front, "image/jpeg"),
            "back_image": ("back.jpg", f_back, "image/jpeg")
        }
        response = client.post("/api/v1/card/process", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "cardVerified" in data
    assert "cardType" in data
    assert "extractedData" in data
    assert "visualRegions" in data or "visual_regions" in data


def test_client_face_verify_multipart(client):
    front_path = get_image_path("cccd_c_mt.jpg")
    if not os.path.exists(front_path):
        pytest.skip("Test image not found")

    with open(front_path, "rb") as f1, open(front_path, "rb") as f2:
        files = {
            "card_portrait": ("card.jpg", f1, "image/jpeg"),
            "selfie_image": ("selfie.jpg", f2, "image/jpeg")
        }
        response = client.post("/api/v1/face/verify", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "faceVerified" in data
    assert "similarityScore" in data


def test_client_full_ekyc_multipart(client):
    front_path = get_image_path("cccd_c_mt.jpg")
    back_path = get_image_path("cccd_c_ms.jpg")

    if not os.path.exists(front_path) or not os.path.exists(back_path):
        pytest.skip("Test images not found")

    with open(front_path, "rb") as f_front, open(back_path, "rb") as f_back, open(front_path, "rb") as f_selfie:
        files = {
            "front_image": ("front.jpg", f_front, "image/jpeg"),
            "back_image": ("back.jpg", f_back, "image/jpeg"),
            "selfie_image": ("selfie.jpg", f_selfie, "image/jpeg")
        }
        response = client.post("/api/v1/ekyc/full", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "requestId" in data
    assert "status" in data
    assert "ekycResult" in data
