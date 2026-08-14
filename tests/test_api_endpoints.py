import os
import base64
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module")
def real_images_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "image")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_card_multipart_files(real_images_dir, client):
    """
    Test POST /api/v1/ekyc/card with standard multipart files.
    """
    front_path = os.path.join(real_images_dir, "cccd_c_mt.jpg")
    back_path = os.path.join(real_images_dir, "cccd_c_ms.jpg")

    if not os.path.exists(front_path):
        pytest.skip("Test images not present")

    with open(front_path, "rb") as f_front, open(back_path, "rb") as f_back:
        files = {
            "front_image": ("cccd_c_mt.jpg", f_front, "image/jpeg"),
            "back_image": ("cccd_c_ms.jpg", f_back, "image/jpeg")
        }
        response = client.post("/api/v1/ekyc/card", files=files)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["success"] is True
    assert data["cardVerified"] is True
    assert data["extractedData"]["identityNumber"] == "087204000897"


def test_api_card_string_and_json_payloads(real_images_dir, client):
    """
    Test POST /api/v1/ekyc/card with Base64 JSON and Form String payloads
    to verify that 422 Unprocessable Entity is completely eliminated.
    """
    front_path = os.path.join(real_images_dir, "cccd_c_mt.jpg")
    if not os.path.exists(front_path):
        pytest.skip("Test images not present")

    with open(front_path, "rb") as f:
        f_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    # 1. JSON Base64 Payload
    r_json = client.post("/api/v1/ekyc/card", json={"front_image": f_b64})
    assert r_json.status_code == 200, f"JSON failed: {r_json.status_code} - {r_json.text}"
    assert r_json.json()["extractedData"]["identityNumber"] == "087204000897"

    # 2. Form Data String Payload (simulating string data sent in form field)
    r_form = client.post("/api/v1/ekyc/card", data={"front_image": f_b64})
    assert r_form.status_code == 200, f"Form text failed: {r_form.status_code} - {r_form.text}"
    assert r_form.json()["extractedData"]["identityNumber"] == "087204000897"
