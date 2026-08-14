import os
import base64
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from main import app
from schemas.card import CardProcessResponse, ExtractedCardData, QualityChecks, VisualRegions
from schemas.face import FaceVerifyResponse
from schemas.liveness import LivenessResponse
from schemas.ekyc import FullEkycResponse


@pytest.fixture(scope="module")
def real_images_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "image")


@pytest.fixture(autouse=True)
def mock_orchestrator():
    """
    Sets up mock orchestrator in app.state to test endpoint input validation,
    non-blocking execution and multi-format payload parsing instantly.
    """
    mock_orch = MagicMock()
    mock_orch.process_card.return_value = CardProcessResponse(
        success=True,
        cardType="CCCD_OLD",
        cardVerified=True,
        confidence=0.98,
        extractedData=ExtractedCardData(
            identityNumber="086173011002",
            fullName="TRAN THI UT",
            dateOfBirth="1973-01-01",
            gender="Nữ",
            nationality="Việt Nam",
            placeOfOrigin="Tân Lược, Bình Tân, Vĩnh Long",
            placeOfResidence="Tân Bình, Châu Thành, Đồng Tháp",
            dateOfExpiry="2033-01-01"
        ),
        qualityChecks=QualityChecks(blurScore=150.0, isBlurry=False, glareDetected=False, darkDetected=False, passed=True),
        visualRegions=VisualRegions(portrait=[10.0, 10.0, 100.0, 100.0]),
        fieldMetadata=[]
    )
    mock_orch.verify_face.return_value = FaceVerifyResponse(
        faceVerified=True,
        similarityScore=0.92,
        threshold=0.60,
        decision="MATCH",
        margin=0.32,
        errors=[]
    )
    mock_orch.detect_liveness.return_value = LivenessResponse(
        livenessVerified=True,
        livenessScore=0.95,
        isLive=True,
        isRealPerson=True,
        antiSpoofScore=0.96,
        spoofDetected=False,
        checksPassed=["BLINK", "HEAD_TURN"],
        errors=[]
    )
    mock_orch.process_full_ekyc.return_value = FullEkycResponse(
        status="SUCCESS",
        ekycResult="EKYC_VERIFIED",
        executionTimeMs=120.5,
        cardResult=mock_orch.process_card.return_value,
        faceResult=mock_orch.verify_face.return_value,
        livenessResult=mock_orch.detect_liveness.return_value,
        failureReasons=[]
    )
    app.state.orchestrator = mock_orch
    return mock_orch


def test_api_card_multipart_and_json(real_images_dir):
    """
    1. Tests POST /api/v1/ekyc/card with Multipart File, Form text, and JSON Base64.
    """
    front_path = os.path.join(real_images_dir, "cccd_c_mt.jpg")
    back_path = os.path.join(real_images_dir, "cccd_c_ms.jpg")

    if not os.path.exists(front_path):
        pytest.skip("Test images not present")

    client = TestClient(app)

    # A. Multipart File Upload
    with open(front_path, "rb") as f_front, open(back_path, "rb") as f_back:
        files = {
            "front_image": ("cccd_c_mt.jpg", f_front, "image/jpeg"),
            "back_image": ("cccd_c_ms.jpg", f_back, "image/jpeg")
        }
        r_file = client.post("/api/v1/ekyc/card", files=files)
    assert r_file.status_code == 200
    assert r_file.json()["extractedData"]["identityNumber"] == "086173011002"

    # B. JSON Base64 Payload
    with open(front_path, "rb") as f:
        f_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    r_json = client.post("/api/v1/ekyc/card", json={"front_image": f_b64})
    assert r_json.status_code == 200
    assert r_json.json()["extractedData"]["identityNumber"] == "086173011002"

    # C. Form String Payload
    r_form = client.post("/api/v1/ekyc/card", data={"front_image": f_b64})
    assert r_form.status_code == 200

    # D. Missing required field -> 400 Bad Request (NOT 422)
    r_missing = client.post("/api/v1/ekyc/card", json={"back_image": f_b64})
    assert r_missing.status_code == 400
    assert "Missing required field" in r_missing.json()["detail"]


def test_api_face_verify_multi_format(real_images_dir):
    """
    2. Tests POST /api/v1/ekyc/face/verify with JSON Base64 and Multipart.
    """
    front_path = os.path.join(real_images_dir, "cccd_c_mt.jpg")
    if not os.path.exists(front_path):
        pytest.skip("Test images not present")

    with open(front_path, "rb") as f:
        f_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    client = TestClient(app)

    # JSON Payload
    r_json = client.post(
        "/api/v1/ekyc/face/verify",
        json={"card_portrait": f_b64, "selfie_image": f_b64}
    )
    assert r_json.status_code == 200
    assert r_json.json()["faceVerified"] is True
    assert r_json.json()["decision"] == "MATCH"

    # Missing selfie -> 400 Bad Request
    r_missing = client.post(
        "/api/v1/ekyc/face/verify",
        json={"card_portrait": f_b64}
    )
    assert r_missing.status_code == 400


def test_api_liveness_and_full_ekyc(real_images_dir):
    """
    3. Tests POST /api/v1/ekyc/face/liveness and /api/v1/ekyc/verify with JSON & Form payloads.
    """
    front_path = os.path.join(real_images_dir, "cccd_c_mt.jpg")
    if not os.path.exists(front_path):
        pytest.skip("Test images not present")

    with open(front_path, "rb") as f:
        f_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    dummy_video_b64 = "data:video/mp4;base64," + base64.b64encode(b"RIFF\x00\x00\x00\x00AVI LIST\x00\x00\x00\x00" * 10).decode("utf-8")

    client = TestClient(app)

    # Liveness Endpoint
    r_live = client.post(
        "/api/v1/ekyc/face/liveness",
        json={"video_file": dummy_video_b64, "expected_gestures": "BLINK,TURN_LEFT"}
    )
    assert r_live.status_code == 200
    assert r_live.json()["livenessVerified"] is True

    # Full eKYC Endpoint
    r_ekyc = client.post(
        "/api/v1/ekyc/verify",
        json={
            "front_image": f_b64,
            "selfie_image": f_b64,
            "video_file": dummy_video_b64
        }
    )
    assert r_ekyc.status_code == 200
    assert r_ekyc.json()["status"] == "SUCCESS"
    assert r_ekyc.json()["ekycResult"] == "EKYC_VERIFIED"
