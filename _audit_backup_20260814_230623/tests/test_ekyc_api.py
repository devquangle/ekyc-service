import cv2
import io
import pytest
from unittest.mock import MagicMock
from schemas.card import CardProcessResponse, ExtractedCardData, QualityChecks, VisualRegions
from schemas.face import FaceVerifyResponse
from schemas.liveness import LivenessResponse
from schemas.ekyc import FullEkycResponse
from schemas.enums import EkycOutcome, EkycExecutionStatus, VerificationDecision, CardType


@pytest.fixture(autouse=True)
def mock_api_orchestrator(test_client):
    """
    Mocks app.state.orchestrator for fast, deterministic, CI/CD-compatible API endpoint testing.
    """
    mock_orch = MagicMock()
    mock_orch.process_card.return_value = CardProcessResponse(
        success=True,
        cardType=CardType.CCCD_OLD,
        cardVerified=True,
        confidence=0.98,
        extractedData=ExtractedCardData(
            identityNumber="087204000897",
            fullName="HUYNH QUANG LE",
            dateOfBirth="2004-10-04",
            gender="Nam",
            nationality="Việt Nam",
            placeOfOrigin="Tân Bình, Châu Thành, Đồng Tháp",
            placeOfResidence="Ấp Tây Tân Bình, Châu Thành, Đồng Tháp",
            dateOfExpiry="2029-10-04"
        ),
        qualityChecks=QualityChecks(blurScore=150.0, isBlurry=False, glareDetected=False, darkDetected=False, passed=True),
        visualRegions=VisualRegions(portrait=[10.0, 10.0, 100.0, 100.0]),
        fieldMetadata=[]
    )
    mock_orch.verify_face.return_value = FaceVerifyResponse(
        faceVerified=True,
        similarityScore=0.92,
        threshold=0.60,
        decision=VerificationDecision.MATCH,
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
        checksPassed=["BLINK"],
        errors=[]
    )
    mock_orch.process_full_ekyc.return_value = FullEkycResponse(
        requestId="test-req-123",
        ekycResult=EkycOutcome.EKYC_VERIFIED,
        status=EkycExecutionStatus.SUCCESS,
        cardResult=mock_orch.process_card.return_value,
        faceResult=mock_orch.verify_face.return_value,
        livenessResult=mock_orch.detect_liveness.return_value,
        errors=[],
        executionTimeMs=45.0
    )

    from main import app
    app.state.orchestrator = mock_orch
    yield mock_orch


def test_health_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert "service" in data


def test_card_api_endpoint(test_client, dummy_card_front_image, dummy_card_back_image):
    _, front_buf = cv2.imencode(".jpg", dummy_card_front_image)
    _, back_buf = cv2.imencode(".jpg", dummy_card_back_image)

    files = {
        "front_image": ("front.jpg", io.BytesIO(front_buf.tobytes()), "image/jpeg"),
        "back_image": ("back.jpg", io.BytesIO(back_buf.tobytes()), "image/jpeg")
    }

    response = test_client.post("/api/v1/ekyc/card", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "cardVerified" in data
    assert "cardType" in data
    assert "extractedData" in data
    assert "identityNumber" in data["extractedData"]
    # Verify strict field names constraint
    assert "issueDate" not in data["extractedData"]
    assert "expiryDate" not in data["extractedData"]


def test_face_api_endpoint(test_client, dummy_card_front_image, dummy_selfie_image):
    _, front_buf = cv2.imencode(".jpg", dummy_card_front_image)
    _, selfie_buf = cv2.imencode(".jpg", dummy_selfie_image)

    files = {
        "card_portrait": ("front.jpg", io.BytesIO(front_buf.tobytes()), "image/jpeg"),
        "selfie_image": ("selfie.jpg", io.BytesIO(selfie_buf.tobytes()), "image/jpeg")
    }

    response = test_client.post("/api/v1/ekyc/face/verify", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "faceVerified" in data
    assert "similarityScore" in data
    assert "threshold" in data


def test_full_ekyc_api_endpoint(test_client, dummy_card_front_image, dummy_card_back_image, dummy_selfie_image):
    _, front_buf = cv2.imencode(".jpg", dummy_card_front_image)
    _, back_buf = cv2.imencode(".jpg", dummy_card_back_image)
    _, selfie_buf = cv2.imencode(".jpg", dummy_selfie_image)

    files = {
        "front_image": ("front.jpg", io.BytesIO(front_buf.tobytes()), "image/jpeg"),
        "back_image": ("back.jpg", io.BytesIO(back_buf.tobytes()), "image/jpeg"),
        "selfie_image": ("selfie.jpg", io.BytesIO(selfie_buf.tobytes()), "image/jpeg")
    }

    response = test_client.post("/api/v1/ekyc/verify", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "requestId" in data
    assert "ekycResult" in data
    assert data["ekycResult"] in ["EKYC_VERIFIED", "EKYC_NOT_VERIFIED", EkycOutcome.EKYC_VERIFIED.value]
    assert "cardResult" in data
    assert "faceResult" in data


def test_card_api_endpoint_camel_case(test_client, dummy_card_front_image, dummy_card_back_image):
    """
    Test uploading card using camelCase field names (frontImage, backImage) does NOT throw 422.
    """
    _, front_buf = cv2.imencode(".jpg", dummy_card_front_image)
    _, back_buf = cv2.imencode(".jpg", dummy_card_back_image)

    files = {
        "frontImage": ("front.jpg", io.BytesIO(front_buf.tobytes()), "image/jpeg"),
        "backImage": ("back.jpg", io.BytesIO(back_buf.tobytes()), "image/jpeg")
    }

    response = test_client.post("/api/v1/ekyc/card", files=files)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "extractedData" in data


def test_card_api_endpoint_single_sided_front_only(test_client, dummy_card_front_image):
    """
    Test uploading only front_image without back_image does NOT throw 422.
    """
    _, front_buf = cv2.imencode(".jpg", dummy_card_front_image)

    files = {
        "front_image": ("front.jpg", io.BytesIO(front_buf.tobytes()), "image/jpeg")
    }

    response = test_client.post("/api/v1/ekyc/card", files=files)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "extractedData" in data


def test_card_api_endpoint_base64_json(test_client, dummy_card_front_image):
    """
    Test posting JSON payload with Base64 image strings.
    """
    import base64
    _, front_buf = cv2.imencode(".jpg", dummy_card_front_image)
    b64_str = "data:image/jpeg;base64," + base64.b64encode(front_buf.tobytes()).decode("utf-8")

    payload = {
        "frontImage": b64_str
    }

    response = test_client.post("/api/v1/ekyc/card", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "extractedData" in data
