import os
import io
import cv2
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from main import app
from api.dependencies import get_orchestrator
from schemas.card import CardProcessResponse, ExtractedCardData, QualityChecks, VisualRegions
from schemas.face import FaceVerifyResponse
from schemas.liveness import LivenessResponse
from schemas.ekyc import FullEkycResponse
from schemas.enums import CardType, VerificationDecision, EkycOutcome, EkycExecutionStatus


def get_image_path(filename: str) -> Path:
    base_dir = Path(__file__).parent / "image"
    return base_dir / filename


@pytest.fixture(autouse=True)
def mock_client_orchestrator():
    """
    Mocks get_orchestrator for fast, reliable, schema-compliant client API testing on CI/CD.
    """
    mock_orch = MagicMock()
    mock_orch.process_card.return_value = CardProcessResponse(
        cardVerified=True,
        cardType=CardType.CCCD_OLD,
        cardTypeConfidence=0.98,
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
        qualityChecks=QualityChecks(isBlur=False, hasGlare=False, isCropped=False),
        visualRegions=VisualRegions(portrait=[10.0, 10.0, 100.0, 100.0]),
        fieldMetadata=[],
        errors=[]
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
        threshold=0.80,
        checksPassed=["BLINK"],
        errors=[]
    )
    mock_orch.process_full_ekyc.return_value = FullEkycResponse(
        requestId="client-test-req-123",
        status=EkycExecutionStatus.SUCCESS,
        ekycResult=EkycOutcome.EKYC_VERIFIED,
        executionTimeMs=35.0,
        cardResult=mock_orch.process_card.return_value,
        faceResult=mock_orch.verify_face.return_value,
        livenessResult=mock_orch.detect_liveness.return_value,
        failureReasons=[]
    )

    app.dependency_overrides[get_orchestrator] = lambda: mock_orch
    yield mock_orch
    app.dependency_overrides.clear()


@pytest.fixture
def client(mock_client_orchestrator):
    return TestClient(app)


def test_client_card_process_multipart(client):
    front_path = get_image_path("cccd_c_mt.jpg")
    back_path = get_image_path("cccd_c_ms.jpg")

    if not front_path.exists() or not back_path.exists():
        pytest.skip(f"Test images not found in {front_path.parent}")

    with open(front_path, "rb") as f_front, open(back_path, "rb") as f_back:
        files = {
            "front_image": ("front.jpg", f_front, "image/jpeg"),
            "back_image": ("back.jpg", f_back, "image/jpeg")
        }
        response = client.post("/api/v1/ekyc/card", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "cardVerified" in data
    assert "cardType" in data
    assert "extractedData" in data
    assert "visualRegions" in data or "visual_regions" in data


def test_client_face_verify_multipart(client):
    front_path = get_image_path("cccd_c_mt.jpg")
    if not front_path.exists():
        pytest.skip(f"Test image not found at {front_path}")

    with open(front_path, "rb") as f1, open(front_path, "rb") as f2:
        files = {
            "card_portrait": ("card.jpg", f1, "image/jpeg"),
            "selfie_image": ("selfie.jpg", f2, "image/jpeg")
        }
        response = client.post("/api/v1/ekyc/face/verify", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "faceVerified" in data
    assert "similarityScore" in data


def test_client_full_ekyc_multipart(client):
    front_path = get_image_path("cccd_c_mt.jpg")
    back_path = get_image_path("cccd_c_ms.jpg")

    if not front_path.exists() or not back_path.exists():
        pytest.skip(f"Test images not found in {front_path.parent}")

    with open(front_path, "rb") as f_front, open(back_path, "rb") as f_back, open(front_path, "rb") as f_selfie:
        files = {
            "front_image": ("front.jpg", f_front, "image/jpeg"),
            "back_image": ("back.jpg", f_back, "image/jpeg"),
            "selfie_image": ("selfie.jpg", f_selfie, "image/jpeg")
        }
        response = client.post("/api/v1/ekyc/verify", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "requestId" in data
    assert "status" in data
    assert "ekycResult" in data
