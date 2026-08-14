import os
import base64
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from main import app
from schemas.card import CardProcessResponse, ExtractedCardData, QualityChecks, VisualRegions


@pytest.fixture(scope="module")
def real_images_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "image")


@pytest.fixture(autouse=True)
def mock_orchestrator():
    """
    Sets up mock orchestrator in app.state to test endpoint input validation and parameter parsing instantly.
    """
    mock_orch = MagicMock()
    mock_orch.process_card.return_value = CardProcessResponse(
        success=True,
        cardType="CCCD_OLD",
        cardVerified=True,
        confidence=0.98,
        extractedData=ExtractedCardData(
            identityNumber="087204000897",
            fullName="HUYNH QUANG LE",
            dateOfBirth="2004-10-04",
            gender="Nam",
            nationality="Việt Nam",
            placeOfOrigin="Tân Bình, Châu Thành, Đồng Tháp",
            placeOfResidence="Ấp Tây, Tân Bình, Châu Thành, Đồng Tháp",
            dateOfIssue="2021-03-30",
            dateOfExpiry="2029-10-04"
        ),
        qualityChecks=QualityChecks(blurScore=150.0, isBlurry=False, glareDetected=False, darkDetected=False, passed=True),
        visualRegions=VisualRegions(portrait=[10.0, 10.0, 100.0, 100.0]),
        fieldMetadata=[]
    )
    app.state.orchestrator = mock_orch
    return mock_orch


def test_api_card_multipart_files(real_images_dir):
    """
    1. Test POST /api/v1/ekyc/card with standard multipart files.
    """
    front_path = os.path.join(real_images_dir, "cccd_c_mt.jpg")
    back_path = os.path.join(real_images_dir, "cccd_c_ms.jpg")

    if not os.path.exists(front_path):
        pytest.skip("Test images not present")

    client = TestClient(app)
    with open(front_path, "rb") as f_front, open(back_path, "rb") as f_back:
        files = {
            "front_image": ("cccd_c_mt.jpg", f_front, "image/jpeg"),
            "back_image": ("cccd_c_ms.jpg", f_back, "image/jpeg")
        }
        response = client.post("/api/v1/ekyc/card", files=files)

    assert response.status_code == 200
    assert response.json()["extractedData"]["identityNumber"] == "087204000897"


def test_api_card_string_input_avoids_422(real_images_dir):
    """
    2. Test POST /api/v1/ekyc/card when client sends form string/text instead of UploadFile
       (resolves the 'Value error, Expected UploadFile, received: <class 'str'>' 422 error).
    """
    front_path = os.path.join(real_images_dir, "cccd_c_mt.jpg")
    if not os.path.exists(front_path):
        pytest.skip("Test images not present")

    with open(front_path, "rb") as f:
        f_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

    client = TestClient(app)

    # Form text data
    r_form = client.post("/api/v1/ekyc/card", data={"front_image": f_b64})
    assert r_form.status_code != 422, f"Got 422 Unprocessable Entity on form data: {r_form.text}"
    assert r_form.status_code == 200

    # JSON payload
    r_json = client.post("/api/v1/ekyc/card", json={"front_image": f_b64})
    assert r_json.status_code != 422, f"Got 422 Unprocessable Entity on JSON: {r_json.text}"
    assert r_json.status_code == 200

    # Raw string payload
    r_raw_str = client.post("/api/v1/ekyc/card", data={"front_image": "r\r\nÁê\u001a£\u0017:É\u0019oH×\u0012<MÝàãl\u0012Ï\r\n"})
    assert r_raw_str.status_code != 422, "FastAPI should not reject raw string with 422"
