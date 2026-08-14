import pytest
from schemas.enums import (
    CardType,
    VerificationDecision,
    CrossValidationStatus,
    EkycOutcome,
    EkycExecutionStatus,
    FieldValidationStatus,
)
from schemas.card import (
    ExtractedCardData,
    VisualRegions,
    FieldMetadata,
    CrossValidationDetail,
    CrossValidationResult,
    QualityChecks,
    CardProcessResponse,
)
from schemas.face import (
    BoundingBoxInfo,
    FaceQualityMetrics,
    FaceVerifyResponse,
)
from schemas.liveness import LivenessResponse
from schemas.ekyc import FullEkycResponse


def test_bounding_box_info_auto_calculations():
    # Test auto-calculation of width/height from x1, y1, x2, y2
    b1 = BoundingBoxInfo(detected=True, x1=50, y1=100, x2=250, y2=400)
    assert b1.width == 200
    assert b1.height == 300
    assert b1.bbox == [50, 100, 250, 400]

    # Test auto-calculation of x2/y2 from width and height
    b2 = BoundingBoxInfo(detected=True, x1=10, y1=20, width=100, height=150)
    assert b2.x2 == 110
    assert b2.y2 == 170
    assert b2.bbox == [10, 20, 110, 170]

    # Test auto-calculation from bbox list
    b3 = BoundingBoxInfo(detected=True, bbox=[30, 40, 130, 190])
    assert b3.x1 == 30
    assert b3.y1 == 40
    assert b3.x2 == 130
    assert b3.y2 == 190
    assert b3.width == 100
    assert b3.height == 150


def test_visual_regions_coordinate_validation():
    vr = VisualRegions(
        portrait=[200.0, 100.0, 50.0, 500.0],  # Flipped x coords: should be auto-ordered
        qrCode=[100, 200, 300, 400],
        mrzBlock=None
    )
    assert vr.portrait == [50.0, 100.0, 200.0, 500.0]
    assert vr.qrCode == [100.0, 200.0, 300.0, 400.0]
    assert vr.mrzBlock is None


def test_field_metadata_camel_and_snake_case_aliases():
    # Test input with snake_case
    meta1 = FieldMetadata(
        field="identityNumber",
        value="087204000897",
        label_box=[100, 50, 200, 80],
        value_box=[210, 50, 400, 80]
    )
    assert meta1.label_box == [100.0, 50.0, 200.0, 80.0]
    assert meta1.labelBox == [100.0, 50.0, 200.0, 80.0]
    assert meta1.value_box == [210.0, 50.0, 400.0, 80.0]
    assert meta1.valueBox == [210.0, 50.0, 400.0, 80.0]

    # Test input with camelCase
    meta2 = FieldMetadata(
        field="fullName",
        value="HUYNH QUANG LE",
        labelBox=[100, 90, 250, 110],
        valueBox=[100, 120, 350, 140],
        keyword="Họ và tên / Full name:"
    )
    assert meta2.label_box == [100.0, 90.0, 250.0, 110.0]
    assert meta2.labelBox == [100.0, 90.0, 250.0, 110.0]
    assert meta2.label == "Họ và tên / Full name:"


def test_card_process_response_visual_regions_aliases():
    resp = CardProcessResponse(
        cardVerified=True,
        cardType=CardType.CCCD_OLD,
        visualRegions=VisualRegions(portrait=[10, 20, 110, 120])
    )
    assert resp.visualRegions is not None
    assert resp.visual_regions is not None
    assert resp.visualRegions.portrait == [10.0, 20.0, 110.0, 120.0]
    assert resp.visual_regions.portrait == [10.0, 20.0, 110.0, 120.0]


def test_enums_compatibility():
    # CardType
    assert CardType.CCCD_OLD == "CCCD_OLD"
    assert CardType.CCCD_NEW == "CCCD_NEW"
    assert CardType.CMND_9 == "CMND_9"
    assert CardType.CMND_12 == "CMND_12"

    # VerificationDecision
    assert VerificationDecision.MATCH == "MATCH"
    assert VerificationDecision.MISMATCH == "MISMATCH"
    assert VerificationDecision.SUSPICIOUS == "SUSPICIOUS"

    # FieldValidationStatus
    assert FieldValidationStatus.MATCH == "MATCH"
    assert FieldValidationStatus.MISMATCH == "MISMATCH"
    assert FieldValidationStatus.NOT_AVAILABLE == "NOT_AVAILABLE"

    # EkycOutcome
    assert EkycOutcome.EKYC_VERIFIED == "EKYC_VERIFIED"
    assert EkycOutcome.EKYC_NOT_VERIFIED == "EKYC_NOT_VERIFIED"
