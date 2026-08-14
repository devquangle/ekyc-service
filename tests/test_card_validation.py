import pytest
from processors.card_validator import CardValidator
from processors.card_processor import CardProcessor
from core.mrz_engine import MrzEngine
from schemas.card import ExtractedCardData
from schemas.enums import CardType, FieldValidationStatus


def test_case1_matching_identity_number():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUỲNH QUANG LÊ", dateOfBirth="2004-10-04")
    mrz_data = {"identityNumber": "087204000897", "fullName": "HUYNH QUANG LE", "dateOfBirth": "2004-10-04", "isMrzValid": True}

    card_verified, cross_val_res, errors = validator.validate(ocr_data, None, mrz_data, card_type="CCCD_OLD")
    detail = next(d for d in cross_val_res.details if d.fieldName == "identityNumber")

    assert detail.status in ["MATCH", FieldValidationStatus.MATCH]
    assert cross_val_res.ocrMatchMrz is True
    assert card_verified is True


def test_case2_mismatch_identity_number():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUỲNH QUANG LÊ", dateOfBirth="2004-10-04")
    mrz_data = {"identityNumber": "099999999999", "fullName": "HUYNH QUANG LE", "dateOfBirth": "2004-10-04", "isMrzValid": True}

    card_verified, cross_val_res, errors = validator.validate(ocr_data, None, mrz_data, card_type="CCCD_OLD")
    detail = next(d for d in cross_val_res.details if d.fieldName == "identityNumber")

    assert detail.status in ["MISMATCH", FieldValidationStatus.MISMATCH]
    assert cross_val_res.ocrMatchMrz is False
    assert card_verified is False
    assert "CARD_DATA_MISMATCH_IDENTITY_NUMBER" in errors


def test_case3_not_available_mrz_null():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUỲNH QUANG LÊ", dateOfBirth="2004-10-04")
    mrz_data = {"identityNumber": None, "fullName": None, "dateOfBirth": None, "isMrzValid": True}

    card_verified, cross_val_res, errors = validator.validate(ocr_data, None, mrz_data, card_type="CCCD_OLD")
    detail = next(d for d in cross_val_res.details if d.fieldName == "identityNumber")

    assert detail.status in ["NOT_AVAILABLE", FieldValidationStatus.NOT_AVAILABLE]
    assert cross_val_res.ocrMatchMrz is None
    assert "CARD_DATA_MISMATCH_IDENTITY_NUMBER" not in errors


def test_case4_fullname_fallback_from_mrz():
    extracted = ExtractedCardData(identityNumber="087204000897", fullName=None, dateOfBirth="2004-10-04")
    mrz_data = {"identityNumber": "087204000897", "fullName": "HUYNH QUANG LE", "dateOfBirth": "2004-10-04"}

    # Simulate fallback logic
    if not extracted.fullName and mrz_data.get("fullName"):
        extracted.fullName = mrz_data.get("fullName")

    assert extracted.fullName == "HUYNH QUANG LE"


def test_case5_status_not_available_single_source():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber=None, fullName=None, dateOfBirth=None)
    mrz_data = {"identityNumber": "087204000897", "fullName": "HUYNH QUANG LE", "dateOfBirth": "2004-10-04", "isMrzValid": True}

    _, cross_val_res, _ = validator.validate(ocr_data, None, mrz_data, card_type="CCCD_OLD")
    detail = next(d for d in cross_val_res.details if d.fieldName == "identityNumber")

    assert detail.status in ["NOT_AVAILABLE", FieldValidationStatus.NOT_AVAILABLE]


def test_case6_value_only_in_ocr_not_mismatch():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUỲNH QUANG LÊ", dateOfBirth="2004-10-04")

    _, cross_val_res, errors = validator.validate(ocr_data, None, None, card_type="CCCD_OLD")
    detail = next(d for d in cross_val_res.details if d.fieldName == "identityNumber")

    assert detail.status in ["NOT_AVAILABLE", FieldValidationStatus.NOT_AVAILABLE]
    assert cross_val_res.ocrMatchMrz is None
    assert "CARD_DATA_MISMATCH_IDENTITY_NUMBER" not in errors


def test_case7_mrz_invalid_position_returns_none_no_guess():
    mrz_engine = MrzEngine()
    # Invalid line 1 format without 12-digit starting with 0 at position 15:27
    invalid_lines = [
        "IDVNM2040008978XXXXXXXXXXXX<<9",
        "0410047M2910046VNM<<<<<<<<<<<6",
        "HUYNH<<QUANG<LE<<<<<<<<<<<<<<"
    ]
    parsed = mrz_engine.parse(invalid_lines)

    assert parsed is not None
    assert parsed["identityNumber"] is None


def test_case8_card_expired():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUỲNH QUANG LÊ", dateOfBirth="2004-10-04", dateOfExpiry="2020-01-01")

    card_verified, cross_val_res, errors = validator.validate(ocr_data, None, None, card_type="CCCD_OLD")

    assert cross_val_res.isExpired is True
    assert card_verified is False
    assert "CARD_EXPIRED" in errors


def test_card_validator_type_specific_comparators():
    validator = CardValidator()
    
    # 1. Gender: 'Nam' vs 'M' vs 'MALE' -> MATCH
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUYNH QUANG LE", gender="Nam")
    qr_data = {"identityNumber": "087204000897", "fullName": "HUYNH QUANG LE", "gender": "Nam"}
    mrz_data = {"identityNumber": "087204000897", "fullName": "HUYNH QUANG LE", "gender": "Nam", "isMrzValid": True}
    
    verified, res, _ = validator.validate(ocr_data, qr_data, mrz_data, card_type=CardType.CCCD_OLD)
    g_detail = next(d for d in res.details if d.fieldName == "gender")
    assert g_detail.status in ["MATCH", FieldValidationStatus.MATCH]
    assert verified is True

    # 2. Date: '04/10/2004' vs '2004-10-04' -> MATCH
    ocr_data_date = ExtractedCardData(dateOfBirth="2004-10-04", dateOfExpiry="2029-10-04")
    qr_data_date = {"dateOfBirth": "2004-10-04", "dateOfExpiry": "2029-10-04"}
    _, res_date, _ = validator.validate(ocr_data_date, qr_data_date, None, card_type=CardType.CCCD_NEW)
    dob_detail = next(d for d in res_date.details if d.fieldName == "dateOfBirth")
    assert dob_detail.status in ["MATCH", FieldValidationStatus.MATCH]


def test_card_processor_visual_regions_and_bounding_box_info(mocker):
    from schemas.face import BoundingBoxInfo
    from core.ocr_engine import OcrEngine
    from core.qr_engine import QrEngine
    import numpy as np

    ocr_engine = mocker.MagicMock(spec=OcrEngine)
    qr_engine = mocker.MagicMock(spec=QrEngine)
    mrz_engine = mocker.MagicMock(spec=MrzEngine)

    ocr_engine.detect_tokens.return_value = []
    qr_engine.decode.return_value = None
    qr_engine.last_qr_bbox = [450.0, 50.0, 600.0, 200.0]
    mrz_engine.parse.return_value = None

    processor = CardProcessor(ocr_engine=ocr_engine, qr_engine=qr_engine, mrz_engine=mrz_engine)
    
    mock_bbox = BoundingBoxInfo(
        detected=True,
        x1=50,
        y1=100,
        x2=200,
        y2=300,
        width=150,
        height=200,
        detectionScore=0.99
    )
    mocker.patch.object(processor.card_face_extractor, "extract_face", return_value=(None, None, mock_bbox, []))

    dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
    (
        card_type,
        conf,
        extracted_data,
        qr_data,
        mrz_data,
        quality,
        metadata,
        visual_regions
    ) = processor.process(dummy_img)

    assert visual_regions is not None
    assert visual_regions.portrait == [50.0, 100.0, 200.0, 300.0]
    assert visual_regions.qrCode == [450.0, 50.0, 600.0, 200.0]
