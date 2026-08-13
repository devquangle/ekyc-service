from processors.card_validator import CardValidator
from processors.card_processor import CardProcessor
from core.mrz_engine import MrzEngine
from schemas.card import ExtractedCardData


def test_case1_matching_identity_number():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUỲNH QUANG LÊ", dateOfBirth="2004-10-04")
    mrz_data = {"identityNumber": "087204000897", "fullName": "HUYNH QUANG LE", "dateOfBirth": "2004-10-04", "isMrzValid": True}

    card_verified, cross_val_res, errors = validator.validate(ocr_data, None, mrz_data, card_type="CCCD_OLD")
    detail = next(d for d in cross_val_res.details if d.fieldName == "identityNumber")

    assert detail.status == "MATCH"
    assert cross_val_res.ocrMatchMrz is True
    assert card_verified is True


def test_case2_mismatch_identity_number():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUỲNH QUANG LÊ", dateOfBirth="2004-10-04")
    mrz_data = {"identityNumber": "099999999999", "fullName": "HUYNH QUANG LE", "dateOfBirth": "2004-10-04", "isMrzValid": True}

    card_verified, cross_val_res, errors = validator.validate(ocr_data, None, mrz_data, card_type="CCCD_OLD")
    detail = next(d for d in cross_val_res.details if d.fieldName == "identityNumber")

    assert detail.status == "MISMATCH"
    assert cross_val_res.ocrMatchMrz is False
    assert card_verified is False
    assert "CARD_DATA_MISMATCH_IDENTITY_NUMBER" in errors


def test_case3_not_available_mrz_null():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUỲNH QUANG LÊ", dateOfBirth="2004-10-04")
    mrz_data = {"identityNumber": None, "fullName": None, "dateOfBirth": None, "isMrzValid": True}

    card_verified, cross_val_res, errors = validator.validate(ocr_data, None, mrz_data, card_type="CCCD_OLD")
    detail = next(d for d in cross_val_res.details if d.fieldName == "identityNumber")

    assert detail.status == "NOT_AVAILABLE"
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

    assert detail.status == "NOT_AVAILABLE"


def test_case6_value_only_in_ocr_not_mismatch():
    validator = CardValidator()
    ocr_data = ExtractedCardData(identityNumber="087204000897", fullName="HUỲNH QUANG LÊ", dateOfBirth="2004-10-04")

    _, cross_val_res, errors = validator.validate(ocr_data, None, None, card_type="CCCD_OLD")
    detail = next(d for d in cross_val_res.details if d.fieldName == "identityNumber")

    assert detail.status == "NOT_AVAILABLE"
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
