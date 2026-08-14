import pytest
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from ocr.mrz_parser import MrzParser


def test_qr_parser_structure():
    """
    Tests Vietnamese CCCD QR code payload parsing into standardized fields.
    """
    qr_engine = QrEngine()
    raw_qr = "087204000897|123456789|HUỲNH QUANG LÊ|04102004|Nam|Ấp Tây Tân Bình, Châu Thành, Đồng Tháp|30032021"
    parsed = qr_engine.parse_qr_string(raw_qr)

    assert parsed is not None
    assert parsed["identityNumber"] == "087204000897"
    assert parsed["fullName"] == "HUỲNH QUANG LÊ"
    assert parsed["dateOfBirth"] == "2004-10-04"
    assert parsed["gender"] == "Nam"
    assert parsed["dateOfIssue"] == "2021-03-30"


def test_mrz_parser_td1():
    """
    Tests 3-line MRZ TD1 ICAO 9303 parser.
    """
    mrz_engine = MrzEngine()
    lines = [
        "IDVNM2040008978087204000897<<9",
        "0410047M2910046VNM<<<<<<<<<<<6",
        "HUYNH<<QUANG<LE<<<<<<<<<<<<<<"
    ]
    parsed = mrz_engine.parse(lines)

    assert parsed is not None
    assert parsed["identityNumber"] == "087204000897"
    assert parsed["fullName"] == "HUYNH QUANG LE"
    assert parsed["dateOfBirth"] == "2004-10-04"
    assert parsed["gender"] == "Nam"
    assert parsed["dateOfExpiry"] == "2029-10-04"
    assert parsed["isMrzValid"] is True


def test_mrz_check_digit_calculation():
    """
    Tests ICAO Doc 9303 Modulo 10 check digit calculation with 7-3-1 weights.
    """
    assert MrzEngine.compute_check_digit("041004") == 7
    assert MrzEngine.compute_check_digit("291004") == 6
    # Test OCR confusion characters (O -> 0)
    assert MrzEngine.compute_check_digit("O41OO4") == 7


def test_mrz_clean_numeric_and_alpha():
    """
    Tests character sanitization for numeric and alphabetic MRZ zones.
    """
    assert MrzParser._clean_numeric_field("O41OO4") == "041004"
    assert MrzParser._clean_numeric_field("ISBZ") == "1582"
    assert MrzParser._clean_alpha_field("015") == "OIS"
