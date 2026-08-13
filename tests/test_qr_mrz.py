from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine


def test_qr_parser_structure():
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
    # Test Modulo 10 algorithm
    assert MrzEngine.compute_check_digit("041004") == 7
    assert MrzEngine.compute_check_digit("291004") == 6
