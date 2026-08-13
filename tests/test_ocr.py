import pytest
from ocr import (
    OCRText,
    LayoutParser,
    FieldExtractor,
    CardTypeClassifier,
    MrzParser,
    QrParser,
    parse_date,
    normalize_gender,
    normalize_identity_number,
    normalize_full_name,
    normalize_address,
    normalize_address_for_compare,
    normalize_text_for_compare,
)


def test_spatial_line_grouping():
    parser = LayoutParser()
    tokens = [
        OCRText(text="Họ", confidence=0.98, bbox=[[10, 100], [40, 100], [40, 120], [10, 120]]),
        OCRText(text="và", confidence=0.98, bbox=[[50, 100], [80, 100], [80, 120], [50, 120]]),
        OCRText(text="tên:", confidence=0.98, bbox=[[90, 102], [140, 102], [140, 122], [90, 122]]),
        OCRText(text="TRẦN", confidence=0.97, bbox=[[150, 100], [200, 100], [200, 120], [150, 120]]),
        OCRText(text="THỊ", confidence=0.97, bbox=[[210, 100], [250, 100], [250, 120], [210, 120]]),
        OCRText(text="ÚT", confidence=0.97, bbox=[[260, 100], [290, 100], [290, 120], [260, 120]]),
    ]
    lines = parser.group_tokens_into_lines(tokens)
    assert len(lines) == 1
    assert lines[0].text == "Họ và tên: TRẦN THỊ ÚT"


def test_field_extractor_full_name():
    extractor = FieldExtractor()
    tokens = [
        OCRText(text="Họ và tên / Full name:", confidence=0.98, bbox=[[100, 100], [300, 100], [300, 120], [100, 120]]),
        OCRText(text="TRẦN THỊ ÚT", confidence=0.97, bbox=[[310, 100], [450, 100], [450, 120], [310, 120]]),
    ]
    fields = extractor.extract_all_fields(tokens)
    assert "fullName" in fields
    assert fields["fullName"].value == "TRAN THI UT"
    assert fields["fullName"].rawText == "TRẦN THỊ ÚT"


def test_gender_normalization():
    assert normalize_gender("NAM") == "Nam"
    assert normalize_gender("NỮ") == "Nữ"
    assert normalize_gender("MALE") == "Nam"
    assert normalize_gender("FEMALE") == "Nữ"
    assert normalize_gender("M") == "Nam"
    assert normalize_gender("F") == "Nữ"
    assert normalize_gender(None) is None
    assert normalize_gender("") is None
    assert normalize_gender("   ") is None
    assert normalize_gender("UNKNOWN") is None


def test_date_parser():
    assert parse_date("01/01/1973") == "1973-01-01"
    assert parse_date("01-01-1973") == "1973-01-01"
    assert parse_date("01.01.1973") == "1973-01-01"
    assert parse_date("1973-01-01") == "1973-01-01"
    assert parse_date("730101") == "1973-01-01"
    
    # Invalid calendar dates MUST return None
    assert parse_date("31/02/1973") is None
    assert parse_date("99/99/1973") is None
    assert parse_date("1973-99-99") is None
    assert parse_date(None) is None
    assert parse_date("") is None
    assert parse_date("abc") is None


def test_normalize_identity_number():
    assert normalize_identity_number("086173011002") == "086173011002"
    assert normalize_identity_number("O86173O11OO2") == "086173011002"
    assert normalize_identity_number(" 086 173 011 002 ") == "086173011002"
    assert normalize_identity_number("123456789") == "123456789"
    assert normalize_identity_number("12345") is None
    assert normalize_identity_number("1234567890123") is None
    assert normalize_identity_number(None) is None
    assert normalize_identity_number("") is None


def test_normalize_full_name():
    canonical, raw_clean = normalize_full_name("TRẦN  THỊ   ÚT")
    assert canonical == "TRAN THI UT"
    assert raw_clean == "TRẦN THỊ ÚT"
    
    c_none, r_none = normalize_full_name(None)
    assert c_none is None and r_none is None


def test_normalize_address():
    norm, raw = normalize_address("123 Đường  Nguyễn Trãi,\n P.5, Q.1")
    assert norm == "123 Đường Nguyễn Trãi, P.5, Q.1"
    assert raw == "123 Đường Nguyễn Trãi, P.5, Q.1"

    cmp = normalize_address_for_compare("123 Đường Nguyễn Trãi, P.5, Q.1")
    assert cmp == "123 DUONG NGUYEN TRAI P 5 Q 1" or "123 DUONG NGUYEN TRAI P5 Q1" in cmp


def test_normalize_text_for_compare():
    assert normalize_text_for_compare(" TRẦN   THỊ ÚT! ") == "TRAN THI UT"
    assert normalize_text_for_compare(None) is None


def test_address_boundary_isolation():
    extractor = FieldExtractor()
    tokens = [
        OCRText(text="Quê quán / Place of origin:", confidence=0.98, bbox=[[100, 100], [300, 100], [300, 120], [100, 120]]),
        OCRText(text="Tân Lược, Bình Tân, Vĩnh Long", confidence=0.97, bbox=[[100, 130], [400, 130], [400, 150], [100, 150]]),
        OCRText(text="Có giá trị đến / Date of expiry:", confidence=0.98, bbox=[[100, 160], [350, 160], [350, 180], [100, 180]]),
        OCRText(text="01/01/2033", confidence=0.97, bbox=[[100, 190], [200, 190], [200, 210], [100, 210]]),
        OCRText(text="Nơi thường trú / Place of residence:", confidence=0.98, bbox=[[100, 220], [400, 220], [400, 240], [100, 240]]),
        OCRText(text="Tân Bình, Châu Thành, Đồng Tháp", confidence=0.97, bbox=[[100, 250], [450, 250], [450, 270], [100, 270]]),
    ]
    fields = extractor.extract_all_fields(tokens)

    assert "placeOfOrigin" in fields
    assert fields["placeOfOrigin"].value == "Tân Lược, Bình Tân, Vĩnh Long"
    assert "dateOfExpiry" in fields
    assert fields["dateOfExpiry"].value == "2033-01-01"
    assert "placeOfResidence" in fields
    assert fields["placeOfResidence"].value == "Tân Bình, Châu Thành, Đồng Tháp"


def test_mrz_parser_check_digits():
    parser = MrzParser()
    l1 = "I<VNM0861730110022086173011002<<"
    l2 = "7301017F3301016VNM<<<<<<<<<<<5"
    l3 = "TRAN<<THI<UT<<<<<<<<<<<<<<<<<<"

    result = parser.parse_mrz_lines([l1, l2, l3])
    assert result is not None
    assert result["fullName"] == "TRAN THI UT"
    assert result["dateOfBirth"] == "1973-01-01"
    assert result["dateOfExpiry"] == "2033-01-01"
    assert result["gender"] == "Nữ"


def test_card_type_classifier():
    classifier = CardTypeClassifier()
    front_tokens = [
        OCRText(text="CĂN CƯỚC CÔNG DÂN", confidence=0.98, bbox=[[100, 50], [300, 50], [300, 70], [100, 70]]),
        OCRText(text="Quê quán / Place of origin", confidence=0.98, bbox=[[100, 100], [300, 100], [300, 120], [100, 120]]),
    ]
    card_type, conf = classifier.classify(front_tokens, [], {})
    assert card_type == "CCCD_OLD"
    assert conf >= 0.95
