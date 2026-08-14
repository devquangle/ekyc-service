import pytest
from ocr import (
    OCRText,
    LayoutParser,
    FieldExtractor,
    CardTypeClassifier,
    MrzParser,
    LabelMatcher,
    normalize_gender,
    normalize_identity_number,
    normalize_full_name,
    parse_date,
)


def test_label_matcher_fuzzy_noise_handling():
    matcher = LabelMatcher()
    
    # Generic OCR noisy labels
    match1 = matcher.match_line_label("Ho vatenl full name:")
    assert match1 is not None
    assert match1[0] == "fullName"

    match2 = matcher.match_line_label("Dulc ofexpir")
    assert match2 is not None
    assert match2[0] == "dateOfExpiry"

    match3 = matcher.match_line_label("Quequanl / Place of origin")
    assert match3 is not None
    assert match3[0] == "placeOfOrigin"


def test_person_a_extraction():
    # Person A: NGUYỄN VĂN AN - 001199000111 - HÀ NỘI
    tokens = [
        OCRText(text="CĂN CƯỚC CÔNG DÂN", confidence=0.98, bbox=[[100, 20], [300, 20], [300, 40], [100, 40]]),
        OCRText(text="Số / No.: 001199000111", confidence=0.97, bbox=[[100, 60], [350, 60], [350, 80], [100, 80]]),
        OCRText(text="Họ và tên / Full name:", confidence=0.98, bbox=[[100, 90], [250, 90], [250, 110], [100, 110]]),
        OCRText(text="NGUYỄN VĂN AN", confidence=0.99, bbox=[[260, 90], [420, 90], [420, 110], [260, 110]]),
        OCRText(text="Ngày sinh / Date of birth:", confidence=0.98, bbox=[[100, 120], [280, 120], [280, 140], [100, 140]]),
        OCRText(text="15/05/1990", confidence=0.99, bbox=[[290, 120], [380, 120], [380, 140], [290, 140]]),
        OCRText(text="Giới tính / Sex: Nam", confidence=0.98, bbox=[[100, 150], [250, 150], [250, 170], [100, 170]]),
        OCRText(text="Quê quán / Place of origin:", confidence=0.98, bbox=[[100, 180], [300, 180], [300, 200], [100, 200]]),
        OCRText(text="Ba Đình, Hà Nội", confidence=0.97, bbox=[[100, 210], [250, 210], [250, 230], [100, 230]]),
    ]

    extractor = FieldExtractor()
    fields = extractor.extract_all_fields(tokens)

    assert fields["identityNumber"].value == "001199000111"
    assert fields["fullName"].value == "NGUYEN VAN AN"
    assert fields["fullName"].rawText == "NGUYỄN VĂN AN"
    assert fields["dateOfBirth"].value == "1990-05-15"
    assert fields["gender"].value == "Nam"
    assert fields["placeOfOrigin"].value == "Ba Đình, Hà Nội"


def test_person_b_extraction():
    # Person B: LÊ THỊ HOA - 079200123456 - TP HỒ CHÍ MINH
    tokens = [
        OCRText(text="Số định danh cá nhân / No: 079200123456", confidence=0.98, bbox=[[100, 50], [400, 50], [400, 70], [100, 70]]),
        OCRText(text="Họ, chữ đệm và tên khai sinh / Surname, given names:", confidence=0.98, bbox=[[100, 80], [450, 80], [450, 100], [100, 100]]),
        OCRText(text="LÊ THỊ HOA", confidence=0.99, bbox=[[100, 110], [250, 110], [250, 130], [100, 130]]),
        OCRText(text="Ngày, tháng, năm sinh / Date of birth: 20/10/2000", confidence=0.98, bbox=[[100, 140], [400, 140], [400, 160], [100, 160]]),
        OCRText(text="Giới tính / Sex: Nữ", confidence=0.98, bbox=[[100, 170], [250, 170], [250, 190], [100, 190]]),
        OCRText(text="Nơi cư trú / Place of residence:", confidence=0.98, bbox=[[100, 200], [350, 200], [350, 220], [100, 220]]),
        OCRText(text="Quận 1, TP Hồ Chí Minh", confidence=0.97, bbox=[[100, 230], [300, 230], [300, 250], [100, 250]]),
    ]

    extractor = FieldExtractor()
    fields = extractor.extract_all_fields(tokens)

    assert fields["identityNumber"].value == "079200123456"
    assert fields["fullName"].value == "LE THI HOA"
    assert fields["fullName"].rawText == "LÊ THỊ HOA"
    assert fields["dateOfBirth"].value == "2000-10-20"
    assert fields["gender"].value == "Nữ"
    assert "Quận 1" in fields["placeOfResidence"].value and "Hồ Chí Minh" in fields["placeOfResidence"].value


def test_person_c_cmnd_9digits():
    # Person C: PHẠM MINH ĐỨC - CMND 9 digits 123456789
    tokens = [
        OCRText(text="Số / No.: 123456789", confidence=0.96, bbox=[[100, 50], [300, 50], [300, 70], [100, 70]]),
        OCRText(text="Họ và tên: PHẠM MINH ĐỨC", confidence=0.98, bbox=[[100, 80], [350, 80], [350, 100], [100, 100]]),
    ]

    extractor = FieldExtractor()
    fields = extractor.extract_all_fields(tokens)

    assert fields["identityNumber"].value == "123456789"
    assert fields["fullName"].value == "PHAM MINH DUC"
    assert fields["fullName"].rawText == "PHẠM MINH ĐỨC"
