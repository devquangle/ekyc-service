import cv2
import numpy as np
from core.ocr_engine import OcrLine
from core.mrz_engine import MrzEngine
from core.qr_engine import QrEngine
from processors.card_processor import CardProcessor, normalize_unicode
from schemas.card import ExtractedCardData, FieldMetadata


def test_1_date_of_birth_raw_text_independent():
    processor = CardProcessor(None, None, None)
    lines = [
        OcrLine(text="Date of birth:", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="04/10/2004", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]])
    ]
    keywords = processor.detect_all_keywords(lines)
    val, raw_text, kw = processor._extract_date_with_keyword_priority(lines, "dateOfBirth", keywords)
    assert val == "2004-10-04"
    assert raw_text == "04/10/2004"
    assert raw_text != "2004-10-20"


def test_2_date_of_expiry_raw_text_independent():
    processor = CardProcessor(None, None, None)
    lines = [
        OcrLine(text="Có giá trị đến:", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="04/10/2029", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]])
    ]
    keywords = processor.detect_all_keywords(lines)
    val, raw_text, kw = processor._extract_date_with_keyword_priority(lines, "dateOfExpiry", keywords)
    assert val == "2029-10-04"
    assert raw_text == "04/10/2029"
    assert raw_text != "2004-10-20"


def test_3_place_of_origin_extracted_not_null():
    processor = CardProcessor(None, None, None)
    lines = [
        OcrLine(text="Quê quán:", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="Tân Bình, Châu Thành, Đồng Tháp", confidence=0.98, boundingBox=[[100, 140], [600, 140], [600, 170], [100, 170]])
    ]
    keywords = processor.detect_all_keywords(lines)
    val, raw_text, kw = processor.extract_field_value_by_boundary(lines, "placeOfOrigin", keywords)
    assert val == "Tân Bình, Châu Thành, Đồng Tháp"
    assert val is not None


def test_4_place_of_residence_extracted_full_address():
    processor = CardProcessor(None, None, None)
    lines = [
        OcrLine(text="Nơi thường trú:", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="Ấp Tây", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]]),
        OcrLine(text="Tân Bình, Châu Thành, Đồng Tháp", confidence=0.98, boundingBox=[[100, 180], [600, 180], [600, 210], [100, 210]])
    ]
    keywords = processor.detect_all_keywords(lines)
    val, raw_text, kw = processor.extract_field_value_by_boundary(lines, "placeOfResidence", keywords)
    assert val == "Ấp Tây, Tân Bình, Châu Thành, Đồng Tháp"
    assert val is not None


def test_5_date_of_issue_extracted_not_null():
    processor = CardProcessor(None, None, None)
    lines = [
        OcrLine(text="Ngày, tháng, năm: 30/03/2021", confidence=0.98, boundingBox=[[100, 100], [500, 100], [500, 130], [100, 130]])
    ]
    keywords = processor.detect_all_keywords(lines)
    val, raw_text, kw = processor._extract_date_with_keyword_priority(lines, "dateOfIssue", keywords)
    assert val == "2021-03-30"
    assert raw_text == "30/03/2021"


def test_6_response_schema_never_contains_place_of_birth():
    data = ExtractedCardData()
    dict_repr = data.model_dump()
    assert "placeOfBirth" not in dict_repr
    assert "identityNumber" in dict_repr
    assert "placeOfOrigin" in dict_repr
    assert "placeOfResidence" in dict_repr
    assert len(dict_repr) == 9


def test_7_metadata_independent_state():
    meta1 = FieldMetadata(field="dateOfBirth", value="2004-10-04", rawText="04/10/2004")
    meta2 = FieldMetadata(field="dateOfExpiry", value="2029-10-04", rawText="04/10/2029")
    assert meta1.rawText != meta2.rawText
    assert meta1.value != meta2.value
