import pytest
from services.text_normalizer import UnicodeNormalizer, MojibakeFixer, VietnameseTextCorrector
from schemas.card import FieldMetadata


def test_field_metadata_raw_vs_ocr_value_separation():
    """
    FieldMetadata must maintain distinct rawText, ocrValue, correctedValue, and correctionConfidence.
    """
    raw_input = "Tan Binh Chäu Thanh Dong Thap"
    raw_txt, ocr_val, corr_val, conf = VietnameseTextCorrector.normalize_pipeline(raw_input)

    meta = FieldMetadata(
        field="placeOfOrigin",
        value=ocr_val,
        rawText=raw_txt,
        ocrValue=ocr_val,
        correctedValue=corr_val,
        correctionConfidence=conf
    )

    assert meta.rawText == "Tan Binh Chäu Thanh Dong Thap"
    assert meta.ocrValue == "Tan Binh Châu Thanh Dong Thap"
    assert meta.value == "Tan Binh Châu Thanh Dong Thap"
    assert meta.correctedValue == "Tan Binh Châu Thanh Dong Thap"
    assert meta.correctionConfidence >= 0.90


def test_all_user_requested_cases():
    """
    Comprehensive verification for all specific test cases listed in user request:
    - Châu
    - Chäu
    - Việt Nam
    - Viá»‡t Nam
    - Đồng Tháp
    - Äá»“ng ThÃ¡p
    - Tân Bình
    - Tan Binh
    """
    cases = [
        # (Input, Expected ocrValue)
        ("Châu", "Châu"),
        ("Chäu", "Châu"),
        ("Việt Nam", "Việt Nam"),
        ("Viá»‡t Nam", "Việt Nam"),
        ("Đồng Tháp", "Đồng Tháp"),
        ("Äá»“ng ThÃ¡p", "Đồng Tháp"),
        ("Tân Bình", "Tân Bình"),
        ("Tan Binh", "Tan Binh"),
    ]

    for inp, expected in cases:
        _, ocr_val, _, _ = VietnameseTextCorrector.normalize_pipeline(inp)
        assert ocr_val == expected, f"Failed case {inp!r} -> expected {expected!r}, got {ocr_val!r}"
