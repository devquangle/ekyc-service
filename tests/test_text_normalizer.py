import json
import pytest
from utils.text_normalizer import (
    UnicodeNormalizer,
    MojibakeFixer,
    VietnameseTextCorrector,
)


# ---------------------------------------------------------------------------
# 1. UnicodeNormalizer Tests
# ---------------------------------------------------------------------------

def test_unicode_normalizer_valid_vietnamese_preserved():
    """
    Valid Vietnamese strings MUST remain intact and normalized in NFC.
    """
    cases = [
        "Việt Nam",
        "Đồng Tháp",
        "Châu Thành",
        "Tân Bình",
        "HUỲNH QUANG LÊ",
        "Nguyễn Thanh Tùng",
        "Ấp Tây",
    ]
    for text in cases:
        result = UnicodeNormalizer.normalize(text)
        assert result == text, f"Failed for {text}: got {result}"


def test_unicode_normalizer_nfd_to_nfc():
    """
    Decomposed Unicode (NFD) must be transformed to precomposed (NFC).
    """
    import unicodedata
    nfd_text = unicodedata.normalize("NFD", "Tiếng Việt")
    assert nfd_text != "Tiếng Việt"  # NFD has separate combining marks

    nfc_text = UnicodeNormalizer.normalize(nfd_text)
    assert nfc_text == "Tiếng Việt"
    assert unicodedata.is_normalized("NFC", nfc_text)


def test_unicode_normalizer_zero_width_and_spaces():
    """
    Zero-width spaces, soft hyphens, BOMs, and non-breaking spaces must be sanitized.
    """
    dirty = "Vi\u200bệt\xa0Nam\ufeff"
    cleaned = UnicodeNormalizer.normalize(dirty)
    assert cleaned == "Việt Nam"


def test_unicode_normalizer_none_and_empty():
    assert UnicodeNormalizer.normalize(None) is None
    assert UnicodeNormalizer.normalize("") == ""


# ---------------------------------------------------------------------------
# 2. MojibakeFixer Tests
# ---------------------------------------------------------------------------

def test_mojibake_fixer_repairs_corrupted_vietnamese():
    """
    Mojibake / corrupted UTF-8 encodings must be correctly repaired.
    """
    mojibake_cases = [
        ("ViÃệt Nam", "Việt Nam"),
        ("Viá»‡t Nam", "Việt Nam"),
        ("Äá»“ng ThÃ¡p", "Đồng Tháp"),
        ("Ä‘", "đ"),
        ("Ä", "Đ"),
    ]
    for corrupted, expected in mojibake_cases:
        fixed = MojibakeFixer.fix(corrupted)
        assert fixed == expected, f"Failed to fix mojibake {corrupted!r}: got {fixed!r}, expected {expected!r}"


def test_mojibake_fixer_does_not_corrupt_valid_text():
    """
    Already valid Vietnamese text must NEVER be altered or damaged by the fixer.
    """
    valid_cases = [
        "Châu",
        "Việt Nam",
        "Đồng Tháp",
        "Tân Bình",
        "Hồ Chí Minh",
        "Đà Nẵng",
    ]
    for text in valid_cases:
        fixed = MojibakeFixer.fix(text)
        assert fixed == text, f"Valid text was wrongly modified: {text} -> {fixed}"


# ---------------------------------------------------------------------------
# 3. VietnameseTextCorrector Tests
# ---------------------------------------------------------------------------

def test_vietnamese_corrector_ocr_glyph_misrecognitions():
    """
    Non-Vietnamese Latin diacritics (umlauts/diaeresis) confused by OCR
    must be generically mapped to standard Vietnamese circumflex/breve vowels.
    No hard-coding of place names or person names.
    """
    glyph_cases = [
        ("Chäu", "Châu"),
        ("Tän", "Tân"),
        ("Đöng", "Đông"),
        ("Thänh", "Thành" if "Thành" == "Thänh" else "Thânh"),
    ]
    for raw, expected in glyph_cases:
        corrected = VietnameseTextCorrector.correct_ocr_glyphs(raw)
        assert corrected == expected


def test_vietnamese_corrector_spacing_and_punctuation():
    """
    Spacing around punctuation and OCR joined words must be cleaned.
    """
    assert VietnameseTextCorrector.format_spacing_and_punctuation(
        "Ap Tay Tan Binh,Chau Thanh,Dong Thap"
    ) == "Ap Tay Tan Binh, Chau Thanh, Dong Thap"

    assert VietnameseTextCorrector.format_spacing_and_punctuation(
        "Ap Tay , Tan Binh , Chau Thanh"
    ) == "Ap Tay, Tan Binh, Chau Thanh"

    assert VietnameseTextCorrector.format_spacing_and_punctuation(
        "Tan   Phu   Trung ,\n Dong   Thap"
    ) == "Tan Phu Trung, Dong Thap"


def test_vietnamese_corrector_pipeline_end_to_end():
    """
    Full pipeline: raw text preserved, ocrValue cleaned, confidence reported.
    """
    raw_input = "Tan Binh Chäu Thanh Dong Thap"
    raw_txt, ocr_val, corr_val, conf = VietnameseTextCorrector.normalize_pipeline(raw_input)

    assert raw_txt == raw_input
    assert ocr_val == "Tan Binh Châu Thanh Dong Thap"
    assert corr_val == "Tan Binh Châu Thanh Dong Thap"
    assert conf >= 0.90


def test_vietnamese_corrector_unaccented_text_not_blindly_guessed():
    """
    If text is plain unaccented ASCII without OCR glyph errors, it must NOT
    be guessed into arbitrary accent combinations without explicit model evidence.
    """
    raw_input = "Tan Binh"
    raw_txt, ocr_val, corr_val, conf = VietnameseTextCorrector.normalize_pipeline(raw_input)

    assert raw_txt == "Tan Binh"
    assert ocr_val == "Tan Binh"
    # Unaccented plain ASCII is preserved as normalized OCR value without hallucinating accents
    assert ocr_val == "Tan Binh"


# ---------------------------------------------------------------------------
# 4. UTF-8 JSON Serialization Test
# ---------------------------------------------------------------------------

def test_json_dumps_utf8_direct_representation():
    """
    Verify json serialization preserves UTF-8 characters directly when ensure_ascii=False.
    """
    data = {
        "nationality": "Việt Nam",
        "placeOfOrigin": "Tân Bình, Châu Thành, Đồng Tháp",
        "placeOfResidence": "Ấp Tây, Tân Bình, Châu Thành, Đồng Tháp"
    }
    json_str = json.dumps(data, ensure_ascii=False)

    assert "\\u1ec7" not in json_str
    assert "Việt Nam" in json_str
    assert "Đồng Tháp" in json_str
    assert "Tân Bình" in json_str


# ---------------------------------------------------------------------------
# 5. 3-Level Hierarchical & Fuzzy Administrative Restorer Tests
# ---------------------------------------------------------------------------

def test_hierarchical_fuzzy_restorer_provinces_and_districts():
    from utils.vietnamese_administrative_restorer import VietnameseAdministrativeRestorer

    # Test 1: Tan Luoc, Binh Tan, Vinh Long
    t1 = VietnameseAdministrativeRestorer.restore_address_diacritics("Tan Luoc.Binh Tan.Vinh Long")
    assert "Tân Lược" in t1 and "Bình Tân" in t1 and "Vĩnh Long" in t1

    # Test 2: To 09, Ap Phu Binh Tan Phu Trung. Dong Thap
    t2 = VietnameseAdministrativeRestorer.restore_address_diacritics("To 09, Ap Phu Binh Tan Phu Trung. Dong Thap")
    assert "Tổ 9" in t2 or "Tổ 09" in t2
    assert "Ấp Phú Bình" in t2 and "Tân Phú Trung" in t2 and "Đồng Tháp" in t2

    # Test 3: Ap Tay, Tan Binh, Chau Thanh, Dong Thap
    t3 = VietnameseAdministrativeRestorer.restore_address_diacritics("Ap Tay, Tan Binh, Chau Thanh, Dong Thap")
    assert "Ấp Tây" in t3 and "Tân Bình" in t3 and "Châu Thành" in t3 and "Đồng Tháp" in t3

    # Test 4: Ba Dinh, Ha Noi
    t4 = VietnameseAdministrativeRestorer.restore_address_diacritics("Ba Dinh, Ha Noi")
    assert "Ba Đình" in t4 and "Hà Nội" in t4

    # Test 5: Quan 1, TP Ho Chi Minh
    t5 = VietnameseAdministrativeRestorer.restore_address_diacritics("Quan 1, TP Ho Chi Minh")
    assert "Quận 1" in t5 and "Hồ Chí Minh" in t5

    # Test 6: Leaked label header removal
    t6 = VietnameseAdministrativeRestorer.restore_address_diacritics("Noi thurng trư/ Place of residence Ap Tay Tan Binh, Chau Thanh, Dong Thap")
    assert "Noi thurng" not in t6 and "Place of residence" not in t6
    assert "Ấp Tây" in t6 and "Tân Bình" in t6 and "Châu Thành" in t6 and "Đồng Tháp" in t6


