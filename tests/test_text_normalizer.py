import json
import pytest
from utils.text_normalizer import (
    UnicodeNormalizer,
    MojibakeFixer,
    VietnameseTextCorrector,
    normalize_vietnamese_text,
    OCR_DIACRITIC_MAP,
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
    Standard Windows-1252 / CP1258 / UTF-8 double-encoding mojibake patterns.
    """
    # "Việt Nam" double-encoded as Windows-1252 -> "Viá»t Nam"
    corrupted_1 = "Viá»\x87t Nam"
    fixed_1 = MojibakeFixer.fix(corrupted_1)
    assert "Việt" in fixed_1 or fixed_1 == "Việt Nam"

    # "Đồng Tháp" -> "Ä\x90á»\x93ng ThÃ¡p"
    corrupted_2 = "Ä\x90á»\x93ng ThÃ¡p"
    fixed_2 = MojibakeFixer.fix(corrupted_2)
    assert "Đồng Tháp" in fixed_2


def test_mojibake_fixer_does_not_corrupt_valid_text():
    """
    MojibakeFixer must NOT alter already-valid Vietnamese text.
    """
    valid_texts = [
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
        "Độc lập - Tự do - Hạnh phúc",
        "Ấp Tây Tân Bình, Châu Thành, Đồng Tháp",
        "Tân Phú Trung, Huyện Châu Thành, Tỉnh Đồng Tháp",
        "NGUYỄN VĂN AN",
        "Số 123 Đường Lê Lợi, Phường Bến Nghé, Quận 1, TP. Hồ Chí Minh",
    ]
    for text in valid_texts:
        assert MojibakeFixer.fix(text) == text


# ---------------------------------------------------------------------------
# 3. VietnameseTextCorrector Tests
# ---------------------------------------------------------------------------

def test_vietnamese_corrector_ocr_glyph_misrecognitions():
    """
    OCR character confusions (diacritics, OCR-specific symbols).
    """
    # 1. OCR diacritic mapping (ä -> â, ë -> ê, ö -> ô, ü -> ư)
    assert VietnameseTextCorrector.correct_vietnamese_ocr_glyphs("Tân Phú Trüng") == "Tân Phú Trưng"
    assert VietnameseTextCorrector.correct_vietnamese_ocr_glyphs("Bình Dưöng") == "Bình Dương"

    # 2. Case preservation
    assert VietnameseTextCorrector.correct_vietnamese_ocr_glyphs("TÂN PHÚ TRÜNG") == "TÂN PHÚ TRƯNG"


def test_vietnamese_corrector_spacing_and_punctuation():
    """
    Tests spacing around punctuation, commas, dots, and colons.
    """
    raw_addr = "Tân Phú Trung ,Châu Thành ,Đồng Tháp"
    cleaned = VietnameseTextCorrector.clean_spacing_and_punctuation(raw_addr)
    assert cleaned == "Tân Phú Trung, Châu Thành, Đồng Tháp"

    raw_dob = "04 / 10 / 2004"
    cleaned_dob = VietnameseTextCorrector.clean_spacing_and_punctuation(raw_dob)
    assert cleaned_dob == "04/10/2004"


def test_vietnamese_corrector_pipeline_end_to_end():
    """
    Full pipeline: normalize_vietnamese_text.
    """
    raw = "  Tân Phú Trüng , Châu Thành , Ä\x90á»\x93ng ThÃ¡p  "
    restored = normalize_vietnamese_text(raw)
    assert "Tân Phú" in restored
    assert "Châu Thành" in restored
    assert "Đồng Tháp" in restored


def test_vietnamese_corrector_unaccented_text_not_blindly_guessed():
    """
    A person's unaccented name (e.g. from MRZ: 'HUYNH QUANG LE') MUST NOT be blindly guessed.
    """
    name = "HUYNH QUANG LE"
    corrected = normalize_vietnamese_text(name)
    assert corrected == "HUYNH QUANG LE"


def test_json_dumps_utf8_direct_representation():
    """
    Ensure Vietnamese text serialized with ensure_ascii=False renders directly in UTF-8 without unicode escapes.
    """
    data = {
        "fullName": "HUỲNH QUANG LÊ",
        "placeOfOrigin": "Tân Bình, Châu Thành, Đồng Tháp",
        "placeOfResidence": "Ấp Tây Tân Bình, Châu Thành, Đồng Tháp"
    }
    json_str = json.dumps(data, ensure_ascii=False)
    assert "\\u" not in json_str
    assert "HUỲNH QUANG LÊ" in json_str
    assert "Đồng Tháp" in json_str


def test_hierarchical_fuzzy_restorer_provinces_and_districts():
    """
    Tests 3-level hierarchical address restoration for provinces, districts and communes.
    """
    from utils.vietnamese_administrative_restorer import VietnameseAdministrativeRestorer

    # Province restoration
    res1 = VietnameseAdministrativeRestorer.restore_address_diacritics("Dong Thap")
    assert "Đồng Tháp" in res1

    # District + Province
    res2 = VietnameseAdministrativeRestorer.restore_address_diacritics("Chau Thanh, Dong Thap")
    assert "Châu Thành" in res2
    assert "Đồng Tháp" in res2

    # Hamlet + Commune + District + Province
    res3 = VietnameseAdministrativeRestorer.restore_address_diacritics("Ap Tay Tan Binh, Chau Thanh, Dong Thap")
    assert "Ấp Tây" in res3
    assert "Tân Bình" in res3
    assert "Châu Thành" in res3
    assert "Đồng Tháp" in res3
