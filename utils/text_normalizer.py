import re
import unicodedata
from typing import Optional, Tuple
import ftfy


class UnicodeNormalizer:
    """
    Ensures standard Unicode NFC encoding and strips invisible, zero-width,
    BOM, and formatting control characters without altering valid Vietnamese characters.
    """

    @staticmethod
    def normalize(text: Optional[str]) -> Optional[str]:
        if not text:
            return text
        # Step 1: Standard NFC decomposition & recomposition
        norm = unicodedata.normalize("NFC", str(text))
        # Step 2: Remove zero-width spaces, soft hyphens, BOM characters
        norm = re.sub(r'[\u200b\u200c\u200d\ufeff\xad]', '', norm)
        # Step 3: Replace non-breaking space with regular space
        norm = norm.replace('\xa0', ' ')
        return norm


class MojibakeFixer:
    """
    Detects and repairs mojibake (corrupted UTF-8 encoded text that was decoded
    as Windows-1252, Latin-1, or cp1258).
    Uses ftfy with fallback heuristics.
    Never alters already valid Vietnamese text.
    """

    @staticmethod
    def fix(text: Optional[str]) -> Optional[str]:
        if not text:
            return text

        # Step 1: Pre-process specific standalone corrupted byte sequences
        fixed = (
            str(text)
            .replace('Ä‘', 'đ')
            .replace('Ä\x91', 'đ')
            .replace('\xc4\u2018', 'đ')
            .replace('Ä\x90', 'Đ')
        )

        # Step 2: Use ftfy.fix_text to reconstruct mangled UTF-8 bytes
        try:
            fixed = ftfy.fix_text(fixed)
        except Exception:
            pass

        # Step 3: Strip phantom mojibake prefix like 'Ã' directly preceding an already accented vowel (e.g. ViÃệt Nam -> Việt Nam)
        fixed = re.sub(
            r'Ã(?=[àáảãạăằẳẵắặâầẩẫấậeéèẻẽẹêềểễếệiíìỉĩịoóòỏõọôồổỗốộơờởỡợuúùủũụưừửữứựyỳỷỹỵÀÁẢÃẠĂẰẲẴẮẶÂẦẨẪẤẬEÉÈẺẼẸÊỀỂỄẾỆIÍÌỈĨỊOÓÒỎÕỌÔỒỔỖỐỘƠỜỞỠỚỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ])',
            '',
            fixed
        )

        # Step 4: Post-fix standalone characters
        fixed = (
            fixed.replace('Ä‘', 'đ')
            .replace('Ä\x91', 'đ')
            .replace('\xc4\u2018', 'đ')
            .replace('Ä\x90', 'Đ')
            .replace('Ä', 'Đ')
        )

        return UnicodeNormalizer.normalize(fixed)


class VietnameseTextCorrector:
    """
    Handles generic OCR recognition errors, non-Vietnamese Latin diacritics
    (such as German/French umlauts mistakenly recognized for Vietnamese circumflex),
    spacing around punctuation, and confidence evaluation.
    Does NOT hardcode specific province, district, commune, or person names.
    """

    # Generic OCR confusable diacritic glyph mapping:
    # PaddleOCR occasionally mistakes Vietnamese circumflex (^) / breve (˘) / horn (ơ/ư)
    # for Latin diaeresis / umlaut (¨) due to low contrast or font kerning.
    OCR_DIACRITIC_MAP = {
        'ä': 'â', 'Ä': 'Â',
        'ë': 'ê', 'Ë': 'Ê',
        'ï': 'i', 'Ï': 'I',
        'ö': 'ô', 'Ö': 'Ô',
        'ü': 'ư', 'Ü': 'Ư',
        'ÿ': 'y', 'Ÿ': 'Y',
    }

    @classmethod
    def correct_ocr_glyphs(cls, text: Optional[str]) -> str:
        """
        Generically maps non-Vietnamese Latin diacritics to Vietnamese equivalents.
        """
        if not text:
            return ""
        return "".join(cls.OCR_DIACRITIC_MAP.get(c, c) for c in text)

    @classmethod
    def format_spacing_and_punctuation(cls, text: Optional[str]) -> str:
        """
        Standardizes spacing around punctuation and word boundaries:
        - Removes spaces before commas/colons: 'Ap Tay , Tan Binh' -> 'Ap Tay, Tan Binh'
        - Ensures space after commas: 'Tan Binh,Chau Thanh' -> 'Tan Binh, Chau Thanh'
        - Separates joined camelCase words from OCR: 'PhuTrung' -> 'Phu Trung'
        - Collapses redundant whitespace and newlines.
        """
        if not text:
            return ""

        # Convert period between words into comma + space: 'Phu Trung. Dong Thap' -> 'Phu Trung, Dong Thap'
        cleaned = re.sub(r'([a-zA-Z\d\u00C0-\u1EF9])\.\s+([a-zA-Z\u00C0-\u1EF9])', r'\1, \2', text)
        cleaned = re.sub(r'([a-zA-Z\d\u00C0-\u1EF9])\.(?=[A-Z\u00C0-\u1EF9])', r'\1, ', cleaned)
        # Normalize unit abbreviation like T09 or T.09 or T9 before Ap / Thon to Tổ
        cleaned = re.sub(r'\bT0?(\d+)\b(?=\s*[,.]?\s*(?:Ap|Ấp|Thon|Thôn|Khom|Khóm|Khu|Xa|Xã|Phu|Binh))', r'Tổ \1', cleaned)
        # Remove spaces preceding commas, periods, colons, semicolons
        cleaned = re.sub(r'\s+([,.:;])', r'\1', cleaned)
        # Ensure single space after punctuation when followed immediately by alphanumeric
        cleaned = re.sub(r'([,.:;])([^\s\d,.:;])', r'\1 \2', cleaned)
        # Separate OCR-concatenated words where lowercase is immediately followed by uppercase
        cleaned = re.sub(
            r'([a-zàáảãạăắằẳẵặâấầnẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ])([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ])',
            r'\1 \2',
            cleaned
        )
        # Collapse newlines, tabs, and multiple spaces into single space
        cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    @classmethod
    def normalize_pipeline(
        cls,
        raw_text: Optional[str]
    ) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
        """
        Executes the full Vietnamese Text Normalization Pipeline:
        1. Unicode NFC Normalization
        2. Mojibake Detection and Repair (ftfy)
        3. Generic OCR Glyph Diacritic Correction
        4. Whitespace and Punctuation Formatting

        Returns:
            (raw_text, ocr_value, corrected_value, confidence)
        """
        if not raw_text:
            return None, None, None, 0.0

        raw_str = str(raw_text)

        # 1. Unicode NFC
        step1 = UnicodeNormalizer.normalize(raw_str)

        # 2. Mojibake Fix
        step2 = MojibakeFixer.fix(step1)

        # 3. OCR Glyph Correction
        step3 = cls.correct_ocr_glyphs(step2)

        # 4. Spacing & Punctuation
        ocr_value = cls.format_spacing_and_punctuation(step3)
        ocr_value = UnicodeNormalizer.normalize(ocr_value)

        was_modified = (ocr_value != raw_str.strip())
        confidence = 0.95 if was_modified else 1.0
        corrected_value = ocr_value if was_modified else None

        return raw_str, ocr_value, corrected_value, confidence
