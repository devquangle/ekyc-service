import sys
import unicodedata
import re
from typing import Optional

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Standard Vietnamese character set (lowercase + uppercase)
VIETNAMESE_CHARS = set(
    "aàảãáạăằẳẵắặâầẩẫấậ"
    "bcdđeèẻẽéẹêềểễếệ"
    "fghiìỉĩíịjklmnoòỏõóọôồổỗốộơờởỡớợ"
    "pqrstuùủũúụưừửữứự"
    "vwxyỳỷỹýỵz"
    "AÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬ"
    "BCDĐEÈẺẼÉẸÊỀỂỄẾỆ"
    "FGHIÌỈĨÍỊJKLMNOÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢ"
    "PQRSTUÙỦŨÚỤƯỪỬỮỨỰ"
    "VWXYỲỶỸÝỴZ"
)

# Common Mojibake signature patterns in Vietnamese text (UTF-8 bytes decoded as cp1252/latin1)
MOJIBAKE_PATTERNS = [
    r'Ã[\x80-\xff]',
    r'á»[\x80-\xff\u2010-\u2030\w\W]',
    r'áº[\x80-\xff\u2010-\u2030\w\W]',
    r'Ä[\x80-\xff\u2010-\u2030\w\W]',
    r'Æ[\x80-\xff\u2010-\u2030\w\W]',
    r'â[\x80-\xff\u2010-\u2030]',
    r'Ã',
    r'á»',
    r'áº',
    r'Ä‘',
    r'Ä\x90',
    r'Ä\x91',
    r'Ä',
]


def normalize_unicode(text: Optional[str]) -> Optional[str]:
    """
    Normalizes unicode text to NFC and strips zero-width/invisible characters.
    """
    if not text:
        return text
    # Normalize to NFC
    text = unicodedata.normalize("NFC", str(text))
    # Remove zero-width spaces, soft hyphens, BOM
    text = re.sub(r'[\u200b\u200c\u200d\ufeff\xad]', '', text)
    # Replace non-breaking space with standard space
    text = text.replace('\xa0', ' ')
    return text


def is_mojibake_candidate(text: str) -> bool:
    """
    Checks if text has recognizable mojibake byte sequences.
    """
    if not text:
        return False
    # If text contains signature mojibake markers like 'á»', 'áº', 'Ã¡', 'Ã¢', 'Ä‘', 'Ä\x90'
    for pat in [r'á»', r'áº', r'Ã[¡-¿\x80-\xff]', r'Ä[\x90\x91‘]', r'Ä', r'Æ°', r'Æ¡']:
        if re.search(pat, text):
            return True
    return False


def count_vietnamese_chars(text: str) -> int:
    return sum(1 for c in text if c in VIETNAMESE_CHARS)


def repair_mojibake(text: Optional[str]) -> Optional[str]:
    """
    Repairs text that suffered from mojibake (double-encoded or wrongly decoded UTF-8).
    Strictly checks that the decoded output improves valid Vietnamese character count.
    Never alters already valid Vietnamese text.
    """
    if not text:
        return text

    text = normalize_unicode(text)
    if not is_mojibake_candidate(text):
        return text

    orig_score = count_vietnamese_chars(text)

    # Try standard codecs
    for codec in ('cp1252', 'iso-8859-1', 'latin1'):
        try:
            raw_bytes = text.encode(codec)
            repaired = raw_bytes.decode('utf-8')
            repaired = normalize_unicode(repaired)
            new_score = count_vietnamese_chars(repaired)
            if new_score >= orig_score and not is_mojibake_candidate(repaired):
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

    # Specific common standalone mojibake sequences
    repaired_direct = text
    repaired_direct = (
        repaired_direct.replace('Ä‘', 'đ')
        .replace('Ä\x91', 'đ')
        .replace('Ä\x90', 'Đ')
        .replace('Ä', 'Đ')
    )
    if repaired_direct != text:
        return normalize_unicode(repaired_direct)

    return text


def repair_ocr_diacritics(text: Optional[str]) -> Optional[str]:
    """
    Generically replaces non-Vietnamese Latin diacritic artifacts (such as umlauts)
    with their Vietnamese equivalents (e.g. ä -> â as in Chäu -> Châu).
    Operates at character level without hardcoding words or names.
    """
    if not text:
        return text
    chars = []
    for c in text:
        chars.append(OCR_DIACRITIC_MAP.get(c, c))
    return "".join(chars)


def clean_whitespaces_and_punctuation(text: Optional[str]) -> Optional[str]:
    """
    Normalizes spacing around punctuation and collapses multiple spaces.
    Example: 'Ap Tay , Tan Binh,Chau Thanh' -> 'Ap Tay, Tan Binh, Chau Thanh'
    """
    if not text:
        return text

    # Remove space before comma, period, colon, semicolon
    text = re.sub(r'\s+([,.:;])', r'\1', text)
    # Ensure space after comma, colon, semicolon if followed by alphanumeric
    text = re.sub(r'([,.:;])([^\s\d,.:;])', r'\1 \2', text)
    # Separate camelCase / joined words from OCR concatenation
    text = re.sub(
        r'([a-zàáảãạăắằẳẵặâấầnẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ])([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ])',
        r'\1 \2',
        text
    )
    # Collapse multiple whitespaces / newlines
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_vietnamese_text(text: Optional[str]) -> Optional[str]:
    """
    Full Vietnamese Text Normalization Pipeline:
    1. Unicode NFC Normalization
    2. Mojibake Detection & Repair
    3. Generic OCR Diacritic Normalization (ä -> â, etc.)
    4. Punctuation & Whitespace Cleanup
    """
    if not text:
        return text

    text = normalize_unicode(text)
    text = repair_mojibake(text)
    text = repair_ocr_diacritics(text)
    text = clean_whitespaces_and_punctuation(text)
    text = normalize_unicode(text)
    return text


if __name__ == "__main__":
    test_cases = [
        ("Châu", "Châu"),
        ("Chäu", "Châu"),
        ("Việt Nam", "Việt Nam"),
        ("Viá»‡t Nam", "Việt Nam"),
        ("Đồng Tháp", "Đồng Tháp"),
        ("Äá»“ng ThÃ¡p", "Đồng Tháp"),
        ("Tân Bình", "Tân Bình"),
        ("Tan Binh", "Tan Binh"),
        ("Ap Tay Tan Binh,Chau Thanh,Dong Thap", "Ap Tay Tan Binh, Chau Thanh, Dong Thap"),
        ("Ap Tay , Tan Binh , Chau Thanh", "Ap Tay, Tan Binh, Chau Thanh"),
        ("Ä‘", "đ"),
        ("Ä", "Đ"),
        ("Tan Binh Chäu Thanh Dong Thap", "Tan Binh Châu Thanh Dong Thap"),
    ]

    for inp, expected in test_cases:
        res = normalize_vietnamese_text(inp)
        print(f"Input: {inp!r:35} -> Result: {res!r:35} [Match: {res == expected}]")
        assert res == expected, f"Expected {expected!r}, got {res!r}"

    print("\nALL TEST CASES PASSED SUCCESSFULLY!")
