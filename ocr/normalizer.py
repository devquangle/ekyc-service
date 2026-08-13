import re
import unicodedata
from typing import Optional, Tuple
from utils.text_utils import remove_vietnamese_accents


def normalize_unicode(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    return unicodedata.normalize("NFC", text.strip())


def normalize_gender(raw_gender: Optional[str]) -> Optional[str]:
    """
    Normalizes raw OCR/MRZ/QR gender input to canonical 'Nam' or 'Nữ'.
    NAM / M / MALE -> Nam
    NỮ / NU / F / FEMALE -> Nữ
    """
    if not raw_gender:
        return None
    clean = remove_vietnamese_accents(raw_gender.strip()).upper()

    if clean in ["NAM", "M", "MALE"]:
        return "Nam"
    elif clean in ["NU", "F", "FEMALE", "NÜ"]:
        return "Nữ"
    return None


def parse_date(date_str: Optional[str]) -> Optional[str]:
    """
    Parses date strings into ISO format YYYY-MM-DD.
    Supports DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD, YYMMDD.
    """
    if not date_str:
        return None
    clean = date_str.strip()

    # Match YYYY-MM-DD
    if re.match(r'^(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$', clean):
        return clean

    # Match DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    dmy_match = re.search(r'\b(0[1-9]|[12]\d|3[01])[\/.\-](0[1-9]|1[0-2])[\/.\-]((?:19|20)\d{2})\b', clean)
    if dmy_match:
        day, month, year = dmy_match.group(1), dmy_match.group(2), dmy_match.group(3)
        return f"{year}-{month}-{day}"

    # Match YYMMDD (6 digits from MRZ)
    if re.match(r'^\d{6}$', clean):
        yy, mm, dd = clean[0:2], clean[2:4], clean[4:6]
        year = f"20{yy}" if int(yy) <= 50 else f"19{yy}"
        return f"{year}-{mm}-{dd}"

    return None


def normalize_full_name(raw_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (canonical_normalized_name, raw_text).
    Example: 'TRẦN  THỊ ÚT' -> ('TRAN THI UT', 'TRẦN THỊ ÚT')
    """
    if not raw_name:
        return None, None

    raw_clean = normalize_unicode(re.sub(r'\s+', ' ', raw_name).strip())
    canonical = remove_vietnamese_accents(raw_clean).upper()
    return canonical, raw_clean


def normalize_address(raw_address: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Preserves word boundaries and formats administrative address strings.
    Returns: (normalized_value, clean_raw_text)
    """
    if not raw_address:
        return None, None

    clean_raw = re.sub(
        r'([a-zàáảãạăắằẳẵặâấầnẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ])([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ])',
        r'\1 \2',
        raw_address
    )

    clean_raw = re.sub(r'[\n]+', ' ', clean_raw).strip()
    words = [w.strip() for w in clean_raw.split() if w.strip()]
    raw_joined = " ".join(words)

    normalized_val = normalize_unicode(raw_joined)
    clean_raw_val = normalize_unicode(clean_raw)
    return normalized_val, clean_raw_val
