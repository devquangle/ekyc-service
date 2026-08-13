import re
import unicodedata
from datetime import datetime
from typing import Optional, Tuple
from utils.text_utils import remove_vietnamese_accents


def normalize_unicode(text: Optional[str]) -> Optional[str]:
    """
    Normalizes unicode text to NFC and strips leading/trailing whitespaces.
    Returns None if input is empty, None, or whitespace-only.
    """
    if not text:
        return None
    clean = str(text).strip()
    if not clean:
        return None
    return unicodedata.normalize("NFC", clean)


def normalize_gender(raw_gender: Optional[str]) -> Optional[str]:
    """
    Normalizes raw OCR/MRZ/QR gender input to canonical 'Nam' or 'Nữ'.
    NAM / M / MALE -> Nam
    NỮ / NU / F / FEMALE -> Nữ
    Returns None if unrecognized.
    """
    if not raw_gender:
        return None
    clean = remove_vietnamese_accents(str(raw_gender)).upper().strip()

    if clean in ["NAM", "M", "MALE"]:
        return "Nam"
    elif clean in ["NU", "F", "FEMALE"]:
        return "Nữ"
    return None


def parse_date(date_str: Optional[str]) -> Optional[str]:
    """
    Parses date strings into ISO format YYYY-MM-DD with strict datetime validation.
    Supports DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY, YYYY-MM-DD, YYMMDD, DDMMYYYY.
    Returns None for invalid calendar dates (e.g. 31/02/1973) or malformed strings.
    """
    if not date_str:
        return None
    clean = str(date_str).strip()
    if not clean:
        return None

    # 1. Match YYYY-MM-DD
    match_iso = re.match(r'^(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$', clean)
    if match_iso:
        try:
            dt = datetime.strptime(clean, "%Y-%m-%d")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # 2. Match DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    dmy_match = re.search(
        r'\b(0[1-9]|[12]\d|3[01])[\/.\-](0[1-9]|1[0-2])[\/.\-]((?:19|20)\d{2})\b',
        clean
    )
    if dmy_match:
        day, month, year = int(dmy_match.group(1)), int(dmy_match.group(2)), int(dmy_match.group(3))
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # 3. Match YYMMDD (6 digits from MRZ)
    if re.match(r'^\d{6}$', clean):
        yy, mm, dd = int(clean[0:2]), int(clean[2:4]), int(clean[4:6])
        year = 2000 + yy if yy <= 50 else 1900 + yy
        try:
            dt = datetime(year, mm, dd)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    # 4. Match DDMMYYYY (8 digits)
    if re.match(r'^\d{8}$', clean):
        day, month, year = int(clean[0:2]), int(clean[2:4]), int(clean[4:8])
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def normalize_identity_number(raw_identity_number: Optional[str]) -> Optional[str]:
    """
    Normalizes raw identity number string by removing whitespace, fixing common OCR confusions
    (O->0, I->1, L->1), stripping non-digits, and validating length (9 digits for CMND or 12 for CCCD).
    Returns None if length is invalid.
    """
    if not raw_identity_number:
        return None
    clean = str(raw_identity_number).strip().upper()
    if not clean:
        return None

    # Remove whitespace
    clean = re.sub(r'\s+', '', clean)

    # OCR character replacements: O->0, I->1, L->1
    clean = clean.replace('O', '0').replace('I', '1').replace('L', '1')

    # Keep digits only
    digits_only = re.sub(r'[^\d]', '', clean)

    # Valid lengths: 9 digits (CMND) or 12 digits (CCCD)
    if len(digits_only) in (9, 12):
        return digits_only

    return None


def normalize_full_name(raw_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalizes full name returning a tuple of (canonical_normalized_name, raw_clean_text).
    Example: 'TRẦN  THỊ   ÚT' -> ('TRAN THI UT', 'TRẦN THỊ ÚT')
    """
    if not raw_name:
        return None, None

    text = str(raw_name).strip()
    if not text:
        return None, None

    text_nfc = unicodedata.normalize("NFC", text)
    raw_clean = re.sub(r'\s+', ' ', text_nfc).strip()
    if not raw_clean:
        return None, None

    canonical = remove_vietnamese_accents(raw_clean).upper()
    return canonical, raw_clean


def normalize_address(raw_address: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalizes administrative address preserving original word boundaries, accents, and casing.
    Returns a tuple of (normalized_value, clean_raw_text).
    Does NOT alter administrative names or remove accents.
    """
    if not raw_address:
        return None, None

    text = str(raw_address).strip()
    if not text:
        return None, None

    # Separate camelCase or concatenated words if present
    clean_raw = re.sub(
        r'([a-zàáảãạăắằẳẵặâấầnẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ])([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ])',
        r'\1 \2',
        text
    )
    clean_raw = re.sub(r'[\n\r]+', ' ', clean_raw)
    clean_raw = re.sub(r'\s+', ' ', clean_raw).strip()

    if not clean_raw:
        return None, None

    normalized_val = unicodedata.normalize("NFC", clean_raw)

    return normalized_val, normalized_val


def normalize_address_for_compare(raw_address: Optional[str]) -> Optional[str]:
    """
    Generates an unaccented, uppercase, clean string from an address for comparison purposes.
    Example: '123 Đường Nguyễn Trãi, P.5, Q.1' -> '123 DUONG NGUYEN TRAI P5 Q1'
    """
    if not raw_address:
        return None
    norm_val, _ = normalize_address(raw_address)
    if not norm_val:
        return None

    unaccented = remove_vietnamese_accents(norm_val)
    clean = re.sub(r'[^A-Z0-9\s]', ' ', unaccented)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean if clean else None


def normalize_text_for_compare(value: Optional[str]) -> Optional[str]:
    """
    Generates a standardized comparison key (unaccented, uppercase, collapsed whitespace,
    stripped punctuation) for matching OCR ↔ QR, OCR ↔ MRZ, or OCR ↔ DB.
    """
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None

    unaccented = remove_vietnamese_accents(text)
    clean = re.sub(r'[^A-Z0-9\s]', ' ', unaccented)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean if clean else None
