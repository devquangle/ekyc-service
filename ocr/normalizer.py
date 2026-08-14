import re
import unicodedata
from datetime import datetime
from typing import Optional, Tuple
from utils.text_utils import remove_vietnamese_accents
from utils.text_normalizer import UnicodeNormalizer, MojibakeFixer, VietnameseTextCorrector
from utils.vietnamese_administrative_restorer import VietnameseAdministrativeRestorer
from utils.logger import logger


def normalize_unicode(text: Optional[str]) -> Optional[str]:
    """
    Normalizes unicode text to NFC, fixes mojibake and strips leading/trailing whitespaces.
    Returns None if input is empty, None, or whitespace-only.
    """
    if not text:
        return None
    clean = UnicodeNormalizer.normalize(str(text))
    if not clean or not clean.strip():
        return None
    fixed = MojibakeFixer.fix(clean)
    return fixed.strip() if fixed else None


def normalize_gender(raw_gender: Optional[str]) -> Optional[str]:
    """
    Normalizes raw OCR/MRZ/QR gender input to canonical 'Nam' or 'Nữ'.
    NAM / M / MALE -> Nam
    NỮ / NU / F / FEMALE -> Nữ
    Returns None if unrecognized or if text is Nationality (Việt Nam).
    """
    if not raw_gender:
        return None
    norm = normalize_unicode(raw_gender)
    if not norm:
        return None
    clean = remove_vietnamese_accents(norm).upper().strip()
    if "VIET NAM" in clean or "VIETNAM" in clean:
        return None

    clean = re.sub(r'^(?:GIOI\s*TINH\s*[\/:]\s*SEX|GIOI\s*TINH|SEX)\s*[:\.\s]*', '', clean).strip()

    if clean in ["NAM", "M", "MALE"]:
        return "Nam"
    elif clean in ["NU", "F", "FEMALE"]:
        return "Nữ"
    elif clean == "M":
        return "Nam"
    elif clean == "F":
        return "Nữ"
    
    if re.search(r'\bNAM\b', clean) and not re.search(r'\bVIET\b', clean):
        return "Nam"
    if re.search(r'\bNU\b', clean):
        return "Nữ"
    return None


def parse_date(date_str: Optional[str], is_expiry: bool = False) -> Optional[str]:
    """
    Parses date strings into ISO format YYYY-MM-DD with strict datetime validation.
    Smartly disambiguates DDMMYYYY vs YYYYMMDD and handles 6-digit YYMMDD from MRZ.

    Args:
        date_str: Raw date string.
        is_expiry: Set to True when parsing expiry date to apply forward century disambiguation.

    Returns:
        ISO formatted string 'YYYY-MM-DD' or None if invalid.
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

    # 2. Match Delimited DD/MM/YYYY, D/M/YYYY, DD-MM-YYYY, DD.MM.YYYY
    dmy_matches = list(re.finditer(
        r'(?:\b|\D)([1-9]|0[1-9]|[12]\d|3[01])[\/.\-]([1-9]|0[1-9]|1[0-2])[\/.\-]((?:19|20)\d{2})\b',
        clean
    ))
    if dmy_matches:
        candidate_dates = []
        for m in dmy_matches:
            day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                dt = datetime(year, month, day)
                candidate_dates.append(dt)
            except ValueError:
                continue
        if candidate_dates:
            selected_dt = max(candidate_dates) if is_expiry else candidate_dates[0]
            return selected_dt.strftime("%Y-%m-%d")


    # 3. Match 8-digit numeric strings: Disambiguate DDMMYYYY vs YYYYMMDD
    if re.match(r'^\d{8}$', clean):
        # A. Try DDMMYYYY first (Standard for Vietnamese QR / Forms: e.g. 01011973, 24122003)
        d_day, d_month, d_year = int(clean[0:2]), int(clean[2:4]), int(clean[4:8])
        # B. Try YYYYMMDD (e.g. 19730101, 20260223)
        y_year, y_month, y_day = int(clean[0:4]), int(clean[4:6]), int(clean[6:8])

        is_valid_dmy = False
        is_valid_ymd = False
        dt_dmy = None
        dt_ymd = None

        if 1900 <= d_year <= 2100 and 1 <= d_month <= 12 and 1 <= d_day <= 31:
            try:
                dt_dmy = datetime(d_year, d_month, d_day)
                is_valid_dmy = True
            except ValueError:
                pass

        if 1900 <= y_year <= 2100 and 1 <= y_month <= 12 and 1 <= y_day <= 31:
            try:
                dt_ymd = datetime(y_year, y_month, y_day)
                is_valid_ymd = True
            except ValueError:
                pass

        # Disambiguation logic
        if is_valid_dmy and not is_valid_ymd:
            return dt_dmy.strftime("%Y-%m-%d")
        elif is_valid_ymd and not is_valid_dmy:
            return dt_ymd.strftime("%Y-%m-%d")
        elif is_valid_dmy and is_valid_ymd:
            # Both are structurally valid dates.
            # If clean[0:4] looks like a reasonable year in 1900-2099 and clean[4:8] does not, prefer YMD.
            # Otherwise, standard Vietnamese QR data format is DDMMYYYY.
            if 1920 <= y_year <= 2030 and (d_year < 1920 or d_year > 2030):
                return dt_ymd.strftime("%Y-%m-%d")
            return dt_dmy.strftime("%Y-%m-%d")

    # 4. Match 6-digit YYMMDD (from MRZ standard)
    if re.match(r'^\d{6}$', clean):
        yy, mm, dd = int(clean[0:2]), int(clean[2:4]), int(clean[4:6])
        current_year_2d = datetime.now().year % 100

        if is_expiry:
            # Expiry dates are always in the current 21st century
            year = 2000 + yy
        else:
            # Date of Birth: if 2-digit YY > current 2-digit year -> 1900s, else 2000s
            year = 2000 + yy if yy <= current_year_2d else 1900 + yy

        try:
            dt = datetime(year, mm, dd)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def normalize_identity_number(raw_identity_number: Optional[str]) -> Optional[str]:
    """
    Normalizes raw identity number string:
    - Finds 12-digit CCCD or 9-digit CMND patterns.
    - Fixes common OCR confusions (O->0, I->1, L->1).
    - Enforces >= 65% digit density to prevent false positives on sentence lines.
    """
    if not raw_identity_number:
        return None
    raw_str = str(raw_identity_number).strip()
    if not raw_str:
        return None

    # 1. Direct regex match for clean 12 digits (CCCD) or 9 digits (CMND)
    m12 = re.search(r'\b([0-9OIL]{12})\b', raw_str, re.IGNORECASE)
    if m12:
        candidate = m12.group(1).upper().replace('O', '0').replace('I', '1').replace('L', '1')
        if len(candidate) == 12 and candidate.isdigit():
            return candidate

    m9 = re.search(r'\b([0-9OIL]{9})\b', raw_str, re.IGNORECASE)
    if m9:
        candidate = m9.group(1).upper().replace('O', '0').replace('I', '1').replace('L', '1')
        if len(candidate) == 9 and candidate.isdigit():
            return candidate

    # 2. Strip label prefixes like 'Số / No.:', 'Số:', 'No:'
    clean_no_header = re.sub(
        r'^(?:s[o\u1ed1]\s*[\/:]\s*no\.?|s[o\u1ed1]\s*[:\.\s]|no\.?[:\s]|s[o\u1ed1]\s*dinh\s*danh\s*ca\s*nhan[\s\/:]*)\s*',
        '', raw_str, flags=re.IGNORECASE
    ).strip()

    clean = clean_no_header.upper().replace('O', '0').replace('I', '1').replace('L', '1')
    digits_only = re.sub(r'[^\d]', '', clean)

    if len(digits_only) in (9, 12):
        non_ws_len = len(re.sub(r'\s+', '', clean_no_header))
        if non_ws_len > 0 and (len(digits_only) / non_ws_len) >= 0.65:
            return digits_only

    return None


def normalize_full_name(raw_name: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalizes full name returning a tuple of (canonical_normalized_name, raw_clean_text).
    Applies Unicode NFC normalization, Mojibake repair, and space cleanup.
    Example: 'TRẦN  THỊ   ÚT' -> ('TRAN THI UT', 'TRẦN THỊ ÚT')
    """
    if not raw_name:
        return None, None

    norm = normalize_unicode(raw_name)
    if not norm:
        return None, None

    clean = re.sub(r'^(?:h[o\u1ecd]\s*v[a\u00e0]\s*t[e\u00ean]|full\s*name|h[o\u1ecd]\s*t[e\u00ean])\s*[\/:]*\s*', '', norm, flags=re.IGNORECASE).strip()
    clean = re.sub(r'[^a-zA-Z\u00C0-\u1EF9\s]', '', clean)
    clean = re.sub(r'\s+', ' ', clean).strip().upper()

    if not clean or len(clean) < 2:
        return None, None

    canonical = remove_vietnamese_accents(clean).upper()
    return canonical, clean


def normalize_address(raw_address: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Normalizes address using hierarchical 3-level Vietnamese administrative gazetteer.
    Returns: (restored_accented_address, canonical_unaccented_address)
    """
    if not raw_address:
        return None, None

    norm = normalize_unicode(raw_address)
    if not norm:
        return None, None

    restored = VietnameseAdministrativeRestorer.restore_address_diacritics(norm)
    if not restored:
        return None, None

    canonical = remove_vietnamese_accents(restored).strip()
    return restored, canonical


def normalize_address_for_compare(address: Optional[str]) -> str:
    """
    Normalizes address for cross-validation comparison:
    Strips special characters, converts to uppercase alphanumeric string with space separation.
    """
    if not address:
        return ""
    norm = normalize_unicode(address)
    if not norm:
        return ""
    clean = remove_vietnamese_accents(norm).upper()
    clean = re.sub(r'[^A-Z0-9\s]', ' ', clean)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def normalize_text_for_compare(text: Optional[str]) -> Optional[str]:
    """
    Normalizes arbitrary text for cross-validation equality checks.
    Strips accents, converts to uppercase, removes special characters and excess spaces.
    Returns None if input is None or empty.
    """
    if not text:
        return None
    norm = normalize_unicode(str(text))
    if not norm:
        return None
    unaccented = remove_vietnamese_accents(norm).upper()
    clean = re.sub(r'[^A-Z0-9\s]', '', unaccented)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean if clean else None

