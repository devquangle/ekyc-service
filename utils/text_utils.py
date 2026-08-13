import re
import unicodedata
from typing import Optional

try:
    from unidecode import unidecode
except ImportError:
    def unidecode(text: str) -> str:
        nfd_text = unicodedata.normalize("NFD", text)
        unaccented = "".join([c for c in nfd_text if unicodedata.category(c) != "Mn"])
        return unaccented.replace("đ", "d").replace("Đ", "D")

try:
    import Levenshtein
    def get_levenshtein_ratio(s1: str, s2: str) -> float:
        return Levenshtein.ratio(s1, s2)
except ImportError:
    import difflib
    def get_levenshtein_ratio(s1: str, s2: str) -> float:
        return difflib.SequenceMatcher(None, s1, s2).ratio()


def normalize_text(text: Optional[str]) -> Optional[str]:
    """
    Normalizes unicode text to NFC and strips excessive whitespaces.
    """
    if not text:
        return None
    text = unicodedata.normalize("NFC", text.strip())
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_vietnamese_accents(text: Optional[str]) -> str:
    """
    Removes Vietnamese accent marks and converts to uppercase for matching.
    """
    if not text:
        return ""
    text_nfc = unicodedata.normalize("NFC", text)
    unaccented = unidecode(text_nfc)
    return unaccented.upper().strip()


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """
    Converts various date formats (DD/MM/YYYY, DDMMYYYY, YYMMDD) to ISO YYYY-MM-DD.
    Strictly outputs ISO YYYY-MM-DD format.
    """
    if not date_str:
        return None

    clean_str = re.sub(r'[^\d]', '', str(date_str).strip())

    # Format DDMMYYYY (8 digits) -> YYYY-MM-DD
    if len(clean_str) == 8:
        day = clean_str[:2]
        month = clean_str[2:4]
        year = clean_str[4:]
        if 1 <= int(day) <= 31 and 1 <= int(month) <= 12:
            return f"{year}-{month}-{day}"

    # Format YYMMDD (6 digits from MRZ) -> YYYY-MM-DD
    if len(clean_str) == 6:
        yy = int(clean_str[:2])
        mm = clean_str[2:4]
        dd = clean_str[4:6]

        year = f"19{yy:02d}" if yy > 40 else f"20{yy:02d}"
        if 1 <= int(dd) <= 31 and 1 <= int(mm) <= 12:
            return f"{year}-{mm}-{dd}"

    # Regex for DD/MM/YYYY or DD-MM-YYYY
    match = re.search(r'\b(0[1-9]|[12]\d|3[01])[/.-](0[1-9]|1[0-2])[/.-]((?:19|20)\d{2})\b', str(date_str).strip())
    if match:
        day, month, year = match.groups()
        return f"{year}-{month}-{day}"

    return None


def compare_names(name1: Optional[str], name2: Optional[str]) -> float:
    """
    Computes similarity ratio between two Vietnamese full names after accent removal.
    Returns float between 0.0 and 1.0.
    """
    if not name1 or not name2:
        return 0.0

    clean_name1 = remove_vietnamese_accents(name1)
    clean_name2 = remove_vietnamese_accents(name2)

    if clean_name1 == clean_name2:
        return 1.0

    return get_levenshtein_ratio(clean_name1, clean_name2)

