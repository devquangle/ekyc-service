import re
import unicodedata
from datetime import datetime
from typing import Optional, List

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


def _is_valid_date(year: int, month: int, day: int) -> bool:
    """
    Validates calendar date according to the Gregorian calendar (e.g. rejects 30/02 or 31/04).
    """
    try:
        if year < 1900 or year > 2100:
            return False
        datetime(year, month, day)
        return True
    except (ValueError, OverflowError):
        return False


def normalize_text(text: Optional[str]) -> Optional[str]:
    """
    Normalizes unicode text to NFC and strips excessive whitespaces.
    """
    if not text:
        return None
    text = unicodedata.normalize("NFC", text.strip())
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
    Converts various date formats (DD/MM/YYYY, D/M/YYYY, DDMMYYYY, YYMMDD) to ISO YYYY-MM-DD.
    Enforces strict Gregorian calendar validation to eliminate impossible dates (e.g. 30/02, 31/04).
    """
    if not date_str:
        return None

    raw_str = str(date_str).strip()

    # 1. Regex for DD/MM/YYYY or D/M/YYYY or DD-MM-YYYY
    match = re.search(
        r'\b([1-9]|0[1-9]|[12]\d|3[01])[/.-]([1-9]|0[1-9]|1[0-2])[/.-]((?:19|20)\d{2})\b',
        raw_str
    )
    if match:
        day_raw, month_raw, year_raw = match.groups()
        day = int(day_raw)
        month = int(month_raw)
        year = int(year_raw)
        if _is_valid_date(year, month, day):
            return f"{year:04d}-{month:02d}-{day:02d}"

    # 2. Extract only digits
    clean_str = re.sub(r'[^\d]', '', raw_str)

    # Format DDMMYYYY (8 digits) -> YYYY-MM-DD
    if len(clean_str) == 8:
        day = int(clean_str[:2])
        month = int(clean_str[2:4])
        year = int(clean_str[4:])
        if _is_valid_date(year, month, day):
            return f"{year:04d}-{month:02d}-{day:02d}"

    # Format YYMMDD (6 digits from MRZ) -> YYYY-MM-DD
    if len(clean_str) == 6:
        yy = int(clean_str[:2])
        month = int(clean_str[2:4])
        day = int(clean_str[4:6])
        current_year_short = datetime.now().year % 100
        # If yy <= current_year_short + 10 -> 20yy, else 19yy
        year = (2000 + yy) if yy <= (current_year_short + 15) else (1900 + yy)
        if _is_valid_date(year, month, day):
            return f"{year:04d}-{month:02d}-{day:02d}"

    return None


def compare_names(name1: Optional[str], name2: Optional[str]) -> float:
    """
    Computes similarity ratio between two Vietnamese full names after accent removal.
    Supports order-insensitive token matching, abbreviations/initials, and Levenshtein distance.
    Returns float between 0.0 and 1.0.
    """
    if not name1 or not name2:
        return 0.0

    clean_name1 = remove_vietnamese_accents(name1)
    clean_name2 = remove_vietnamese_accents(name2)

    if not clean_name1 or not clean_name2:
        return 0.0

    if clean_name1 == clean_name2:
        return 1.0

    tokens1 = clean_name1.split()
    tokens2 = clean_name2.split()

    if not tokens1 or not tokens2:
        return 0.0

    # 1. Exact word tokens permutation (e.g. "NGUYEN VAN A" vs "A NGUYEN VAN")
    if sorted(tokens1) == sorted(tokens2):
        return 1.0

    # 2. Token overlap ratio (Jaccard)
    set1 = set(tokens1)
    set2 = set(tokens2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    jaccard_ratio = len(intersection) / float(len(union)) if union else 0.0

    # 3. Check for prefix/abbreviation matching (e.g. "N VAN A" vs "NGUYEN VAN A" or "H QUANG LE" vs "HUYNH QUANG LE")
    abbrev_match = False
    if len(tokens1) == len(tokens2) and len(tokens1) >= 2:
        matches = 0
        for t1, t2 in zip(tokens1, tokens2):
            if t1 == t2 or (len(t1) == 1 and t2.startswith(t1)) or (len(t2) == 1 and t1.startswith(t2)):
                matches += 1
        if matches == len(tokens1):
            abbrev_match = True

    # 4. Levenshtein ratio on original string and sorted token string
    sorted_str1 = " ".join(sorted(tokens1))
    sorted_str2 = " ".join(sorted(tokens2))
    lev_sorted = get_levenshtein_ratio(sorted_str1, sorted_str2)
    lev_direct = get_levenshtein_ratio(clean_name1, clean_name2)

    candidates = [lev_direct, lev_sorted, jaccard_ratio]
    if abbrev_match:
        candidates.append(0.95)

    return float(max(candidates))
