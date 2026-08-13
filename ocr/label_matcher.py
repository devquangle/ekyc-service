import re
import difflib
from typing import Dict, List, Optional, Tuple
from utils.text_utils import remove_vietnamese_accents

FIELD_LABELS: Dict[str, List[str]] = {
    "identityNumber": [
        "số định danh cá nhân / no:",
        "số định danh cá nhân / no",
        "số định danh cá nhân",
        "số / no.:",
        "số / no:",
        "personal identification number",
    ],
    "fullName": [
        "họ, chữ đệm và tên khai sinh / surname, given names:",
        "họ, chữ đệm và tên khai sinh / surname, given names",
        "họ, chữ đệm và tên khai sinh",
        "surname, given names",
        "họ và tên / full name:",
        "họ và tên / full name",
        "họ và tên",
        "full name",
    ],
    "dateOfBirth": [
        "ngày, tháng, năm sinh / date of birth:",
        "ngày, tháng, năm sinh / date of birth",
        "ngày, tháng, năm sinh",
        "ngày sinh / date of birth:",
        "ngày sinh / date of birth",
        "ngày sinh",
        "date of birth",
    ],
    "gender": [
        "giới tính / sex:",
        "giới tính / sex",
        "giới tính",
    ],
    "nationality": [
        "quốc tịch / nationality:",
        "quốc tịch / nationality",
        "quốc tịch",
        "nationality",
    ],
    "placeOfOrigin": [
        "nơi đăng ký khai sinh / place of birth registration:",
        "nơi đăng ký khai sinh / place of birth registration",
        "nơi đăng ký khai sinh",
        "place of birth registration",
        "quê quán / place of origin:",
        "quê quán / place of origin",
        "quê quán",
        "place of origin",
    ],
    "placeOfResidence": [
        "nơi cư trú / place of residence:",
        "nơi cư trú / place of residence",
        "nơi cư trú",
        "nơi thường trú / place of residence:",
        "nơi thường trú / place of residence",
        "nơi thường trú",
        "place of residence",
    ],
    "dateOfIssue": [
        "ngày, tháng, năm cấp / date, month, year",
        "ngày, tháng, năm cấp",
        "ngày, tháng, năm / date, month, year:",
        "ngày, tháng, năm / date, month, year",
        "date, month, year",
    ],
    "dateOfExpiry": [
        "có giá trị đến / date of expiry",
        "có giá trị đến",
        "date of expiry",
    ]
}


class LabelMatcher:
    """
    Generic Data-Driven Label Matcher.
    Uses normalized string matching and sequence/token similarity against official field labels
    to recognize noisy OCR labels without hardcoding specific typos.
    """

    def __init__(self, min_similarity_threshold: float = 0.65):
        self.min_similarity_threshold = min_similarity_threshold

    def match_line_label(self, line_text: str) -> Optional[Tuple[str, str, float]]:
        if not line_text or len(line_text.strip()) < 2:
            return None

        clean_line = remove_vietnamese_accents(line_text).lower()
        clean_line_nopunct = re.sub(r'[\/._\-:]+', ' ', clean_line).strip()
        clean_line_nopunct = re.sub(r'\s+', ' ', clean_line_nopunct)

        best_match: Optional[Tuple[str, str, float]] = None
        best_score = 0.0

        for field_name, labels in FIELD_LABELS.items():
            for label in labels:
                label_unaccented = remove_vietnamese_accents(label).lower()
                label_clean = re.sub(r'[\/._\-:]+', ' ', label_unaccented).strip()
                label_clean = re.sub(r'\s+', ' ', label_clean)

                # 1. Exact or Word Boundary Substring Match
                if len(label_clean) <= 4:
                    pattern = r'\b' + re.escape(label_clean) + r'[:\s\/._]*'
                    if re.search(pattern, clean_line) or re.search(pattern, clean_line_nopunct):
                        return (field_name, label, 1.0)
                else:
                    pattern = r'\b' + re.escape(label_clean) + r'\b'
                    if re.search(pattern, clean_line) or re.search(pattern, clean_line_nopunct) or label_clean in clean_line_nopunct:
                        return (field_name, label, 1.0)

                # 2. Generic Fuzzy Matching (Sequence & Substring Character Similarity)
                if len(label_clean) > 4:
                    score = self._compute_fuzzy_similarity(clean_line_nopunct, label_clean)
                    if score >= self.min_similarity_threshold and score > best_score:
                        best_score = score
                        best_match = (field_name, label, round(best_score, 2))

        return best_match

    def _compute_fuzzy_similarity(self, text: str, label: str) -> float:
        # Full string similarity ratio
        full_ratio = difflib.SequenceMatcher(None, text, label).ratio()

        # Compact (space-removed) similarity ratio for concatenated OCR tokens
        text_compact = text.replace(" ", "")
        label_compact = label.replace(" ", "")
        compact_ratio = difflib.SequenceMatcher(None, text_compact, label_compact).ratio()

        return max(full_ratio, compact_ratio)
