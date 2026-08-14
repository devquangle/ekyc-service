import re
import difflib
from typing import Dict, List, Optional, Tuple
from utils.text_utils import remove_vietnamese_accents

FIELD_LABELS: Dict[str, List[str]] = {
    # 1. Số định danh / Số CCCD
    "identityNumber": [
        # Thẻ Mới
        "số định danh cá nhân / personal identification number",
        "số định danh cá nhân / no:",
        "số định danh cá nhân / no",
        "số định danh cá nhân",
        "personal identification number",
        # Thẻ Cũ — keep only forms long enough to be unambiguous
        "số / no.:",
        "số / no:",
        "số / no",
        "số/no.:",
        "số/no:",
    ],

    # 2. Họ và tên
    "fullName": [
        # Thẻ Mới
        "họ, chữ đệm và tên khai sinh / surname, given names:",
        "họ, chữ đệm và tên khai sinh / surname, given names",
        "họ, chữ đệm và tên khai sinh / full name:",
        "họ, chữ đệm và tên khai sinh / full name",
        "họ, chữ đệm và tên khai sinh:",
        "họ, chữ đệm và tên khai sinh",
        "surname, given names:",
        "surname, given names",
        # Thẻ Cũ
        "họ và tên / full name:",
        "họ và tên / full name",
        "họ và tên:",
        "họ và tên",
        "full name:",
        "full name",
    ],

    # 3. Ngày sinh
    "dateOfBirth": [
        # Thẻ Mới
        "ngày, tháng, năm sinh / date of birth:",
        "ngày, tháng, năm sinh / date of birth",
        "ngày, tháng, năm sinh:",
        "ngày, tháng, năm sinh",
        # Thẻ Cũ
        "ngày sinh / date of birth:",
        "ngày sinh / date of birth",
        "ngày sinh:",
        "ngày sinh",
        "date of birth:",
        "date of birth",
        "dob:",
    ],

    # 4. Giới tính
    "gender": [
        "giới tính / sex:",
        "giới tính / sex",
        "giới tính:",
        "giới tính",
        "sex:",
        "sex",
    ],

    # 5. Quốc tịch
    "nationality": [
        "quốc tịch / nationality:",
        "quốc tịch / nationality",
        "quốc tịch:",
        "quốc tịch",
        "nationality:",
        "nationality",
    ],

    # 6. Quê quán / Nơi đăng ký khai sinh
    "placeOfOrigin": [
        # Thẻ Mới (Mặt sau)
        "nơi đăng ký khai sinh / place of birth registration:",
        "nơi đăng ký khai sinh / place of birth registration",
        "nơi đăng ký khai sinh / place of birth:",
        "nơi đăng ký khai sinh / place of birth",
        "nơi đăng ký khai sinh:",
        "nơi đăng ký khai sinh",
        "place of birth registration:",
        "place of birth registration",
        "place of birth:",
        "place of birth",
        # Thẻ Cũ (Mặt trước)
        "quê quán / place of origin:",
        "quê quán / place of origin",
        "quê quán:",
        "quê quán",
        "place of origin:",
        "place of origin",
    ],

    # 7. Nơi cư trú / Nơi thường trú
    "placeOfResidence": [
        # Thẻ Mới (Mặt sau)
        "nơi cư trú / place of residence:",
        "nơi cư trú / place of residence",
        "nơi cư trú:",
        "nơi cư trú",
        # Thẻ Cũ (Mặt trước)
        "nơi thường trú / place of residence:",
        "nơi thường trú / place of residence",
        "nơi thường trú:",
        "nơi thường trú",
        "place of residence:",
        "place of residence",
    ],

    # 8. Ngày cấp (Mặt sau)
    "dateOfIssue": [
        # Thẻ Mới
        "ngày, tháng, năm cấp / date of issue:",
        "ngày, tháng, năm cấp / date of issue",
        "ngày, tháng, năm cấp / date, month, year:",
        "ngày, tháng, năm cấp / date, month, year",
        "ngày, tháng, năm cấp:",
        "ngày, tháng, năm cấp",
        "date of issue:",
        "date of issue",
        # Thẻ Cũ
        "ngày, tháng, năm / date, month, year:",
        "ngày, tháng, năm / date, month, year",
        "date, month, year:",
        "date, month, year",
        "ngày cấp",
        "ngay cap",
    ],

    # 9. Ngày hết hạn
    "dateOfExpiry": [
        # Thẻ Mới (Mặt sau)
        "ngày, tháng, năm hết hạn / date of expiry:",
        "ngày, tháng, năm hết hạn / date of expiry",
        "ngày, tháng, năm hết hạn:",
        "ngày, tháng, năm hết hạn",
        "ngay, thang, nam het han / date of expiry:",
        "ngay, thang, nam het han / date of expiry",
        "ngay, thang, nam het han",
        # Thẻ Cũ (Mặt trước)
        "có giá trị đến / date of expiry:",
        "có giá trị đến / date of expiry",
        "có giá trị đến:",
        "có giá trị đến",
        "date of expiry:",
        "date of expiry",
    ],
}


# ---------------------------------------------------------------------------
# Card Type Classification Keywords
# ---------------------------------------------------------------------------
CARD_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "CCCD_NEW": [
        "căn cước",
        "identity card",
        "số định danh cá nhân",
        "personal identification number",
        "họ, chữ đệm và tên khai sinh",
        "nơi đăng ký khai sinh",
        "nơi cư trú",
        "ngày, tháng, năm hết hạn",
        "bộ công an",
        "ministry of public security",
    ],
    "CCCD_OLD": [
        "căn cước công dân",
        "citizen identity card",
        "quê quán",
        "nơi thường trú",
        "đặc điểm nhận dạng",
        "cục trưởng cục cảnh sát",
        "ngón trỏ trái",
        "ngón trỏ phải",
    ],
}


# ---------------------------------------------------------------------------
# Address Overflow Stop Keywords
# Used by _extract_address_field to prevent reading beyond address boundaries.
# All entries are accent-stripped lowercase for fast matching via
# remove_vietnamese_accents(line.text).lower().
# ---------------------------------------------------------------------------
ADDRESS_STOP_KEYWORDS: Dict[str, List[str]] = {
    # Stops for placeOfResidence
    "placeOfResidence": [
        "noi dang ky khai sinh",
        "place of birth registration",
        "place of birth",
        "que quan",
        "place of origin",
        "co gia tri den",
        "date of expiry",
        "ngay thang nam het han",
        "ngay thang nam cap",
        "date of issue",
        "bo cong an",
        "ministry",
        "cuc truong",
        "ngon tro trai",
    ],
    # Stops for placeOfOrigin
    "placeOfOrigin": [
        "noi thuong tru",
        "noi cu tru",
        "place of residence",
        "co gia tri den",
        "date of expiry",
        "ngay thang nam het han",
        "ngay thang nam cap",
        "date of issue",
        "bo cong an",
        "ministry",
        "cuc truong",
    ],
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
