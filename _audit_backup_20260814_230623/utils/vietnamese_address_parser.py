"""
Vietnamese Address Parser & Hierarchical Normalizer
===================================================
A high-performance Vietnamese address parser and hierarchical normalizer powered by
General Statistics Office (GSO) administrative database structures and RapidFuzz.

Decomposes and standardizes raw Vietnamese address strings into 4 distinct levels:
1. `province`: Tỉnh / Thành phố trực thuộc trung ương (Level 1)
2. `district`: Quận / Huyện / Thị xã / Thành phố thuộc tỉnh (Level 2)
3. `ward`: Phường / Xã / Thị trấn (Level 3)
4. `street`: Số nhà, tên đường, ngõ/ngách, thôn/ấp/tổ/khóm/bản/làng (Level 4)

Key Features:
- Right-to-Left hierarchical candidate pruning (Province -> District -> Ward -> Street)
- RapidFuzz fuzzy matching with OCR error tolerance (Mojibake, missing diacritics, character confusion)
- Specialized regex handlers for numerical administrative units (Quận 1, Phường 10, Tổ 5)
- Short place name boundary protection (Huế, Ba Vì, Ô Môn, Ea H'leo)
- Standardized administrative prefix normalization
- Fully typed Pydantic V2 schema output with GSO codes and confidence scoring
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict

import sys
from pathlib import Path

# Add root directory to sys.path for standalone script execution
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from utils.text_normalizer import UnicodeNormalizer, MojibakeFixer, VietnameseTextCorrector
from utils.text_utils import remove_vietnamese_accents


# ===========================================================================
# 1. Administrative Prefix Mapping & Normalization Dictionaries
# ===========================================================================

ADMINISTRATIVE_PREFIX_MAP: Dict[str, str] = {
    # Province / City level
    "thanh pho": "TP.",
    "thành phố": "TP.",
    "tp.": "TP.",
    "tp": "TP.",
    "tinh": "Tỉnh",
    "tỉnh": "Tỉnh",
    "t.": "Tỉnh",
    "t": "Tỉnh",

    # District / Town level
    "quan": "Quận",
    "quận": "Quận",
    "q.": "Quận",
    "q": "Quận",
    "huyen": "Huyện",
    "huyện": "Huyện",
    "h.": "Huyện",
    "h": "Huyện",
    "thi xa": "Thị xã",
    "thị xã": "Thị xã",
    "tx.": "Thị xã",
    "tx": "Thị xã",

    # Ward / Commune / Town level
    "phuong": "Phường",
    "phường": "Phường",
    "p.": "Phường",
    "p": "Phường",
    "f.": "Phường",
    "f": "Phường",
    "xa": "Xã",
    "xã": "Xã",
    "x.": "Xã",
    "x": "Xã",
    "thi tran": "Thị trấn",
    "thị trấn": "Thị trấn",
    "tt.": "Thị trấn",
    "tt": "Thị trấn",

    # Street / Village / Hamlet / Group level
    "duong": "Đường",
    "đường": "Đường",
    "d.": "Đ.",
    "đ.": "Đ.",
    "so": "Số",
    "số": "Số",
    "so nha": "Số nhà",
    "số nhà": "Số nhà",
    "ngo": "Ngõ",
    "ngõ": "Ngõ",
    "ngach": "Ngách",
    "ngách": "Ngách",
    "hem": "Hẻm",
    "hẻm": "Hẻm",
    "ap": "Ấp",
    "ấp": "Ấp",
    "thon": "Thôn",
    "thôn": "Thôn",
    "to": "Tổ",
    "tổ": "Tổ",
    "khom": "Khóm",
    "khóm": "Khóm",
    "khu pho": "Khu phố",
    "khu phố": "Khu phố",
    "kp": "Khu phố",
    "kp.": "Khu phố",
    "ban": "Bản",
    "bản": "Bản",
    "lang": "Làng",
    "làng": "Làng",
    "buon": "Buôn",
    "buôn": "Buôn",
    "soc": "Sóc",
    "sóc": "Sóc",
    "cum": "Cụm",
    "cụm": "Cụm",
    "khu": "Khu",
}

# Aliases for common abbreviations
PROVINCE_ALIASES: Dict[str, str] = {
    "hcm": "TP. Hồ Chí Minh",
    "tp hcm": "TP. Hồ Chí Minh",
    "tp. hcm": "TP. Hồ Chí Minh",
    "tp.hcm": "TP. Hồ Chí Minh",
    "tphcm": "TP. Hồ Chí Minh",
    "sai gon": "TP. Hồ Chí Minh",
    "saigon": "TP. Hồ Chí Minh",
    "sg": "TP. Hồ Chí Minh",
    "hn": "Hà Nội",
    "tp hn": "Hà Nội",
    "tp. hn": "Hà Nội",
    "tp.hn": "Hà Nội",
    "tphn": "Hà Nội",
    "ha noi": "Hà Nội",
    "tp ha noi": "Hà Nội",
    "tp. ha noi": "Hà Nội",
    "thanh pho ha noi": "Hà Nội",
    "da nang": "Đà Nẵng",
    "tp da nang": "Đà Nẵng",
    "tp. da nang": "Đà Nẵng",
    "hai phong": "Hải Phòng",
    "tp hai phong": "Hải Phòng",
    "tp. hai phong": "Hải Phòng",
    "can tho": "Cần Thơ",
    "tp can tho": "Cần Thơ",
    "tp. can tho": "Cần Thơ",
    "ba ria - vung tau": "Bà Rịa - Vũng Tàu",
    "brvt": "Bà Rịa - Vũng Tàu",
    "br-vt": "Bà Rịa - Vũng Tàu",
    "ba ria vung tau": "Bà Rịa - Vũng Tàu",
    "thua thien hue": "Thừa Thiên Huế",
    "tt hue": "Thừa Thiên Huế",
    "tt-hue": "Thừa Thiên Huế",
    "hue": "Thừa Thiên Huế",
}


# ===========================================================================
# 2. Output Schema
# ===========================================================================

class ParsedAddress(BaseModel):
    """
    Standardized Pydantic V2 Model representing the structured 4-level Vietnamese address.
    """
    model_config = ConfigDict(populate_by_name=True)

    original_address: str = Field(..., description="Original raw address string before processing")
    province: Optional[str] = Field(None, description="Level 1: Standardized Province / Municipality name")
    district: Optional[str] = Field(None, description="Level 2: Standardized District / County / Town name")
    ward: Optional[str] = Field(None, description="Level 3: Standardized Ward / Commune / Township name")
    street: Optional[str] = Field(None, description="Level 4: Street number, street name, hamlet, village, or group")
    province_code: Optional[str] = Field(None, description="Official GSO Province Code")
    district_code: Optional[str] = Field(None, description="Official GSO District Code")
    ward_code: Optional[str] = Field(None, description="Official GSO Ward Code")
    full_normalized_address: str = Field(..., description="Complete comma-separated standardized address")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Overall parsing confidence score (0.0 - 1.0)")


class GsoAdminUnit(BaseModel):
    """
    Internal administrative unit node representation.
    """
    code: str
    name: str
    unit_type: str = ""
    normalized_name: str = ""
    sub_units: Dict[str, GsoAdminUnit] = Field(default_factory=dict)


# ===========================================================================
# 3. Comprehensive GSO Administrative In-Memory Database (All 63 Provinces)
# ===========================================================================

GSO_PROVINCES_CATALOG: List[Dict[str, Any]] = [
    {"code": "01", "name": "Hà Nội", "unit_type": "Thành phố Trung ương"},
    {"code": "02", "name": "Hà Giang", "unit_type": "Tỉnh"},
    {"code": "04", "name": "Cao Bằng", "unit_type": "Tỉnh"},
    {"code": "06", "name": "Bắc Kạn", "unit_type": "Tỉnh"},
    {"code": "08", "name": "Tuyên Quang", "unit_type": "Tỉnh"},
    {"code": "10", "name": "Lào Cai", "unit_type": "Tỉnh"},
    {"code": "11", "name": "Điện Biên", "unit_type": "Tỉnh"},
    {"code": "12", "name": "Lai Châu", "unit_type": "Tỉnh"},
    {"code": "14", "name": "Sơn La", "unit_type": "Tỉnh"},
    {"code": "15", "name": "Yên Bái", "unit_type": "Tỉnh"},
    {"code": "17", "name": "Hòa Bình", "unit_type": "Tỉnh"},
    {"code": "19", "name": "Thái Nguyên", "unit_type": "Tỉnh"},
    {"code": "20", "name": "Lạng Sơn", "unit_type": "Tỉnh"},
    {"code": "22", "name": "Quảng Ninh", "unit_type": "Tỉnh"},
    {"code": "24", "name": "Bắc Giang", "unit_type": "Tỉnh"},
    {"code": "25", "name": "Phú Thọ", "unit_type": "Tỉnh"},
    {"code": "26", "name": "Vĩnh Phúc", "unit_type": "Tỉnh"},
    {"code": "27", "name": "Bắc Ninh", "unit_type": "Tỉnh"},
    {"code": "30", "name": "Hải Dương", "unit_type": "Tỉnh"},
    {"code": "31", "name": "Hải Phòng", "unit_type": "Thành phố Trung ương"},
    {"code": "33", "name": "Hưng Yên", "unit_type": "Tỉnh"},
    {"code": "34", "name": "Thái Bình", "unit_type": "Tỉnh"},
    {"code": "35", "name": "Hà Nam", "unit_type": "Tỉnh"},
    {"code": "36", "name": "Nam Định", "unit_type": "Tỉnh"},
    {"code": "37", "name": "Ninh Bình", "unit_type": "Tỉnh"},
    {"code": "38", "name": "Thanh Hóa", "unit_type": "Tỉnh"},
    {"code": "40", "name": "Nghệ An", "unit_type": "Tỉnh"},
    {"code": "42", "name": "Hà Tĩnh", "unit_type": "Tỉnh"},
    {"code": "44", "name": "Quảng Bình", "unit_type": "Tỉnh"},
    {"code": "45", "name": "Quảng Trị", "unit_type": "Tỉnh"},
    {"code": "46", "name": "Thừa Thiên Huế", "unit_type": "Tỉnh"},
    {"code": "48", "name": "Đà Nẵng", "unit_type": "Thành phố Trung ương"},
    {"code": "49", "name": "Quảng Nam", "unit_type": "Tỉnh"},
    {"code": "51", "name": "Quảng Ngãi", "unit_type": "Tỉnh"},
    {"code": "52", "name": "Bình Định", "unit_type": "Tỉnh"},
    {"code": "54", "name": "Phú Yên", "unit_type": "Tỉnh"},
    {"code": "56", "name": "Khánh Hòa", "unit_type": "Tỉnh"},
    {"code": "58", "name": "Ninh Thuận", "unit_type": "Tỉnh"},
    {"code": "60", "name": "Bình Thuận", "unit_type": "Tỉnh"},
    {"code": "62", "name": "Kon Tum", "unit_type": "Tỉnh"},
    {"code": "64", "name": "Gia Lai", "unit_type": "Tỉnh"},
    {"code": "66", "name": "Đắk Lắk", "unit_type": "Tỉnh"},
    {"code": "67", "name": "Đắk Nông", "unit_type": "Tỉnh"},
    {"code": "68", "name": "Lâm Đồng", "unit_type": "Tỉnh"},
    {"code": "70", "name": "Bình Phước", "unit_type": "Tỉnh"},
    {"code": "72", "name": "Tây Ninh", "unit_type": "Tỉnh"},
    {"code": "74", "name": "Bình Dương", "unit_type": "Tỉnh"},
    {"code": "75", "name": "Đồng Nai", "unit_type": "Tỉnh"},
    {"code": "77", "name": "Bà Rịa - Vũng Tàu", "unit_type": "Tỉnh"},
    {"code": "79", "name": "TP. Hồ Chí Minh", "unit_type": "Thành phố Trung ương"},
    {"code": "80", "name": "Long An", "unit_type": "Tỉnh"},
    {"code": "82", "name": "Tiền Giang", "unit_type": "Tỉnh"},
    {"code": "83", "name": "Bến Tre", "unit_type": "Tỉnh"},
    {"code": "84", "name": "Trà Vinh", "unit_type": "Tỉnh"},
    {"code": "86", "name": "Vĩnh Long", "unit_type": "Tỉnh"},
    {"code": "87", "name": "Đồng Tháp", "unit_type": "Tỉnh"},
    {"code": "89", "name": "An Giang", "unit_type": "Tỉnh"},
    {"code": "91", "name": "Kiên Giang", "unit_type": "Tỉnh"},
    {"code": "92", "name": "Cần Thơ", "unit_type": "Thành phố Trung ương"},
    {"code": "93", "name": "Hậu Giang", "unit_type": "Tỉnh"},
    {"code": "94", "name": "Sóc Trăng", "unit_type": "Tỉnh"},
    {"code": "95", "name": "Bạc Liêu", "unit_type": "Tỉnh"},
    {"code": "96", "name": "Cà Mau", "unit_type": "Tỉnh"},
]

# Standard Regional Districts mapping
GSO_DISTRICTS_DATA: Dict[str, List[Dict[str, Any]]] = {
    "01": [  # Hà Nội
        {"code": "001", "name": "Ba Đình", "unit_type": "Quận"},
        {"code": "002", "name": "Hoàn Kiếm", "unit_type": "Quận"},
        {"code": "003", "name": "Tây Hồ", "unit_type": "Quận"},
        {"code": "004", "name": "Long Biên", "unit_type": "Quận"},
        {"code": "005", "name": "Cầu Giấy", "unit_type": "Quận"},
        {"code": "006", "name": "Đống Đa", "unit_type": "Quận"},
        {"code": "007", "name": "Hai Bà Trưng", "unit_type": "Quận"},
        {"code": "008", "name": "Hoàng Mai", "unit_type": "Quận"},
        {"code": "009", "name": "Thanh Xuân", "unit_type": "Quận"},
        {"code": "016", "name": "Sóc Sơn", "unit_type": "Huyện"},
        {"code": "017", "name": "Đông Anh", "unit_type": "Huyện"},
        {"code": "018", "name": "Gia Lâm", "unit_type": "Huyện"},
        {"code": "019", "name": "Nam Từ Liêm", "unit_type": "Quận"},
        {"code": "020", "name": "Thanh Trì", "unit_type": "Huyện"},
        {"code": "021", "name": "Bắc Từ Liêm", "unit_type": "Quận"},
        {"code": "250", "name": "Mê Linh", "unit_type": "Huyện"},
        {"code": "268", "name": "Hà Đông", "unit_type": "Quận"},
        {"code": "269", "name": "Sơn Tây", "unit_type": "Thị xã"},
        {"code": "271", "name": "Ba Vì", "unit_type": "Huyện"},
        {"code": "272", "name": "Phúc Thọ", "unit_type": "Huyện"},
        {"code": "273", "name": "Đan Phượng", "unit_type": "Huyện"},
        {"code": "274", "name": "Hoài Đức", "unit_type": "Huyện"},
        {"code": "275", "name": "Quốc Oai", "unit_type": "Huyện"},
        {"code": "276", "name": "Thạch Thất", "unit_type": "Huyện"},
        {"code": "277", "name": "Chương Mỹ", "unit_type": "Huyện"},
        {"code": "278", "name": "Thanh Oai", "unit_type": "Huyện"},
        {"code": "279", "name": "Thường Tín", "unit_type": "Huyện"},
        {"code": "280", "name": "Phú Xuyên", "unit_type": "Huyện"},
        {"code": "281", "name": "Ứng Hòa", "unit_type": "Huyện"},
        {"code": "282", "name": "Mỹ Đức", "unit_type": "Huyện"},
    ],
    "79": [  # TP. Hồ Chí Minh
        {"code": "760", "name": "Quận 1", "unit_type": "Quận"},
        {"code": "761", "name": "Quận 12", "unit_type": "Quận"},
        {"code": "764", "name": "Gò Vấp", "unit_type": "Quận"},
        {"code": "765", "name": "Bình Thạnh", "unit_type": "Quận"},
        {"code": "766", "name": "Tân Bình", "unit_type": "Quận"},
        {"code": "767", "name": "Tân Phú", "unit_type": "Quận"},
        {"code": "768", "name": "Phú Nhuận", "unit_type": "Quận"},
        {"code": "769", "name": "Thủ Đức", "unit_type": "Thành phố"},
        {"code": "770", "name": "Quận 3", "unit_type": "Quận"},
        {"code": "771", "name": "Quận 10", "unit_type": "Quận"},
        {"code": "772", "name": "Quận 11", "unit_type": "Quận"},
        {"code": "773", "name": "Quận 4", "unit_type": "Quận"},
        {"code": "774", "name": "Quận 5", "unit_type": "Quận"},
        {"code": "775", "name": "Quận 6", "unit_type": "Quận"},
        {"code": "776", "name": "Quận 8", "unit_type": "Quận"},
        {"code": "777", "name": "Bình Tân", "unit_type": "Quận"},
        {"code": "778", "name": "Quận 7", "unit_type": "Quận"},
        {"code": "783", "name": "Củ Chi", "unit_type": "Huyện"},
        {"code": "784", "name": "Hóc Môn", "unit_type": "Huyện"},
        {"code": "785", "name": "Bình Chánh", "unit_type": "Huyện"},
        {"code": "786", "name": "Nhà Bè", "unit_type": "Huyện"},
        {"code": "787", "name": "Cần Giờ", "unit_type": "Huyện"},
    ],
    "48": [  # Đà Nẵng
        {"code": "490", "name": "Hải Châu", "unit_type": "Quận"},
        {"code": "491", "name": "Thanh Khê", "unit_type": "Quận"},
        {"code": "492", "name": "Sơn Trà", "unit_type": "Quận"},
        {"code": "493", "name": "Ngũ Hành Sơn", "unit_type": "Quận"},
        {"code": "494", "name": "Liên Chiểu", "unit_type": "Quận"},
        {"code": "495", "name": "Cẩm Lệ", "unit_type": "Quận"},
        {"code": "497", "name": "Hòa Vang", "unit_type": "Huyện"},
        {"code": "498", "name": "Hoàng Sa", "unit_type": "Huyện"},
    ],
    "31": [  # Hải Phòng
        {"code": "303", "name": "Hồng Bàng", "unit_type": "Quận"},
        {"code": "304", "name": "Ngô Quyền", "unit_type": "Quận"},
        {"code": "305", "name": "Lê Chân", "unit_type": "Quận"},
        {"code": "306", "name": "Hải An", "unit_type": "Quận"},
        {"code": "307", "name": "Kiến An", "unit_type": "Quận"},
        {"code": "308", "name": "Đồ Sơn", "unit_type": "Quận"},
        {"code": "309", "name": "Dương Kinh", "unit_type": "Quận"},
        {"code": "311", "name": "Thủy Nguyên", "unit_type": "Huyện"},
        {"code": "312", "name": "An Dương", "unit_type": "Huyện"},
        {"code": "313", "name": "An Lão", "unit_type": "Huyện"},
        {"code": "314", "name": "Kiến Thụy", "unit_type": "Huyện"},
        {"code": "315", "name": "Tiên Lãng", "unit_type": "Huyện"},
        {"code": "316", "name": "Vĩnh Bảo", "unit_type": "Huyện"},
        {"code": "317", "name": "Cát Hải", "unit_type": "Huyện"},
        {"code": "318", "name": "Bạch Long Vĩ", "unit_type": "Huyện"},
    ],
    "92": [  # Cần Thơ
        {"code": "916", "name": "Ninh Kiều", "unit_type": "Quận"},
        {"code": "917", "name": "Ô Môn", "unit_type": "Quận"},
        {"code": "918", "name": "Bình Thủy", "unit_type": "Quận"},
        {"code": "919", "name": "Cái Răng", "unit_type": "Quận"},
        {"code": "923", "name": "Thốt Nốt", "unit_type": "Quận"},
        {"code": "924", "name": "Vĩnh Thạnh", "unit_type": "Huyện"},
        {"code": "925", "name": "Cờ Đỏ", "unit_type": "Huyện"},
        {"code": "926", "name": "Phong Điền", "unit_type": "Huyện"},
        {"code": "927", "name": "Thới Lai", "unit_type": "Huyện"},
    ],
    "87": [  # Đồng Tháp
        {"code": "866", "name": "Cao Lãnh", "unit_type": "Thành phố"},
        {"code": "867", "name": "Sa Đéc", "unit_type": "Thành phố"},
        {"code": "868", "name": "Hồng Ngự", "unit_type": "Thành phố"},
        {"code": "869", "name": "Tân Hồng", "unit_type": "Huyện"},
        {"code": "870", "name": "Hồng Ngự", "unit_type": "Huyện"},
        {"code": "871", "name": "Tam Nông", "unit_type": "Huyện"},
        {"code": "872", "name": "Tháp Mười", "unit_type": "Huyện"},
        {"code": "873", "name": "Cao Lãnh", "unit_type": "Huyện"},
        {"code": "874", "name": "Thanh Bình", "unit_type": "Huyện"},
        {"code": "875", "name": "Lấp Vò", "unit_type": "Huyện"},
        {"code": "876", "name": "Lai Vung", "unit_type": "Huyện"},
        {"code": "877", "name": "Châu Thành", "unit_type": "Huyện"},
    ],
    "86": [  # Vĩnh Long
        {"code": "855", "name": "Vĩnh Long", "unit_type": "Thành phố"},
        {"code": "857", "name": "Long Hồ", "unit_type": "Huyện"},
        {"code": "858", "name": "Mang Thít", "unit_type": "Huyện"},
        {"code": "859", "name": "Vũng Liêm", "unit_type": "Huyện"},
        {"code": "860", "name": "Tam Bình", "unit_type": "Huyện"},
        {"code": "861", "name": "Bình Minh", "unit_type": "Thị xã"},
        {"code": "862", "name": "Trà Ôn", "unit_type": "Huyện"},
        {"code": "863", "name": "Bình Tân", "unit_type": "Huyện"},
    ],
    "89": [  # An Giang
        {"code": "883", "name": "Long Xuyên", "unit_type": "Thành phố"},
        {"code": "884", "name": "Châu Đốc", "unit_type": "Thành phố"},
        {"code": "886", "name": "An Phú", "unit_type": "Huyện"},
        {"code": "887", "name": "Tân Châu", "unit_type": "Thị xã"},
        {"code": "888", "name": "Phú Tân", "unit_type": "Huyện"},
        {"code": "889", "name": "Châu Phú", "unit_type": "Huyện"},
        {"code": "890", "name": "Tịnh Biên", "unit_type": "Thị xã"},
        {"code": "891", "name": "Tri Tôn", "unit_type": "Huyện"},
        {"code": "892", "name": "Châu Thành", "unit_type": "Huyện"},
        {"code": "893", "name": "Chợ Mới", "unit_type": "Huyện"},
        {"code": "894", "name": "Thoại Sơn", "unit_type": "Huyện"},
    ],
    "83": [  # Bến Tre
        {"code": "829", "name": "Bến Tre", "unit_type": "Thành phố"},
        {"code": "831", "name": "Châu Thành", "unit_type": "Huyện"},
        {"code": "832", "name": "Chợ Lách", "unit_type": "Huyện"},
        {"code": "833", "name": "Mỏ Cày Nam", "unit_type": "Huyện"},
        {"code": "834", "name": "Giồng Trôm", "unit_type": "Huyện"},
        {"code": "835", "name": "Bình Đại", "unit_type": "Huyện"},
        {"code": "836", "name": "Ba Tri", "unit_type": "Huyện"},
        {"code": "837", "name": "Thạnh Phú", "unit_type": "Huyện"},
        {"code": "838", "name": "Mỏ Cày Bắc", "unit_type": "Huyện"},
    ],
    "82": [  # Tiền Giang
        {"code": "815", "name": "Mỹ Tho", "unit_type": "Thành phố"},
        {"code": "816", "name": "Gò Công", "unit_type": "Thành phố"},
        {"code": "817", "name": "Cai Lậy", "unit_type": "Thị xã"},
        {"code": "818", "name": "Tân Phước", "unit_type": "Huyện"},
        {"code": "819", "name": "Cái Bè", "unit_type": "Huyện"},
        {"code": "820", "name": "Cai Lậy", "unit_type": "Huyện"},
        {"code": "821", "name": "Châu Thành", "unit_type": "Huyện"},
        {"code": "822", "name": "Chợ Gạo", "unit_type": "Huyện"},
        {"code": "823", "name": "Gò Công Tây", "unit_type": "Huyện"},
        {"code": "824", "name": "Gò Công Đông", "unit_type": "Huyện"},
        {"code": "825", "name": "Tân Phú Đông", "unit_type": "Huyện"},
    ],
    "80": [  # Long An
        {"code": "794", "name": "Tân An", "unit_type": "Thành phố"},
        {"code": "795", "name": "Kiến Tường", "unit_type": "Thị xã"},
        {"code": "796", "name": "Tân Hưng", "unit_type": "Huyện"},
        {"code": "797", "name": "Vĩnh Hưng", "unit_type": "Huyện"},
        {"code": "798", "name": "Mộc Hóa", "unit_type": "Huyện"},
        {"code": "799", "name": "Tân Thạnh", "unit_type": "Huyện"},
        {"code": "800", "name": "Thạnh Hóa", "unit_type": "Huyện"},
        {"code": "801", "name": "Đức Huệ", "unit_type": "Huyện"},
        {"code": "802", "name": "Đức Hòa", "unit_type": "Huyện"},
        {"code": "803", "name": "Bến Lức", "unit_type": "Huyện"},
        {"code": "804", "name": "Thủ Thừa", "unit_type": "Huyện"},
        {"code": "805", "name": "Tân Trụ", "unit_type": "Huyện"},
        {"code": "806", "name": "Cần Đước", "unit_type": "Huyện"},
        {"code": "807", "name": "Cần Giuộc", "unit_type": "Huyện"},
        {"code": "808", "name": "Châu Thành", "unit_type": "Huyện"},
    ],
    "74": [  # Bình Dương
        {"code": "718", "name": "Thủ Dầu Một", "unit_type": "Thành phố"},
        {"code": "719", "name": "Bàu Bàng", "unit_type": "Huyện"},
        {"code": "720", "name": "Dầu Tiếng", "unit_type": "Huyện"},
        {"code": "721", "name": "Bến Cát", "unit_type": "Thành phố"},
        {"code": "722", "name": "Phú Giáo", "unit_type": "Huyện"},
        {"code": "723", "name": "Tân Uyên", "unit_type": "Thành phố"},
        {"code": "724", "name": "Dĩ An", "unit_type": "Thành phố"},
        {"code": "725", "name": "Thuận An", "unit_type": "Thành phố"},
        {"code": "726", "name": "Bắc Tân Uyên", "unit_type": "Huyện"},
    ],
    "75": [  # Đồng Nai
        {"code": "731", "name": "Biên Hòa", "unit_type": "Thành phố"},
        {"code": "732", "name": "Long Khánh", "unit_type": "Thành phố"},
        {"code": "734", "name": "Tân Phú", "unit_type": "Huyện"},
        {"code": "735", "name": "Vĩnh Cửu", "unit_type": "Huyện"},
        {"code": "736", "name": "Định Quán", "unit_type": "Huyện"},
        {"code": "737", "name": "Trảng Bom", "unit_type": "Huyện"},
        {"code": "738", "name": "Thống Nhất", "unit_type": "Huyện"},
        {"code": "739", "name": "Cẩm Mỹ", "unit_type": "Huyện"},
        {"code": "740", "name": "Long Thành", "unit_type": "Huyện"},
        {"code": "741", "name": "Xuân Lộc", "unit_type": "Huyện"},
        {"code": "742", "name": "Nhơn Trạch", "unit_type": "Huyện"},
    ],
    "77": [  # Bà Rịa - Vũng Tàu
        {"code": "747", "name": "Vũng Tàu", "unit_type": "Thành phố"},
        {"code": "748", "name": "Bà Rịa", "unit_type": "Thành phố"},
        {"code": "750", "name": "Châu Đức", "unit_type": "Huyện"},
        {"code": "751", "name": "Xuyên Mộc", "unit_type": "Huyện"},
        {"code": "752", "name": "Long Điền", "unit_type": "Huyện"},
        {"code": "753", "name": "Đất Đỏ", "unit_type": "Huyện"},
        {"code": "754", "name": "Phú Mỹ", "unit_type": "Thị xã"},
        {"code": "755", "name": "Côn Đảo", "unit_type": "Huyện"},
    ],
    "66": [  # Đắk Lắk
        {"code": "643", "name": "Buôn Ma Thuột", "unit_type": "Thành phố"},
        {"code": "644", "name": "Buôn Hồ", "unit_type": "Thị xã"},
        {"code": "645", "name": "Ea H'leo", "unit_type": "Huyện"},
        {"code": "646", "name": "Ea Súp", "unit_type": "Huyện"},
        {"code": "647", "name": "Buôn Đôn", "unit_type": "Huyện"},
        {"code": "648", "name": "Cư M'gar", "unit_type": "Huyện"},
        {"code": "649", "name": "Krông Búk", "unit_type": "Huyện"},
        {"code": "650", "name": "Krông Năng", "unit_type": "Huyện"},
        {"code": "651", "name": "Ea Kar", "unit_type": "Huyện"},
        {"code": "652", "name": "M'Đrắk", "unit_type": "Huyện"},
        {"code": "653", "name": "Krông Bông", "unit_type": "Huyện"},
        {"code": "654", "name": "Krông Pắc", "unit_type": "Huyện"},
        {"code": "655", "name": "Krông A Na", "unit_type": "Huyện"},
        {"code": "656", "name": "Lắk", "unit_type": "Huyện"},
        {"code": "657", "name": "Cư Kuin", "unit_type": "Huyện"},
    ],
    "46": [  # Thừa Thiên Huế
        {"code": "474", "name": "Huế", "unit_type": "Thành phố"},
        {"code": "476", "name": "Phong Điền", "unit_type": "Huyện"},
        {"code": "477", "name": "Quảng Điền", "unit_type": "Huyện"},
        {"code": "478", "name": "Phú Vang", "unit_type": "Huyện"},
        {"code": "479", "name": "Hương Thủy", "unit_type": "Thị xã"},
        {"code": "480", "name": "Hương Trà", "unit_type": "Thị xã"},
        {"code": "481", "name": "A Lưới", "unit_type": "Huyện"},
        {"code": "482", "name": "Phú Lộc", "unit_type": "Huyện"},
        {"code": "483", "name": "Nam Đông", "unit_type": "Huyện"},
    ],
}

# Key Wards Sample Data for key districts
GSO_WARDS_DATA: Dict[str, List[Dict[str, Any]]] = {
    # Quận 1 (TP. HCM)
    "760": [
        {"code": "26734", "name": "Tân Định", "unit_type": "Phường"},
        {"code": "26737", "name": "Đa Kao", "unit_type": "Phường"},
        {"code": "26740", "name": "Bến Nghé", "unit_type": "Phường"},
        {"code": "26743", "name": "Bến Thành", "unit_type": "Phường"},
        {"code": "26746", "name": "Nguyễn Thái Bình", "unit_type": "Phường"},
        {"code": "26749", "name": "Phạm Ngũ Lão", "unit_type": "Phường"},
        {"code": "26752", "name": "Cầu Ông Lãnh", "unit_type": "Phường"},
        {"code": "26755", "name": "Cô Giang", "unit_type": "Phường"},
        {"code": "26758", "name": "Cầu Kho", "unit_type": "Phường"},
        {"code": "26761", "name": "Nguyễn Cư Trinh", "unit_type": "Phường"},
    ],
    # Tân Bình (TP. HCM)
    "766": [
        {"code": "27088", "name": "Phường 1", "unit_type": "Phường"},
        {"code": "27091", "name": "Phường 2", "unit_type": "Phường"},
        {"code": "27094", "name": "Phường 3", "unit_type": "Phường"},
        {"code": "27097", "name": "Phường 4", "unit_type": "Phường"},
        {"code": "27100", "name": "Phường 5", "unit_type": "Phường"},
        {"code": "27103", "name": "Phường 6", "unit_type": "Phường"},
        {"code": "27106", "name": "Phường 7", "unit_type": "Phường"},
        {"code": "27109", "name": "Phường 8", "unit_type": "Phường"},
        {"code": "27112", "name": "Phường 9", "unit_type": "Phường"},
        {"code": "27115", "name": "Phường 10", "unit_type": "Phường"},
        {"code": "27118", "name": "Phường 11", "unit_type": "Phường"},
        {"code": "27121", "name": "Phường 12", "unit_type": "Phường"},
        {"code": "27124", "name": "Phường 13", "unit_type": "Phường"},
        {"code": "27127", "name": "Phường 14", "unit_type": "Phường"},
        {"code": "27130", "name": "Phường 15", "unit_type": "Phường"},
    ],
    # Ba Đình (Hà Nội)
    "001": [
        {"code": "00001", "name": "Phúc Xá", "unit_type": "Phường"},
        {"code": "00004", "name": "Trúc Bạch", "unit_type": "Phường"},
        {"code": "00006", "name": "Vĩnh Phúc", "unit_type": "Phường"},
        {"code": "00007", "name": "Cống Vị", "unit_type": "Phường"},
        {"code": "00008", "name": "Liễu Giai", "unit_type": "Phường"},
        {"code": "00010", "name": "Nguyễn Trung Trực", "unit_type": "Phường"},
        {"code": "00013", "name": "Quán Thánh", "unit_type": "Phường"},
        {"code": "00016", "name": "Ngọc Hà", "unit_type": "Phường"},
        {"code": "00019", "name": "Điện Biên", "unit_type": "Phường"},
        {"code": "00022", "name": "Đội Cấn", "unit_type": "Phường"},
        {"code": "00025", "name": "Ngọc Khánh", "unit_type": "Phường"},
        {"code": "00028", "name": "Kim Mã", "unit_type": "Phường"},
        {"code": "00031", "name": "Giảng Võ", "unit_type": "Phường"},
        {"code": "00034", "name": "Thành Công", "unit_type": "Phường"},
    ],
    # Hoàng Mai (Hà Nội)
    "008": [
        {"code": "00307", "name": "Thanh Trì", "unit_type": "Phường"},
        {"code": "00310", "name": "Vĩnh Hưng", "unit_type": "Phường"},
        {"code": "00313", "name": "Định Công", "unit_type": "Phường"},
        {"code": "00316", "name": "Mai Động", "unit_type": "Phường"},
        {"code": "00319", "name": "Tương Mai", "unit_type": "Phường"},
        {"code": "00322", "name": "Đại Kim", "unit_type": "Phường"},
        {"code": "00325", "name": "Tân Mai", "unit_type": "Phường"},
        {"code": "00328", "name": "Hoàng Văn Thụ", "unit_type": "Phường"},
        {"code": "00331", "name": "Giáp Bát", "unit_type": "Phường"},
        {"code": "00334", "name": "Lĩnh Nam", "unit_type": "Phường"},
        {"code": "00337", "name": "Thịnh Liệt", "unit_type": "Phường"},
        {"code": "00340", "name": "Trần Phú", "unit_type": "Phường"},
        {"code": "00343", "name": "Hoàng Liệt", "unit_type": "Phường"},
        {"code": "00346", "name": "Yên Sở", "unit_type": "Phường"},
    ],
    # Châu Thành (Đồng Tháp)
    "877": [
        {"code": "30229", "name": "Cái Tàu Hạ", "unit_type": "Thị trấn"},
        {"code": "30232", "name": "Tân Phú Trung", "unit_type": "Xã"},
        {"code": "30235", "name": "Tân Nhuận Đông", "unit_type": "Xã"},
        {"code": "30238", "name": "Tân Phú", "unit_type": "Xã"},
        {"code": "30241", "name": "Tân Bình", "unit_type": "Xã"},
        {"code": "30244", "name": "Hòa Tân", "unit_type": "Xã"},
        {"code": "30247", "name": "Phú Hựu", "unit_type": "Xã"},
        {"code": "30250", "name": "Phú Long", "unit_type": "Xã"},
        {"code": "30253", "name": "An Nhơn", "unit_type": "Xã"},
        {"code": "30256", "name": "An Phú Thuận", "unit_type": "Xã"},
        {"code": "30259", "name": "An Khánh", "unit_type": "Xã"},
        {"code": "30262", "name": "An Hiệp", "unit_type": "Xã"},
    ],
    # Bình Tân (Vĩnh Long)
    "863": [
        {"code": "29917", "name": "Tân Quới", "unit_type": "Thị trấn"},
        {"code": "29920", "name": "Tân Lược", "unit_type": "Xã"},
        {"code": "29923", "name": "Tân An Thạnh", "unit_type": "Xã"},
        {"code": "29926", "name": "Tân Hưng", "unit_type": "Xã"},
        {"code": "29929", "name": "Tân Thành", "unit_type": "Xã"},
        {"code": "29932", "name": "Tân Bình", "unit_type": "Xã"},
        {"code": "29935", "name": "Thành Lợi", "unit_type": "Xã"},
        {"code": "29938", "name": "Thành Trung", "unit_type": "Xã"},
        {"code": "29941", "name": "Thành Đông", "unit_type": "Xã"},
        {"code": "29944", "name": "Nguyễn Văn Thảnh", "unit_type": "Xã"},
        {"code": "29947", "name": "Mỹ Thuận", "unit_type": "Xã"},
    ],
    # Ea H'leo (Đắk Lắk)
    "645": [
        {"code": "24310", "name": "Ea Drăng", "unit_type": "Thị trấn"},
        {"code": "24313", "name": "Ea H'leo", "unit_type": "Xã"},
        {"code": "24316", "name": "Ea Sol", "unit_type": "Xã"},
        {"code": "24319", "name": "Ea Răl", "unit_type": "Xã"},
        {"code": "24322", "name": "Ea Wy", "unit_type": "Xã"},
        {"code": "24325", "name": "Cư Mốt", "unit_type": "Xã"},
        {"code": "24328", "name": "Ea Hiao", "unit_type": "Xã"},
        {"code": "24331", "name": "Ea Tir", "unit_type": "Xã"},
        {"code": "24334", "name": "Dliê Yang", "unit_type": "Xã"},
        {"code": "24337", "name": "Ea Khal", "unit_type": "Xã"},
        {"code": "24340", "name": "Cư A Mung", "unit_type": "Xã"},
        {"code": "24343", "name": "Ea Nam", "unit_type": "Xã"},
    ],
}


# ===========================================================================
# 4. GSO Administrative Database Manager
# ===========================================================================

class GsoDatabase:
    """
    Manages in-memory hierarchical GSO administrative catalog.
    Provides O(1) indexed lookups by unaccented keys, GSO code lookups, and candidate pools.
    """

    def __init__(self, json_path: Optional[Union[str, Path]] = None) -> None:
        self.provinces: Dict[str, GsoAdminUnit] = {}  # code -> GsoAdminUnit
        self.province_lookup: Dict[str, str] = {}     # normalized_key -> code
        self.district_lookup_all: Dict[str, List[Tuple[str, str]]] = {}  # norm_key -> [(prov_code, dist_code)]

        self._initialize_default_data()
        if json_path:
            self.load_from_json(json_path)

    def _normalize_key(self, text: str) -> str:
        if not text:
            return ""
        text_clean = UnicodeNormalizer.normalize(text)
        unaccented = remove_vietnamese_accents(text_clean).lower()
        unaccented = re.sub(r'[^a-z0-9\s]', ' ', unaccented)
        return re.sub(r'\s+', ' ', unaccented).strip()

    def _strip_admin_prefix(self, normalized_text: str) -> str:
        """
        Strips common administrative prefixes to obtain core place name.
        """
        tokens = normalized_text.split()
        if not tokens:
            return ""
        # Check 2-word prefixes first
        if len(tokens) >= 3 and f"{tokens[0]} {tokens[1]}" in ("thanh pho", "thi xa", "thi tran", "khu pho", "so nha"):
            return " ".join(tokens[2:])
        if tokens[0] in ("tinh", "tp", "quan", "q", "huyen", "h", "tx", "phuong", "p", "f", "xa", "x", "tt", "ap", "thon", "to", "khom", "kp", "ban", "lang", "buon", "soc"):
            return " ".join(tokens[1:])
        return normalized_text

    def _initialize_default_data(self) -> None:
        # 1. Initialize Provinces
        for p in GSO_PROVINCES_CATALOG:
            code = p["code"]
            name = p["name"]
            unit_type = p["unit_type"]
            norm_key = self._normalize_key(name)
            node = GsoAdminUnit(code=code, name=name, unit_type=unit_type, normalized_name=norm_key)
            self.provinces[code] = node
            self.province_lookup[norm_key] = code

            # Register stripped versions and aliases
            stripped = self._strip_admin_prefix(norm_key)
            if stripped and stripped != norm_key:
                self.province_lookup[stripped] = code

        # Register province aliases
        for alias, prov_name in PROVINCE_ALIASES.items():
            norm_alias = self._normalize_key(alias)
            # Find matching province code
            for c, node in self.provinces.items():
                if node.name.lower() == prov_name.lower() or self._normalize_key(node.name) == self._normalize_key(prov_name):
                    self.province_lookup[norm_alias] = c
                    break

        # 2. Initialize Districts
        for prov_code, dist_list in GSO_DISTRICTS_DATA.items():
            if prov_code not in self.provinces:
                continue
            prov_node = self.provinces[prov_code]
            for d in dist_list:
                d_code = d["code"]
                d_name = d["name"]
                d_type = d["unit_type"]
                d_norm = self._normalize_key(d_name)
                d_node = GsoAdminUnit(code=d_code, name=d_name, unit_type=d_type, normalized_name=d_norm)
                prov_node.sub_units[d_code] = d_node

                # Global lookup index
                self.district_lookup_all.setdefault(d_norm, []).append((prov_code, d_code))
                stripped_d = self._strip_admin_prefix(d_norm)
                if stripped_d and stripped_d != d_norm:
                    self.district_lookup_all.setdefault(stripped_d, []).append((prov_code, d_code))

        # 3. Initialize Wards
        for dist_code, ward_list in GSO_WARDS_DATA.items():
            # Find district in provinces
            for prov_code, prov_node in self.provinces.items():
                if dist_code in prov_node.sub_units:
                    dist_node = prov_node.sub_units[dist_code]
                    for w in ward_list:
                        w_code = w["code"]
                        w_name = w["name"]
                        w_type = w["unit_type"]
                        w_norm = self._normalize_key(w_name)
                        w_node = GsoAdminUnit(code=w_code, name=w_name, unit_type=w_type, normalized_name=w_norm)
                        dist_node.sub_units[w_code] = w_node

    def load_from_json(self, json_path: Union[str, Path]) -> None:
        """
        Loads hierarchical GSO dataset from an external JSON file.
        Accepts tree structure: list of provinces with sub-districts and sub-wards.
        """
        p = Path(json_path)
        if not p.exists():
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    p_code = str(item.get("code", ""))
                    p_name = item.get("name", "")
                    p_type = item.get("unit_type", "Tỉnh")
                    if not p_code or not p_name:
                        continue
                    p_norm = self._normalize_key(p_name)
                    p_node = self.provinces.get(p_code) or GsoAdminUnit(code=p_code, name=p_name, unit_type=p_type, normalized_name=p_norm)
                    self.provinces[p_code] = p_node
                    self.province_lookup[p_norm] = p_code

                    districts = item.get("districts") or item.get("sub_units") or []
                    for d_item in districts:
                        d_code = str(d_item.get("code", ""))
                        d_name = d_item.get("name", "")
                        d_type = d_item.get("unit_type", "Quận")
                        if not d_code or not d_name:
                            continue
                        d_norm = self._normalize_key(d_name)
                        d_node = p_node.sub_units.get(d_code) or GsoAdminUnit(code=d_code, name=d_name, unit_type=d_type, normalized_name=d_norm)
                        p_node.sub_units[d_code] = d_node
                        self.district_lookup_all.setdefault(d_norm, []).append((p_code, d_code))

                        wards = d_item.get("wards") or d_item.get("sub_units") or []
                        for w_item in wards:
                            w_code = str(w_item.get("code", ""))
                            w_name = w_item.get("name", "")
                            w_type = w_item.get("unit_type", "Xã")
                            if not w_code or not w_name:
                                continue
                            w_norm = self._normalize_key(w_name)
                            w_node = GsoAdminUnit(code=w_code, name=w_name, unit_type=w_type, normalized_name=w_norm)
                            d_node.sub_units[w_code] = w_node
        except Exception:
            pass


# Global Singleton Database Instance
_GLOBAL_GSO_DB = GsoDatabase()


# ===========================================================================
# 5. Vietnamese Address Parser Engine
# ===========================================================================

class VietnameseAddressParser:
    """
    High-performance right-to-left hierarchical Vietnamese address parser.
    """

    def __init__(
        self,
        gso_db: Optional[GsoDatabase] = None,
        similarity_threshold: float = 0.75,
        exact_numeric_match: bool = True
    ) -> None:
        self.db = gso_db or _GLOBAL_GSO_DB
        self.similarity_threshold = similarity_threshold
        self.exact_numeric_match = exact_numeric_match

    # -----------------------------------------------------------------------
    # Helper & Preprocessing Methods
    # -----------------------------------------------------------------------

    def preprocess_raw_address(self, text: str) -> str:
        """
        Cleans and normalizes raw address string:
        - Mojibake repairing
        - NFC Unicode normalization
        - OCR diacritic fixing
        - Standardizing punctuation delimiters
        """
        if not text:
            return ""
        # 1. Unicode NFC & Mojibake repair
        cleaned = UnicodeNormalizer.normalize(text) or ""
        cleaned = MojibakeFixer.fix(cleaned) or cleaned

        # 2. Correct OCR diacritic confusions (e.g. ä -> â, ö -> ô)
        cleaned = VietnameseTextCorrector.correct_ocr_glyphs(cleaned)
        cleaned = VietnameseTextCorrector.format_spacing_and_punctuation(cleaned)

        # 3. Normalize separator characters
        cleaned = cleaned.replace(";", ",").replace("|", ",").replace(" - ", ", ")
        # Clean multiple spaces and dots used as separators
        cleaned = re.sub(r'[\t\r\n]+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def _normalize_key(self, text: str) -> str:
        if not text:
            return ""
        text_clean = UnicodeNormalizer.normalize(text)
        unaccented = remove_vietnamese_accents(text_clean).lower()
        unaccented = re.sub(r'[^a-z0-9\s]', ' ', unaccented)
        return re.sub(r'\s+', ' ', unaccented).strip()

    def _strip_prefix(self, norm_text: str) -> Tuple[str, str]:
        """
        Separates administrative prefix and core name from a normalized string.
        Returns: (prefix_label, core_name)
        """
        tokens = norm_text.split()
        if not tokens:
            return "", ""

        # 2-word prefixes
        if len(tokens) >= 3:
            p2 = f"{tokens[0]} {tokens[1]}"
            if p2 in ADMINISTRATIVE_PREFIX_MAP:
                return ADMINISTRATIVE_PREFIX_MAP[p2], " ".join(tokens[2:])

        # 1-word prefixes
        if tokens[0] in ADMINISTRATIVE_PREFIX_MAP:
            return ADMINISTRATIVE_PREFIX_MAP[tokens[0]], " ".join(tokens[1:])

        return "", norm_text

    def _compute_similarity(self, query: str, target: str) -> float:
        """
        Computes composite fuzzy similarity between query and target (0.0 - 1.0).
        """
        if not query or not target:
            return 0.0
        if query == target:
            return 1.0

        if HAS_RAPIDFUZZ:
            r_ratio = fuzz.ratio(query, target) / 100.0
            r_sort = fuzz.token_sort_ratio(query, target) / 100.0
            r_partial = fuzz.partial_ratio(query, target) / 100.0
            # Weighted blend prioritizing token alignment
            score = (r_sort * 0.5) + (r_ratio * 0.3) + (r_partial * 0.2)
            return round(score, 4)
        else:
            from difflib import SequenceMatcher
            return round(SequenceMatcher(None, query, target).ratio(), 4)

    def _is_numeric_unit(self, norm_str: str) -> Optional[int]:
        """
        Detects if an administrative unit name is purely numeric (e.g. '1', '01', 'quan 1', 'p 10').
        """
        # Match 'quan 1', 'q1', 'p10', '1', '12'
        m = re.search(r'\b(?:quan|q|phuong|p|f|to|ap|khom|kp)?\s*0*(\d+)\b', norm_str)
        if m:
            return int(m.group(1))
        return None

    # -----------------------------------------------------------------------
    # Step-by-Step Hierarchical Matchers
    # -----------------------------------------------------------------------

    def _match_province(self, segment: str) -> Optional[Tuple[GsoAdminUnit, float, str]]:
        """
        Fuzzy matches segment against 63 GSO provinces.
        Returns: (GsoAdminUnit, confidence, matched_key) or None
        """
        norm_seg = self._normalize_key(segment)
        if not norm_seg:
            return None

        # 1. Exact alias & lookup check
        if norm_seg in self.db.province_lookup:
            p_code = self.db.province_lookup[norm_seg]
            return self.db.provinces[p_code], 1.0, norm_seg

        # Core name without prefix
        _, core_name = self._strip_prefix(norm_seg)
        if core_name in self.db.province_lookup:
            p_code = self.db.province_lookup[core_name]
            return self.db.provinces[p_code], 0.98, core_name

        # 2. Fuzzy match against all province candidates
        best_node = None
        best_score = 0.0
        best_key = ""

        for key, code in self.db.province_lookup.items():
            # Special check for short names (e.g. 'hue', 'ha nam', 'ha noi')
            threshold = 0.88 if len(key) <= 5 else self.similarity_threshold
            score = self._compute_similarity(core_name or norm_seg, key)

            if score > best_score and score >= threshold:
                best_score = score
                best_node = self.db.provinces[code]
                best_key = key

        if best_node:
            return best_node, best_score, best_key
        return None

    def _match_district(
        self,
        segment: str,
        province_node: Optional[GsoAdminUnit] = None
    ) -> Optional[Tuple[GsoAdminUnit, float, str]]:
        """
        Matches segment against districts within the specified province (or across all districts if none).
        """
        norm_seg = self._normalize_key(segment)
        if not norm_seg:
            return None

        prefix, core_name = self._strip_prefix(norm_seg)
        query_numeric = self._is_numeric_unit(norm_seg)

        # Build candidate pool
        candidates: List[GsoAdminUnit] = []
        if province_node and province_node.sub_units:
            candidates = list(province_node.sub_units.values())
        else:
            # Fallback across all loaded districts
            for p_node in self.db.provinces.values():
                candidates.extend(p_node.sub_units.values())

        if not candidates:
            # If no district dataset exists in DB, construct a dynamic normalized unit from segment
            if core_name or norm_seg:
                raw_name = (core_name or norm_seg).title()
                unit_label = prefix or "Huyện"
                dynamic_node = GsoAdminUnit(
                    code="",
                    name=f"{unit_label} {raw_name}".strip() if prefix else raw_name,
                    unit_type=unit_label,
                    normalized_name=norm_seg
                )
                return dynamic_node, 0.80, norm_seg
            return None

        # 1. Exact numeric check for numbered districts (e.g. Quận 1, Quận 10)
        if query_numeric is not None and self.exact_numeric_match:
            for dist in candidates:
                cand_num = self._is_numeric_unit(dist.normalized_name)
                if cand_num is not None and cand_num == query_numeric:
                    return dist, 1.0, dist.normalized_name

        # 2. Exact match check
        for dist in candidates:
            dist_core = self._strip_prefix(dist.normalized_name)[1]
            if core_name == dist_core or norm_seg == dist.normalized_name:
                return dist, 0.99, dist.normalized_name

        # 3. Fuzzy match check
        best_node = None
        best_score = 0.0
        best_key = ""

        for dist in candidates:
            dist_core = self._strip_prefix(dist.normalized_name)[1]
            # Avoid comparing numbers fuzzy to names
            if query_numeric is not None and not any(char.isdigit() for char in dist_core):
                continue

            threshold = 0.85 if len(dist_core) <= 4 else self.similarity_threshold
            score = self._compute_similarity(core_name or norm_seg, dist_core)

            if score > best_score and score >= threshold:
                best_score = score
                best_node = dist
                best_key = dist.normalized_name

        if best_node:
            return best_node, best_score, best_key

        # Dynamic fallback if not found in catalog but prefix indicates district
        if prefix in ("Quận", "Huyện", "Thị xã", "TP.") and core_name:
            dynamic_node = GsoAdminUnit(
                code="",
                name=f"{prefix} {core_name.title()}",
                unit_type=prefix,
                normalized_name=norm_seg
            )
            return dynamic_node, 0.75, norm_seg

        return None

    def _match_ward(
        self,
        segment: str,
        district_node: Optional[GsoAdminUnit] = None,
        province_node: Optional[GsoAdminUnit] = None
    ) -> Optional[Tuple[GsoAdminUnit, float, str]]:
        """
        Matches segment against wards within the specified district.
        """
        norm_seg = self._normalize_key(segment)
        if not norm_seg:
            return None

        prefix, core_name = self._strip_prefix(norm_seg)
        query_numeric = self._is_numeric_unit(norm_seg)

        # Candidate pool
        candidates: List[GsoAdminUnit] = []
        if district_node and district_node.sub_units:
            candidates = list(district_node.sub_units.values())
        elif province_node:
            for d_node in province_node.sub_units.values():
                candidates.extend(d_node.sub_units.values())

        if not candidates:
            # Dynamic creation from segment if prefix or valid text exists
            if core_name or norm_seg:
                raw_name = (core_name or norm_seg).title()
                unit_label = prefix or "Xã"
                dynamic_node = GsoAdminUnit(
                    code="",
                    name=f"{unit_label} {raw_name}".strip() if prefix else raw_name,
                    unit_type=unit_label,
                    normalized_name=norm_seg
                )
                return dynamic_node, 0.80, norm_seg
            return None

        # 1. Exact numeric check for numbered wards (e.g. Phường 1, Phường 15)
        if query_numeric is not None and self.exact_numeric_match:
            for ward in candidates:
                cand_num = self._is_numeric_unit(ward.normalized_name)
                if cand_num is not None and cand_num == query_numeric:
                    return ward, 1.0, ward.normalized_name

        # 2. Exact match check
        for ward in candidates:
            ward_core = self._strip_prefix(ward.normalized_name)[1]
            if core_name == ward_core or norm_seg == ward.normalized_name:
                return ward, 0.99, ward.normalized_name

        # 3. Fuzzy match check
        best_node = None
        best_score = 0.0
        best_key = ""

        for ward in candidates:
            ward_core = self._strip_prefix(ward.normalized_name)[1]
            if query_numeric is not None and not any(char.isdigit() for char in ward_core):
                continue

            threshold = 0.85 if len(ward_core) <= 4 else self.similarity_threshold
            score = self._compute_similarity(core_name or norm_seg, ward_core)

            if score > best_score and score >= threshold:
                best_score = score
                best_node = ward
                best_key = ward.normalized_name

        if best_node:
            return best_node, best_score, best_key

        # Dynamic fallback
        if prefix in ("Phường", "Xã", "Thị trấn") and core_name:
            dynamic_node = GsoAdminUnit(
                code="",
                name=f"{prefix} {core_name.title()}",
                unit_type=prefix,
                normalized_name=norm_seg
            )
            return dynamic_node, 0.75, norm_seg

        return None

    def _clean_street_address(self, street_segment: str) -> str:
        """
        Cleans and formats the remaining Level 4 street/hamlet string.
        """
        if not street_segment:
            return ""
        # Clean leading/trailing commas, dots, hyphens
        cleaned = re.sub(r'^[\s,.\-_/]+|[\s,.\-_/]+$', '', street_segment)

        # Standardize known prefixes in street (e.g. 'ap Tay' -> 'Ấp Tây', 'to 5' -> 'Tổ 5')
        tokens = cleaned.split()
        normalized_tokens: List[str] = []
        i = 0
        n = len(tokens)

        while i < n:
            tok = tokens[i]
            tok_lower = tok.lower()
            norm_tok = self._normalize_key(tok)

            if norm_tok in ADMINISTRATIVE_PREFIX_MAP:
                std_prefix = ADMINISTRATIVE_PREFIX_MAP[norm_tok]
                if i + 1 < n and re.match(r'^\d+[a-zA-Z]?$', tokens[i + 1]):
                    normalized_tokens.append(f"{std_prefix} {tokens[i + 1]}")
                    i += 2
                    continue
                normalized_tokens.append(std_prefix)
                i += 1
            else:
                # Capitalize first letter of proper words
                normalized_tokens.append(tok)
                i += 1

        res = " ".join(normalized_tokens)
        return re.sub(r'\s+', ' ', res).strip()

    # -----------------------------------------------------------------------
    # Main Parsing Workflow (Right-to-Left)
    # -----------------------------------------------------------------------

    def parse(self, raw_address: str) -> ParsedAddress:
        """
        Executes right-to-left hierarchical parsing on a raw Vietnamese address string.
        """
        if not raw_address or not raw_address.strip():
            return ParsedAddress(
                original_address=raw_address or "",
                full_normalized_address="",
                confidence=0.0
            )

        preprocessed = self.preprocess_raw_address(raw_address)

        # Split into segments by comma or slash
        if "," in preprocessed:
            raw_segments = [s.strip() for s in preprocessed.split(",") if s.strip()]
        elif "/" in preprocessed and not re.search(r'\d+/\d+', preprocessed):
            raw_segments = [s.strip() for s in preprocessed.split("/") if s.strip()]
        else:
            # Non-delimited space-separated string: tokenized segments
            raw_segments = [preprocessed.strip()]

        segments = list(raw_segments)
        confidences: List[float] = []

        province_unit: Optional[GsoAdminUnit] = None
        district_unit: Optional[GsoAdminUnit] = None
        ward_unit: Optional[GsoAdminUnit] = None

        # -------------------------------------------------------------------
        # Step 1: Match Province from Rightmost Segment
        # -------------------------------------------------------------------
        if segments:
            # Check last segment
            p_match = self._match_province(segments[-1])
            if p_match:
                province_unit, p_conf, _ = p_match
                confidences.append(p_conf)
                segments.pop()
            elif len(segments) >= 2:
                # Check second to last in case trailing country or noise exists
                p_match2 = self._match_province(segments[-2])
                if p_match2:
                    province_unit, p_conf, _ = p_match2
                    confidences.append(p_conf)
                    segments.pop(-2)
            elif len(segments) == 1:
                # Sliding window search from right on single space-separated string
                words = segments[0].split()
                for window_len in range(min(5, len(words)), 0, -1):
                    cand_str = " ".join(words[-window_len:])
                    p_match = self._match_province(cand_str)
                    if p_match:
                        province_unit, p_conf, _ = p_match
                        confidences.append(p_conf)
                        segments = [" ".join(words[:-window_len])]
                        break

        # -------------------------------------------------------------------
        # Step 2: Match District from Rightmost Remaining Segment
        # -------------------------------------------------------------------
        if segments:
            if len(segments) > 1 or (len(segments) == 1 and not (province_unit and len(segments[0].split()) > 4)):
                d_match = self._match_district(segments[-1], province_node=province_unit)
                if d_match:
                    district_unit, d_conf, _ = d_match
                    confidences.append(d_conf)
                    segments.pop()
            else:
                # Single remaining long text: sliding window for district
                words = segments[0].split()
                for window_len in range(min(5, len(words)), 0, -1):
                    cand_str = " ".join(words[-window_len:])
                    d_match = self._match_district(cand_str, province_node=province_unit)
                    if d_match:
                        district_unit, d_conf, _ = d_match
                        confidences.append(d_conf)
                        segments = [" ".join(words[:-window_len])]
                        break

        # -------------------------------------------------------------------
        # Step 3: Match Ward from Rightmost Remaining Segment
        # -------------------------------------------------------------------
        if segments:
            if len(segments) > 1 or (len(segments) == 1 and not len(segments[0].split()) > 4):
                w_match = self._match_ward(segments[-1], district_node=district_unit, province_node=province_unit)
                if w_match:
                    ward_unit, w_conf, _ = w_match
                    confidences.append(w_conf)
                    segments.pop()
            else:
                # Single remaining string: sliding window for ward
                words = segments[0].split()
                for window_len in range(min(5, len(words)), 0, -1):
                    cand_str = " ".join(words[-window_len:])
                    w_match = self._match_ward(cand_str, district_node=district_unit, province_node=province_unit)
                    if w_match:
                        ward_unit, w_conf, _ = w_match
                        confidences.append(w_conf)
                        segments = [" ".join(words[:-window_len])]
                        break

        # -------------------------------------------------------------------
        # Step 4: Clean and Format Street / Detail Level
        # -------------------------------------------------------------------
        street_raw = ", ".join(segments).strip()
        street_clean = self._clean_street_address(street_raw)

        # -------------------------------------------------------------------
        # Step 5: Format Standardized Names & Full Address
        # -------------------------------------------------------------------
        prov_name = province_unit.name if province_unit else None
        dist_name = district_unit.name if district_unit else None
        ward_name = ward_unit.name if ward_unit else None

        # Build clean comma-separated full address
        addr_parts: List[str] = []
        if street_clean:
            addr_parts.append(street_clean)
        if ward_name:
            addr_parts.append(ward_name)
        if dist_name:
            addr_parts.append(dist_name)
        if prov_name:
            addr_parts.append(prov_name)

        full_addr = ", ".join(addr_parts) if addr_parts else preprocessed

        # Overall confidence calculation
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.50

        return ParsedAddress(
            original_address=raw_address,
            province=prov_name,
            district=dist_name,
            ward=ward_name,
            street=street_clean or None,
            province_code=province_unit.code if (province_unit and province_unit.code) else None,
            district_code=district_unit.code if (district_unit and district_unit.code) else None,
            ward_code=ward_unit.code if (ward_unit and ward_unit.code) else None,
            full_normalized_address=full_addr,
            confidence=avg_confidence
        )


# ===========================================================================
# 6. Global Helper Function
# ===========================================================================

def parse_address(raw_address: str, similarity_threshold: float = 0.75) -> ParsedAddress:
    """
    Convenient standalone helper function to parse and standardize a Vietnamese address.

    Args:
        raw_address (str): Raw input address string from OCR or user input.
        similarity_threshold (float): Minimum fuzzy match threshold (default 0.75).

    Returns:
        ParsedAddress: 4-level parsed and standardized address object.
    """
    parser = VietnameseAddressParser(similarity_threshold=similarity_threshold)
    return parser.parse(raw_address)


# ===========================================================================
# 7. Verification & Practical Real-World Test Cases
# ===========================================================================

if __name__ == "__main__":
    test_cases = [
        # 1. Full standard address with diacritics
        "Số 123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP. Hồ Chí Minh",

        # 2. Unaccented, abbreviated with dot notation
        "123 nguyen hue, p. ben nghe, q1, hcm",

        # 3. Mekong Delta rural address with hamlet, commune, district, province
        "Ap Tay, Xa Tan Binh, Huyen Chau Thanh, Tinh Dong Thap",

        # 4. Mojibake OCR error & strange diacritics
        "Äá»“ng ThÃ¡p, Huyện Chäu Thành, Xã Tân Bình, Ấp Tây",

        # 5. Non-delimited space-separated string with numbered ward
        "To 5 Phuong Hoang Van Thu Quan Hoang Mai Ha Noi",

        # 6. Highlands & Special administrative names
        "Thị trấn Ea Drăng, Huyện Ea H'leo, Tỉnh Đắk Lắk",

        # 7. Central Vietnam & Short place names
        "Phú Hội, Thành phố Huế, Thừa Thiên Huế",

        # 8. Numeric ward & district in TP. HCM
        "Số 45/2 Tân Sơn, Phường 15, Quận Tân Bình, TP. Hồ Chí Minh"
    ]

    print("=" * 80)
    print("VIETNAMESE ADDRESS PARSER - TEST SUITE EXECUTION")
    print("=" * 80)

    for idx, raw in enumerate(test_cases, 1):
        parsed = parse_address(raw)
        print(f"\n[Case {idx}] Input: {raw}")
        print(f"  -> Street   (Level 4): {parsed.street}")
        print(f"  -> Ward     (Level 3): {parsed.ward} (Code: {parsed.ward_code})")
        print(f"  -> District (Level 2): {parsed.district} (Code: {parsed.district_code})")
        print(f"  -> Province (Level 1): {parsed.province} (Code: {parsed.province_code})")
        print(f"  -> Full Normalized   : {parsed.full_normalized_address}")
        print(f"  -> Confidence        : {parsed.confidence:.2f}")
    print("\n" + "=" * 80)
