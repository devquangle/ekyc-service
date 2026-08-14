import re
import unicodedata
from typing import Optional, List, Dict, Tuple, Any
from utils.text_normalizer import UnicodeNormalizer
from utils.text_utils import remove_vietnamese_accents

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# ---------------------------------------------------------------------------
# 1. Official Administrative Prefixes (Thôn, Ấp, Tổ, Khóm, Số, Đường...)
# ---------------------------------------------------------------------------
ADMINISTRATIVE_PREFIXES: Dict[str, str] = {
    "ap": "Ấp",
    "ấp": "Ấp",
    "thon": "Thôn",
    "thôn": "Thôn",
    "to": "Tổ",
    "tổ": "Tổ",
    "khom": "Khóm",
    "khóm": "Khóm",
    "ban": "Bản",
    "bản": "Bản",
    "lang": "Làng",
    "làng": "Làng",
    "buon": "Buôn",
    "buôn": "Buôn",
    "soc": "Sóc",
    "sóc": "Sóc",
    "khu pho": "Khu phố",
    "khu phố": "Khu phố",
    "kp": "KP.",
    "kp.": "KP.",
    "khu": "Khu",
    "cum": "Cụm",
    "cụm": "Cụm",
    "xa": "Xã",
    "xã": "Xã",
    "x.": "X.",
    "phuong": "Phường",
    "phường": "Phường",
    "p.": "P.",
    "p": "P.",
    "f.": "P.",
    "f": "P.",
    "thi tran": "Thị trấn",
    "thị trấn": "Thị trấn",
    "tt": "TT.",
    "tt.": "TT.",
    "thi xa": "Thị xã",
    "thị xã": "Thị xã",
    "tx": "TX.",
    "tx.": "TX.",
    "huyen": "Huyện",
    "huyện": "Huyện",
    "h.": "H.",
    "quan": "Quận",
    "quận": "Quận",
    "q.": "Q.",
    "q": "Q.",
    "thanh pho": "Thành phố",
    "thành phố": "Thành phố",
    "tp": "TP.",
    "tp.": "TP.",
    "tinh": "Tỉnh",
    "tỉnh": "Tỉnh",
    "t.": "T.",
    "so": "Số",
    "số": "Số",
    "duong": "Đường",
    "đường": "Đường",
    "d.": "Đ.",
    "đ.": "Đ.",
    "hem": "Hẻm",
    "hẻm": "Hẻm",
    "ngach": "Ngách",
    "ngách": "Ngách",
    "ngõ": "Ngõ",
    "ngo": "Ngõ",
    "tay": "Tây",
    "dong": "Đông",
    "nam": "Nam",
    "bac": "Bắc",
    "trung": "Trung",
    "thuong": "Thượng",
    "ha": "Hạ",
}

# ---------------------------------------------------------------------------
# 2. Complete Gazetteer of 63 Provinces & Municipalities in Vietnam
# ---------------------------------------------------------------------------
PROVINCES_GAZETTEER: Dict[str, str] = {
    "an giang": "An Giang",
    "ba ria vung tau": "Bà Rịa - Vũng Tàu",
    "bac giang": "Bắc Giang",
    "bac kan": "Bắc Kạn",
    "bac lieu": "Bạc Liêu",
    "bac ninh": "Bắc Ninh",
    "ben tre": "Bến Tre",
    "binh dinh": "Bình Định",
    "binh duong": "Bình Dương",
    "binh phuoc": "Bình Phước",
    "binh thuan": "Bình Thuận",
    "ca mau": "Cà Mau",
    "can tho": "Cần Thơ",
    "cao bang": "Cao Bằng",
    "da nang": "Đà Nẵng",
    "dak lak": "Đắk Lắk",
    "dak nong": "Đắk Nông",
    "dien bien": "Điện Biên",
    "dong nai": "Đồng Nai",
    "dong thap": "Đồng Tháp",
    "gia lai": "Gia Lai",
    "ha giang": "Hà Giang",
    "ha nam": "Hà Nam",
    "ha noi": "Hà Nội",
    "ha tinh": "Hà Tĩnh",
    "hai duong": "Hải Dương",
    "hai phong": "Hải Phòng",
    "hau giang": "Hậu Giang",
    "hoa binh": "Hòa Bình",
    "hung yen": "Hưng Yên",
    "khanh hoa": "Khánh Hòa",
    "kien giang": "Kiên Giang",
    "kon tum": "Kon Tum",
    "lai chau": "Lai Châu",
    "lam dong": "Lâm Đồng",
    "lang son": "Lạng Sơn",
    "lao cai": "Lào Cai",
    "long an": "Long An",
    "nam dinh": "Nam Định",
    "nghe an": "Nghệ An",
    "ninh binh": "Ninh Bình",
    "ninh thuan": "Ninh Thuận",
    "phu tho": "Phú Thọ",
    "phu yen": "Phú Yên",
    "quang binh": "Quảng Bình",
    "quang nam": "Quảng Nam",
    "quang ngai": "Quảng Ngãi",
    "quang ninh": "Quảng Ninh",
    "quang tri": "Quảng Trị",
    "soc trang": "Sóc Trăng",
    "son la": "Sơn La",
    "tay ninh": "Tây Ninh",
    "thai binh": "Thái Bình",
    "thai nguyen": "Thái Nguyên",
    "thanh hoa": "Thanh Hóa",
    "thua thien hue": "Thừa Thiên Huế",
    "tien giang": "Tiền Giang",
    "tp ho chi minh": "TP. Hồ Chí Minh",
    "ho chi minh": "TP. Hồ Chí Minh",
    "hcm": "TP. Hồ Chí Minh",
    "tra vinh": "Trà Vinh",
    "tuyen quang": "Tuyên Quang",
    "vinh long": "Vĩnh Long",
    "vinh phuc": "Vĩnh Phúc",
    "yen bai": "Yên Bái",
}

# ---------------------------------------------------------------------------
# 3. Complete Hierarchical Districts by Province (Tất cả Quận/Huyện/Thị Xã)
# ---------------------------------------------------------------------------
DISTRICTS_BY_PROVINCE: Dict[str, Dict[str, str]] = {
    "ha noi": {
        "ba dinh": "Ba Đình", "hoan kiem": "Hoàn Kiếm", "tay ho": "Tây Hồ", "long bien": "Long Biên",
        "cau giay": "Cầu Giấy", "dong da": "Đống Đa", "hai ba trung": "Hai Bà Trưng", "hoang mai": "Hoàng Mai",
        "thanh xuan": "Thanh Xuân", "soc son": "Sóc Sơn", "dong anh": "Đông Anh", "gia lam": "Gia Lâm",
        "nam tu liem": "Nam Từ Liêm", "bac tu liem": "Bắc Từ Liêm", "thanh tri": "Thanh Trì", "me linh": "Mê Linh",
        "ha dong": "Hà Đông", "son tay": "Sơn Tây", "ba vi": "Ba Vì", "phuc tho": "Phúc Thọ",
        "dan phuong": "Đan Phượng", "hoai duc": "Hoài Đức", "quoc oai": "Quốc Oai", "thach that": "Thạch Thất",
        "chuong my": "Chương Mỹ", "thanh oai": "Thanh Oai", "thuong tin": "Thường Tín", "phu xuyen": "Phú Xuyên",
        "ung hoa": "Ứng Hòa", "my duc": "Mỹ Đức"
    },
    "tp ho chi minh": {
        "quan 1": "Quận 1", "quan 2": "Quận 2", "quan 3": "Quận 3", "quan 4": "Quận 4",
        "quan 5": "Quận 5", "quan 6": "Quận 6", "quan 7": "Quận 7", "quan 8": "Quận 8",
        "quan 9": "Quận 9", "quan 10": "Quận 10", "quan 11": "Quận 11", "quan 12": "Quận 12",
        "thu duc": "Thủ Đức", "go vap": "Gò Vấp", "binh thanh": "Bình Thạnh", "tan binh": "Tân Bình",
        "tan phu": "Tân Phú", "phu nhuan": "Phú Nhuận", "binh tan": "Bình Tân", "cu chi": "Củ Chi",
        "hoc mon": "Hóc Môn", "binh chanh": "Bình Chánh", "nha be": "Nhà Bè", "can gio": "Cần Giờ"
    },
    "hai phong": {
        "hong bang": "Hồng Bàng", "ngo quyen": "Ngô Quyền", "le chan": "Lê Chân", "hai an": "Hải An",
        "kien an": "Kiến An", "do son": "Đồ Sơn", "duong kinh": "Dương Kinh", "thuy nguyen": "Thủy Nguyên",
        "an duong": "An Dương", "an lao": "An Lão", "kien thuy": "Kiến Thụy", "tien lang": "Tiên Lãng",
        "vinh bao": "Vĩnh Bảo", "cat hai": "Cát Hải", "bach long vi": "Bạch Long Vĩ"
    },
    "da nang": {
        "hai chau": "Hải Châu", "thanh khe": "Thanh Khê", "son tra": "Sơn Trà", "ngu hanh son": "Ngũ Hành Sơn",
        "lien chieu": "Liên Chiểu", "cam le": "Cẩm Lệ", "hoa vang": "Hòa Vang", "hoang sa": "Hoàng Sa"
    },
    "can tho": {
        "ninh kieu": "Ninh Kiều", "o mon": "Ô Môn", "binh thuy": "Bình Thủy", "cai rang": "Cái Răng",
        "thot not": "Thốt Nốt", "vinh thanh": "Vĩnh Thạnh", "co do": "Cờ Đỏ", "phong dien": "Phong Điền",
        "thoi lai": "Thới Lai"
    },
    "dong thap": {
        "cao lanh": "Cao Lãnh", "sa dec": "Sa Đéc", "hong ngu": "Hồng Ngự", "tan hong": "Tân Hồng",
        "tam nong": "Tam Nông", "thap muoi": "Tháp Mười", "thanh binh": "Thanh Bình", "lap vo": "Lấp Vò",
        "lai vung": "Lai Vung", "chau thanh": "Châu Thành"
    },
    "vinh long": {
        "vinh long": "Vĩnh Long", "long ho": "Long Hồ", "mang thit": "Mang Thít", "vung liem": "Vũng Liêm",
        "tam binh": "Tam Bình", "binh minh": "Bình Minh", "tra on": "Trà Ôn", "binh tan": "Bình Tân"
    },
    "an giang": {
        "long xuyen": "Long Xuyên", "chau doc": "Châu Đốc", "an phu": "An Phú", "tan chau": "Tân Châu",
        "phu tan": "Phú Tân", "chau phu": "Châu Phú", "tinh bien": "Tịnh Biên", "tri ton": "Tri Tôn",
        "chau thanh": "Châu Thành", "cho moi": "Chợ Mới", "thoai son": "Thoại Sơn"
    },
    "ben tre": {
        "ben tre": "Bến Tre", "chau thanh": "Châu Thành", "cho lach": "Chợ Lách", "mo cay nam": "Mỏ Cày Nam",
        "giong trom": "Giồng Trôm", "binh dai": "Bình Đại", "ba tri": "Ba Tri", "thanh phu": "Thạnh Phú",
        "mo cay bac": "Mỏ Cày Bắc"
    },
    "tien giang": {
        "my tho": "Mỹ Tho", "go cong": "Gò Công", "cai lay": "Cai Lậy", "tan phuoc": "Tân Phước",
        "cai be": "Cái Bè", "chau thanh": "Châu Thành", "cho gao": "Chợ Gạo", "go cong tay": "Gò Công Tây",
        "go cong dong": "Gò Công Đông", "tan phu dong": "Tân Phú Đông"
    },
    "long an": {
        "tan an": "Tân An", "kien tuong": "Kiến Tường", "tan hung": "Tân Hưng", "vinh hung": "Vĩnh Hưng",
        "moc hoa": "Mộc Hóa", "tan thanh": "Tân Thạnh", "thanh hoa": "Thạnh Hóa", "duc hue": "Đức Huệ",
        "duc hoa": "Đức Hòa", "ben luc": "Bến Lức", "thu thua": "Thủ Thừa", "tan tru": "Tân Trụ",
        "can duoc": "Cần Đước", "can giuoc": "Cần Giuộc", "chau thanh": "Châu Thành"
    },
    "kien giang": {
        "rach gia": "Rạch Giá", "ha tien": "Hà Tiên", "kien luong": "Kiên Lương", "hon dat": "Hòn Đất",
        "tan hiep": "Tân Hiệp", "chau thanh": "Châu Thành", "giong rieng": "Giồng Riềng", "go quao": "Gò Quao",
        "an bien": "An Biên", "an minh": "An Minh", "vinh thuan": "Vĩnh Thuận", "phu quoc": "Phú Quốc",
        "kien hai": "Kiên Hải", "u minh thuong": "U Minh Thượng", "giang thanh": "Giang Thành"
    },
    "hau giang": {
        "vi thanh": "Vị Thanh", "nga bay": "Ngã Bảy", "chau thanh a": "Châu Thành A", "chau thanh": "Châu Thành",
        "phung hiep": "Phụng Hiệp", "vi thuy": "Vị Thủy", "long my": "Long Mỹ"
    },
    "soc trang": {
        "soc trang": "Sóc Trăng", "chau thanh": "Châu Thành", "ke sach": "Kế Sách", "my tu": "Mỹ Tú",
        "cu lao dung": "Cù Lao Dung", "long phu": "Long Phú", "my xuyen": "Mỹ Xuyên", "nga nam": "Ngã Năm",
        "thanh tri": "Thạnh Trị", "vinh chau": "Vĩnh Châu", "tran de": "Trần Đề"
    },
    "tra vinh": {
        "tra vinh": "Trà Vinh", "cang long": "Càng Long", "cau ke": "Cầu Kè", "tieu can": "Tiểu Cần",
        "chau thanh": "Châu Thành", "cau ngang": "Cầu Ngang", "tra cu": "Trà Cú", "duyen hai": "Duyên Hải"
    },
    "bac lieu": {
        "bac lieu": "Bạc Liêu", "hong dan": "Hồng Dân", "phuoc long": "Phước Long", "vinh loi": "Vĩnh Lợi",
        "gia rai": "Giá Rai", "dong hai": "Đông Hải", "hoa binh": "Hòa Bình"
    },
    "ca mau": {
        "ca mau": "Cà Mau", "u minh": "U Minh", "thoi binh": "Thới Bình", "tran van thoi": "Trần Văn Thời",
        "cai nuoc": "Cái Nước", "dam doi": "Đầm Dơi", "nam can": "Năm Căn", "phu tan": "Phú Tân",
        "ngoc hien": "Ngọc Hiển"
    },
    "binh duong": {
        "thu dau mot": "Thủ Dầu Một", "ben cat": "Bến Cát", "tan uyen": "Tân Uyên", "di an": "Dĩ An",
        "thuan an": "Thuận An", "dau tieng": "Dầu Tiếng", "bau bang": "Bàu Bàng", "phu giao": "Phú Giáo",
        "bac tan uyen": "Bắc Tân Uyên"
    },
    "dong nai": {
        "bien hoa": "Biên Hòa", "long khanh": "Long Khánh", "tan phu": "Tân Phú", "vinh cuu": "Vĩnh Cửu",
        "dinh quan": "Định Quán", "trang bom": "Trảng Bom", "thong nhat": "Thống Nhất", "cam my": "Cẩm Mỹ",
        "long thanh": "Long Thành", "xuan loc": "Xuân Lộc", "nhon trach": "Nhơn Trạch"
    },
    "ba ria vung tau": {
        "vung tau": "Vũng Tàu", "ba ria": "Bà Rịa", "chau duc": "Châu Đức", "xuyen moc": "Xuyên Mộc",
        "long dien": "Long Điền", "dat do": "Đất Đỏ", "phu my": "Phú Mỹ", "con dao": "Côn Đảo"
    },
    "tay ninh": {
        "tay ninh": "Tây Ninh", "tan bien": "Tân Biên", "tan chau": "Tân Châu", "duong minh chau": "Dương Minh Châu",
        "chau thanh": "Châu Thành", "hoa thanh": "Hòa Thành", "go dau": "Gò Dầu", "ben cau": "Bến Cầu",
        "trang bang": "Trảng Bàng"
    },
    "binh phuoc": {
        "dong xoai": "Đồng Xoài", "binh long": "Bình Long", "phuoc long": "Phước Long", "bu gia map": "Bù Gia Mập",
        "loc ninh": "Lộc Ninh", "bu dop": "Bù Đốp", "hon quan": "Hớn Quản", "dong phu": "Đồng Phú",
        "bu dang": "Bù Đăng", "chon thanh": "Chơn Thành", "phu rieng": "Phú Riềng"
    },
}

# ---------------------------------------------------------------------------
# 4. Universal National Gazetteer of Wards / Communes / Hamlets
# ---------------------------------------------------------------------------
COMMUNES_AND_HAMLETS_GAZETTEER: Dict[str, str] = {
    # Key communes & towns
    "tan luoc": "Tân Lược",
    "tan phu trung": "Tân Phú Trung",
    "tan binh": "Tân Bình",
    "tan phu": "Tân Phú",
    "tan thoi": "Tân Thới",
    "tan thanh": "Tân Thạnh",
    "tan thuan": "Tân Thuận",
    "tan hoa": "Tân Hòa",
    "tan hiep": "Tân Hiệp",
    "tan an": "Tân An",
    "tan phong": "Tân Phong",
    "tan my": "Tân Mỹ",
    "tan quy": "Tân Quy",
    "tan duong": "Tân Dương",
    "tan hung": "Tân Hưng",
    "tan kien": "Tân Kiên",
    "tan tao": "Tân Tạo",
    "phu binh": "Phú Bình",
    "phu thuan": "Phú Thuận",
    "phu thanh": "Phú Thạnh",
    "phu thọ": "Phú Thọ",
    "phu tho": "Phú Thọ",
    "phu dien": "Phú Điền",
    "phu tam": "Phú Tâm",
    "phu tan": "Phú Tân",
    "phu long": "Phú Long",
    "phu hoa": "Phú Hòa",
    "phu duc": "Phú Đức",
    "phu cuong": "Phú Cường",
    "phu my": "Phú Mỹ",
    "phu huu": "Phú Hữu",
    "phu loi": "Phú Lợi",
    "phu giao": "Phú Giáo",
    "binh thanh": "Bình Thạnh",
    "binh hoa": "Bình Hòa",
    "binh phu": "Bình Phú",
    "binh thuy": "Bình Thủy",
    "binh minh": "Bình Minh",
    "binh chanh": "Bình Chánh",
    "binh tan": "Bình Tân",
    "binh an": "Bình An",
    "binh hung": "Bình Hưng",
    "binh my": "Bình Mỹ",
    "binh son": "Bình Sơn",
    "hoa an": "Hòa An",
    "hoa binh": "Hòa Bình",
    "hoa thanh": "Hòa Thành",
    "hoa hiep": "Hòa Hiệp",
    "hoa thuan": "Hòa Thuận",
    "hoa phu": "Hòa Phú",
    "hoa loi": "Hòa Lợi",
    "long an": "Long An",
    "long hau": "Long Hậu",
    "long hoa": "Long Hòa",
    "long thanh": "Long Thành",
    "long duc": "Long Đức",
    "long hiep": "Long Hiệp",
    "long phu": "Long Phú",
    "long thoi": "Long Thới",
    "my an": "Mỹ An",
    "my hoa": "Mỹ Hòa",
    "my phu": "Mỹ Phú",
    "my thuan": "Mỹ Thuận",
    "my tho": "Mỹ Tho",
    "my phuoc": "Mỹ Phước",
    "my hanh": "Mỹ Hạnh",
    "thanh an": "Thạnh An",
    "thanh hoa": "Thạnh Hóa",
    "thanh phu": "Thạnh Phú",
    "thanh thoi": "Thạnh Thới",
    "thanh loc": "Thạnh Lộc",
    "thanh my": "Thạnh Mỹ",
    "thoi an": "Thới An",
    "thoi hoa": "Thới Hòa",
    "thoi binh": "Thới Bình",
    "thoi tam thon": "Thới Tam Thôn",
    "vinh an": "Vĩnh An",
    "vinh hoa": "Vĩnh Hòa",
    "vinh loc": "Vĩnh Lộc",
    "vinh thanh": "Vĩnh Thạnh",
    "vinh thuan": "Vĩnh Thuận",
    "vinh phu": "Vĩnh Phú",
    # Directional / Hamlets
    "ap tay": "Ấp Tây",
    "ap dong": "Ấp Đông",
    "ap nam": "Ấp Nam",
    "ap bac": "Ấp Bắc",
    "ap trung": "Ấp Trung",
    "tay": "Tây",
    "dong": "Đông",
    "nam": "Nam",
    "bac": "Bắc",
    "trung": "Trung",
    "thuong": "Thượng",
    "ha": "Hạ",
}


class VietnameseAdministrativeRestorer:
    """
    Upgraded 3-Level Hierarchical & Fuzzy Matching Engine for Nationwide Vietnamese Addresses.
    Uses RapidFuzz (score >= 80%) across:
      - Level 1: 63 Provinces / Municipalities
      - Level 2: Districts constrained to matched Province
      - Level 3: Wards / Communes constrained to matched District
      - Lower-level Units: Villages, Hamlets, Groups (Thôn, Ấp, Tổ, Khóm, Số nhà, Đường)
    """

    SIMILARITY_THRESHOLD = 80.0  # 80% fuzzy match threshold

    @classmethod
    def fuzzy_match(cls, query: str, candidates: Dict[str, str], threshold: float = 80.0) -> Optional[Tuple[str, str, float]]:
        """
        Fuzzy matches a raw query string against candidate dictionary {unaccented_key: canonical_accented_val}.
        Uses Levenshtein ratio (fuzz.ratio) with threshold >= 80% to robustly handle OCR character corruptions.
        """
        if not query:
            return None

        clean_q = remove_vietnamese_accents(query).lower().strip()
        clean_q = re.sub(r'[\/._\-:]+', ' ', clean_q)
        clean_q = re.sub(r'\s+', ' ', clean_q).strip()

        if not clean_q:
            return None

        # 1. Exact match fast path
        if clean_q in candidates:
            return clean_q, candidates[clean_q], 100.0

        # 2. Fuzzy match via RapidFuzz ratio
        if HAS_RAPIDFUZZ:
            best_match = None
            best_score = 0.0

            len_q = len(clean_q)
            for cand_key, cand_val in candidates.items():
                len_c = len(cand_key)
                if abs(len_q - len_c) > max(2, int(0.25 * len_c)):
                    continue

                score = fuzz.ratio(clean_q, cand_key)
                if score > best_score:
                    best_score = score
                    best_match = (cand_key, cand_val, float(score))

            if best_match and best_score >= threshold:
                return best_match
        else:
            import difflib
            for cand_key, cand_val in candidates.items():
                if abs(len(clean_q) - len(cand_key)) > 2:
                    continue
                ratio = difflib.SequenceMatcher(None, clean_q, cand_key).ratio() * 100.0
                if ratio >= threshold:
                    return cand_key, cand_val, ratio

        return None

    @classmethod
    def restore_address_diacritics(cls, address: Optional[str]) -> Optional[str]:
        """
        Main 3-Level Hierarchical Address Restoration Pipeline:
        1. Pre-process OCR dot/comma confusions and normalize units (T09 -> Tổ 9, P.5, Q.1)
        2. Level 1: Match Province (Tỉnh/Thành phố) at the end of the address.
        3. Level 2: Match District (Quận/Huyện) constrained to the matched Province.
        4. Level 3: Match Commune/Ward (Xã/Phường/Thị trấn) constrained to the District.
        5. Match & preserve lower units (Thôn, Ấp, Tổ, Khóm, Số nhà, Đường).
        """
        if not address:
            return address

        norm = UnicodeNormalizer.normalize(str(address))
        if not norm:
            return norm

        # Tiền xử lý sửa lỗi OCR biến dấu phẩy/khoảng trắng thành dấu chấm giữa các từ địa danh
        # (VD: "Tan Luoc.Binh Tan" -> "Tan Luoc, Binh Tan"), bảo toàn P.5, Q.1
        norm = re.sub(r'(?<=[a-zA-Z\u00C0-\u1EF9]{2})\.(?=[a-zA-Z0-9\u00C0-\u1EF9])', ', ', norm)
        norm = re.sub(r'(?<=[a-zA-Z\u00C0-\u1EF9]{2})\.\s+(?=[a-zA-Z0-9\u00C0-\u1EF9])', ', ', norm)

        # Loại bỏ các nhãn tiêu đề OCR bị lẫn vào chuỗi địa chỉ
        # (VD: "Noi thurng trư/ Place of residence", "Place of origin:", "Nơi thường trú:")
        norm = re.sub(
            r'^(?:n[o\u01a1]i\s+)?(?:th[u\u01b0][o\u01a1]ng|thurng|c[u\u01b0]|thuong)\s*(?:tr[u\u00fa\u01b0]|tru)?\s*(?:[\/\-:\.]|\s+)*(?:place\s+of\s+residence|residence)?[:\s\/._,-]*',
            '', norm, flags=re.IGNORECASE
        ).strip()
        norm = re.sub(
            r'^(?:qu[e\xea]\s+qu[a\xe1]n|noi\s+dang\s+ky\s+khai\s+sinh|dang\s+ky\s+khai\s+sinh|khai\s+sinh)\s*(?:[\/\-:\.]|\s+)*(?:place\s+of\s+origin|place\s+of\s+birth(?:\s+registration)?)?[:\s\/._,-]*',
            '', norm, flags=re.IGNORECASE
        ).strip()
        norm = re.sub(r'^(?:place\s+of\s+(?:residence|origin|birth(?:\s+registration)?)|pace\s+of\s+brth|residence)[:\s\/._,-]*', '', norm, flags=re.IGNORECASE).strip()

        # Normalize unit prefixes like 'T09', 'T.9', 'To 9' -> 'Tổ 9'
        norm = re.sub(r'\bT0?(\d+)\b', r'Tổ \1', norm)
        norm = re.sub(r'\bTo\s+(\d+)\b', r'Tổ \1', norm)

        # Chuẩn hóa khoảng trắng thừa và dấu phẩy liên tiếp
        norm = re.sub(r'[\r\n\t]+', ' ', norm)
        norm = re.sub(r'\s*,\s*', ', ', norm)
        norm = re.sub(r',+', ',', norm)
        norm = re.sub(r'\s+', ' ', norm).strip()

        # Split into comma-separated segments
        raw_segments = [s.strip() for s in norm.split(',') if s.strip()]
        if not raw_segments:
            return norm

        # Run 3-Level Hierarchical Matching
        restored_segments = cls._hierarchical_match_address(raw_segments)

        result = ", ".join(restored_segments)
        return UnicodeNormalizer.normalize(result)

    @classmethod
    def _hierarchical_match_address(cls, segments: List[str]) -> List[str]:
        """
        Hierarchical 3-Level Matcher traversing segments from right (Province) to left (Hamlet/Group/House).
        """
        num_segs = len(segments)
        restored: List[str] = [s for s in segments]

        matched_province_key: Optional[str] = None
        matched_province_name: Optional[str] = None
        matched_province_idx: Optional[int] = None

        # -------------------------------------------------------------
        # Level 1: Find Province/City in the rightmost segments
        # -------------------------------------------------------------
        for idx in range(num_segs - 1, max(-1, num_segs - 3), -1):
            seg_text = segments[idx]
            # Strip province prefix words
            clean_p = re.sub(r'^(?:tinh|thanh pho|tp\.?|t\.?)\s*', '', remove_vietnamese_accents(seg_text).lower()).strip()
            p_match = cls.fuzzy_match(clean_p, PROVINCES_GAZETTEER, threshold=cls.SIMILARITY_THRESHOLD)
            if p_match:
                matched_province_key, matched_province_name, _ = p_match
                matched_province_idx = idx
                restored[idx] = matched_province_name
                break

        # -------------------------------------------------------------
        # Level 2: Find District in segments preceding Province
        # -------------------------------------------------------------
        # District candidates: if province is matched, restrict to districts of that province; else all districts
        district_candidates: Dict[str, str] = {}
        if matched_province_key and matched_province_key in DISTRICTS_BY_PROVINCE:
            district_candidates.update(DISTRICTS_BY_PROVINCE[matched_province_key])
        else:
            for p_key, dists in DISTRICTS_BY_PROVINCE.items():
                district_candidates.update(dists)

        matched_district_key: Optional[str] = None
        matched_district_name: Optional[str] = None
        matched_district_idx: Optional[int] = None

        start_dist_idx = (matched_province_idx - 1) if (matched_province_idx is not None and matched_province_idx > 0) else (num_segs - 1 if matched_province_idx is None else -1)

        if start_dist_idx >= 0:
            for idx in range(start_dist_idx, max(-1, start_dist_idx - 2), -1):
                seg_text = segments[idx]
                clean_d = re.sub(r'^(?:huyen|quan|thi xa|thanh pho|tx\.?|tp\.?|q\.?|h\.?)\s*', '', remove_vietnamese_accents(seg_text).lower()).strip()
                d_match = cls.fuzzy_match(clean_d, district_candidates, threshold=cls.SIMILARITY_THRESHOLD)
                if d_match:
                    matched_district_key, matched_district_name, _ = d_match
                    matched_district_idx = idx
                    restored[idx] = matched_district_name
                    break

        # -------------------------------------------------------------
        # Level 3 & Lower Units: Process remaining segments
        # (Wards/Communes and Hamlets/Groups/Street numbers)
        # -------------------------------------------------------------
        for idx in range(num_segs):
            if idx in (matched_province_idx, matched_district_idx):
                continue
            seg_text = segments[idx]
            restored_seg = cls._restore_local_or_commune_segment(seg_text)
            restored[idx] = restored_seg

        return [r for r in restored if r]

    @classmethod
    def _restore_local_or_commune_segment(cls, segment: str) -> str:
        """
        Restores Wards/Communes (Level 3) and Local Units (Thôn, Ấp, Tổ, Số nhà, Đường).
        Preserves exact numbers and prefix structures.
        """
        if not segment:
            return segment

        clean_seg = remove_vietnamese_accents(segment).lower().strip()

        # 1. Commune/Hamlet Exact or Fuzzy Match
        c_match = cls.fuzzy_match(clean_seg, COMMUNES_AND_HAMLETS_GAZETTEER, threshold=cls.SIMILARITY_THRESHOLD)
        if c_match:
            return c_match[1]

        words = segment.split()
        if not words:
            return segment

        # If more than 2 words in the segment, run sliding window phrase scan to catch multiple nested units
        if len(words) > 2:
            return cls._restore_phrase_words(words)

        first_word_clean = remove_vietnamese_accents(words[0]).lower()
        two_words_clean = " ".join(remove_vietnamese_accents(w).lower() for w in words[:2])

        prefix = ""
        remainder_words = words

        if two_words_clean in ADMINISTRATIVE_PREFIXES:
            prefix = ADMINISTRATIVE_PREFIXES[two_words_clean]
            remainder_words = words[2:]
        elif first_word_clean in ADMINISTRATIVE_PREFIXES:
            prefix = ADMINISTRATIVE_PREFIXES[first_word_clean]
            remainder_words = words[1:]

        if prefix:
            if not remainder_words:
                return prefix
            remainder_str = " ".join(remainder_words)
            # If remainder is just a number (e.g. '09', '9', '12B'): preserve directly
            if re.match(r'^\d+[a-zA-Z]?$', remainder_str):
                return f"{prefix} {remainder_str}"

            # Fuzzy match remainder against gazetteers
            rem_match = cls.fuzzy_match(remainder_str, COMMUNES_AND_HAMLETS_GAZETTEER, threshold=cls.SIMILARITY_THRESHOLD)
            if rem_match:
                return f"{prefix} {rem_match[1]}"
            else:
                restored_rem = cls._restore_phrase_words(remainder_words)
                return f"{prefix} {restored_rem}".strip()

        return cls._restore_phrase_words(words)

    @classmethod
    def _restore_phrase_words(cls, words: List[str]) -> str:
        """
        Scans words using sliding windows to match known administrative place names.
        Preserves space separation for ordinary words while inserting commas between distinct administrative units.
        """
        n = len(words)
        tokens: List[Tuple[bool, str]] = []
        i = 0

        while i < n:
            matched = False
            # Try 4-gram, 3-gram, 2-gram
            for l in range(min(4, n - i), 1, -1):
                sub_phrase = " ".join(words[i:i + l])
                m_res = cls.fuzzy_match(sub_phrase, COMMUNES_AND_HAMLETS_GAZETTEER, threshold=cls.SIMILARITY_THRESHOLD)
                if m_res:
                    tokens.append((True, m_res[1]))
                    i += l
                    matched = True
                    break
                p_res = cls.fuzzy_match(sub_phrase, PROVINCES_GAZETTEER, threshold=cls.SIMILARITY_THRESHOLD)
                if p_res:
                    tokens.append((True, p_res[1]))
                    i += l
                    matched = True
                    break

            if not matched:
                w_clean = remove_vietnamese_accents(words[i]).lower()
                if w_clean in ADMINISTRATIVE_PREFIXES:
                    prefix_val = ADMINISTRATIVE_PREFIXES[w_clean]
                    if i + 1 < n:
                        # Check prefix + number
                        if re.match(r'^\d+[a-zA-Z]?$', words[i + 1]):
                            tokens.append((True, f"{prefix_val} {words[i + 1]}"))
                            i += 2
                            continue
                        # Check prefix + next place name
                        sub2_matched = False
                        for l2 in range(min(4, n - (i + 1)), 0, -1):
                            sub2 = " ".join(words[i + 1:i + 1 + l2])
                            m2 = cls.fuzzy_match(sub2, COMMUNES_AND_HAMLETS_GAZETTEER, threshold=cls.SIMILARITY_THRESHOLD)
                            if m2:
                                tokens.append((True, f"{prefix_val} {m2[1]}"))
                                i += 1 + l2
                                sub2_matched = True
                                break
                        if sub2_matched:
                            continue
                    tokens.append((False, prefix_val))
                    i += 1
                elif w_clean in COMMUNES_AND_HAMLETS_GAZETTEER:
                    tokens.append((True, COMMUNES_AND_HAMLETS_GAZETTEER[w_clean]))
                    i += 1
                elif w_clean in PROVINCES_GAZETTEER:
                    tokens.append((True, PROVINCES_GAZETTEER[w_clean]))
                    i += 1
                else:
                    tokens.append((False, words[i]))
                    i += 1

        blocks: List[str] = []
        current_plain: List[str] = []

        for is_admin, txt in tokens:
            if is_admin:
                if current_plain:
                    blocks.append(" ".join(current_plain))
                    current_plain = []
                blocks.append(txt)
            else:
                current_plain.append(txt)

        if current_plain:
            blocks.append(" ".join(current_plain))

        if len(blocks) > 1 and any(is_adm for is_adm, _ in tokens):
            return ", ".join(b for b in blocks if b)
        else:
            return " ".join(b for b in blocks if b)
