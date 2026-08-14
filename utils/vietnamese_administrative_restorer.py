import re
import unicodedata
from typing import Optional, List, Dict
from utils.text_normalizer import UnicodeNormalizer
from utils.text_utils import remove_vietnamese_accents


# ---------------------------------------------------------------------------
# Vietnamese Administrative Lexicon & Unit Prefixes
# Compiled according to the General Statistics Office of Vietnam (GSO)
# and Ministry of Home Affairs administrative divisions standards.
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
    "khu": "Khu",
    "cum": "Cụm",
    "cụm": "Cụm",
    "xa": "Xã",
    "xã": "Xã",
    "phuong": "Phường",
    "phường": "Phường",
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
    "quan": "Quận",
    "quận": "Quận",
    "thanh pho": "Thành phố",
    "thành phố": "Thành phố",
    "tp": "TP.",
    "tp.": "TP.",
    "tinh": "Tỉnh",
    "tỉnh": "Tỉnh",
    "so": "Số",
    "số": "Số",
    "duong": "Đường",
    "đường": "Đường",
    "hem": "Hẻm",
    "hẻm": "Hẻm",
    "ngach": "Ngách",
    "ngách": "Ngách",
    "ngõ": "Ngõ",
    "ngo": "Ngõ",
}

# Standard Gazetteer of 63 Provinces / Municipalities of Vietnam
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
    "ho chi minh": "Hồ Chí Minh",
    "tra vinh": "Trà Vinh",
    "tuyen quang": "Tuyên Quang",
    "vinh long": "Vĩnh Long",
    "vinh phuc": "Vĩnh Phúc",
    "yen bai": "Yên Bái",
}

# Standard Gazetteer of Vietnamese Administrative Units (Districts, Wards, Communes, Hamlets)
# Sourced from GSO Vietnam administrative divisions data.
COMMON_ADMIN_NAMES_GAZETTEER: Dict[str, str] = {
    # Popular Districts & Cities
    "chau thanh": "Châu Thành",
    "cao lanh": "Cao Lãnh",
    "sa dec": "Sa Đéc",
    "hong ngu": "Hồng Ngự",
    "lap vo": "Lấp Vò",
    "lai vung": "Lai Vung",
    "tam nong": "Tam Nông",
    "thap muoi": "Tháp Mười",
    "thanh binh": "Thanh Bình",
    "tan hong": "Tân Hồng",
    "tan phu trung": "Tân Phú Trung",
    "tan phu": "Tân Phú",
    "tan binh": "Tân Bình",
    "phu binh": "Phú Bình",
    "phu thuan": "Phú Thuận",
    "phu thanh": "Phú Thạnh",
    "phu hiep": "Phú Hiệp",
    "phu tho": "Phú Thọ",
    "phu dien": "Phú Điền",
    "phu tam": "Phú Tâm",
    "phu tan": "Phú Tân",
    "phu long": "Phú Long",
    "phu hoa": "Phú Hòa",
    "phu duc": "Phú Đức",
    "phu cuong": "Phú Cường",
    "binh thanh": "Bình Thạnh",
    "binh tan": "Bình Tân",
    "binh chanh": "Bình Chánh",
    "binh minh": "Bình Minh",
    "binh phuoc": "Bình Phước",
    "binh hoa": "Bình Hòa",
    "binh phu": "Bình Phú",
    "binh thuy": "Bình Thủy",
    "cai rang": "Cái Răng",
    "ninh kieu": "Ninh Kiều",
    "o mon": "Ô Môn",
    "thot not": "Thốt Nốt",
    "phong dien": "Phong Điền",
    "co do": "Cờ Đỏ",
    "vinh thanh": "Vĩnh Thạnh",
    "thoi lai": "Thới Lai",
    "cang long": "Càng Long",
    "cau ke": "Cầu Kè",
    "tieu can": "Tiểu Cần",
    "tra cu": "Trà Cú",
    "duyen hai": "Duyên Hải",
    "cau ngang": "Cầu Ngang",
    "long ho": "Long Hồ",
    "mang thit": "Mang Thít",
    "tam binh": "Tam Bình",
    "tra on": "Trà Ôn",
    "vung liem": "Vũng Liêm",
    "ba tri": "Ba Tri",
    "binh dai": "Bình Đại",
    "chau lach": "Châu Lách",
    "giong trom": "Giồng Trôm",
    "mo cay bac": "Mỏ Cày Bắc",
    "mo cay nam": "Mỏ Cày Nam",
    "thanh phu": "Thạnh Phú",
    "an bien": "An Biên",
    "an minh": "An Minh",
    "giong rieng": "Giồng Riềng",
    "go quao": "Gò Quao",
    "hon dat": "Hòn Đất",
    "kien hai": "Kiên Hải",
    "kien luong": "Kiên Lương",
    "phu quoc": "Phú Quốc",
    "tan hiep": "Tân Hiệp",
    "u minh thuong": "U Minh Thượng",
    "vinh thuan": "Vĩnh Thuận",
    "ha tien": "Hà Tiên",
    "rach gia": "Rạch Giá",
    "cai be": "Cái Bè",
    "cai lay": "Cai Lậy",
    "chau thanh": "Châu Thành",
    "cho gao": "Chợ Gạo",
    "go cong": "Gò Công",
    "go cong dong": "Gò Công Đông",
    "go cong tay": "Gò Công Tây",
    "tan phuoc": "Tân Phước",
    "my tho": "Mỹ Tho",
    "tan an": "Tân An",
    "kien tuong": "Kiến Tường",
    "ben luc": "Bến Lức",
    "can duoc": "Cần Đước",
    "can giuoc": "Cần Giuộc",
    "chau thanh": "Châu Thành",
    "duc hoa": "Đức Hòa",
    "duc hue": "Đức Huệ",
    "moc hoa": "Mộc Hóa",
    "tan hung": "Tân Hưng",
    "tan thanh": "Tân Thạnh",
    "tan tru": "Tân Trụ",
    "thanh hoa": "Thạnh Hóa",
    "thu thua": "Thủ Thừa",
    "vinh hung": "Vĩnh Hưng",
    "quan 1": "Quận 1",
    "quan 2": "Quận 2",
    "quan 3": "Quận 3",
    "quan 4": "Quận 4",
    "quan 5": "Quận 5",
    "quan 6": "Quận 6",
    "quan 7": "Quận 7",
    "quan 8": "Quận 8",
    "quan 9": "Quận 9",
    "quan 10": "Quận 10",
    "quan 11": "Quận 11",
    "quan 12": "Quận 12",
    "thu duc": "Thủ Đức",
    "go vap": "Gò Vấp",
    "phu nhuan": "Phú Nhuận",
    "tan binh": "Tân Bình",
    "tan phu": "Tân Phú",
    "binh thanh": "Bình Thạnh",
    "binh tan": "Bình Tân",
    "cu chi": "Củ Chi",
    "hoc mon": "Hóc Môn",
    "binh chanh": "Bình Chánh",
    "nha be": "Nhà Bè",
    "can gio": "Cần Giờ",
    "ba dinh": "Ba Đình",
    "hoan kiem": "Hoàn Kiếm",
    "tay ho": "Tây Hồ",
    "long bien": "Long Biên",
    "cau giay": "Cầu Giấy",
    "dong da": "Đống Đa",
    "hai ba trung": "Hai Bà Trưng",
    "hoang mai": "Hoàng Mai",
    "thanh xuan": "Thanh Xuân",
    "soc son": "Sóc Sơn",
    "dong anh": "Đông Anh",
    "gia lam": "Gia Lâm",
    "nam tu liem": "Nam Từ Liêm",
    "bac tu liem": "Bắc Từ Liêm",
    "thanh tri": "Thanh Trì",
    "ha dong": "Hà Đông",
    "son tay": "Sơn Tây",
    "ba vi": "Ba Vì",
    "phuc tho": "Phúc Thọ",
    "dan phuong": "Đan Phượng",
    "hoai duc": "Hoài Đức",
    "quoc oai": "Quốc Oai",
    "thach that": "Thạch Thất",
    "chuong my": "Chương Mỹ",
    "thanh oai": "Thanh Oai",
    "thuong tin": "Thường Tín",
    "phu xuyen": "Phú Xuyên",
    "ung hoa": "Ứng Hòa",
    "my duc": "Mỹ Đức",
    # Directional / Hamlets
    "ap tay": "Ấp Tây",
    "ap dong": "Ấp Đông",
    "ap nam": "Ấp Nam",
    "ap bac": "Ấp Bắc",
    "ap trung": "Ấp Trung",
    "ap 1": "Ấp 1",
    "ap 2": "Ấp 2",
    "ap 3": "Ấp 3",
    "ap 4": "Ấp 4",
    "ap 5": "Ấp 5",
    "ap 6": "Ấp 6",
    "ap 7": "Ấp 7",
    "ap 8": "Ấp 8",
    "ap 9": "Ấp 9",
    "thon 1": "Thôn 1",
    "thon 2": "Thôn 2",
    "thon 3": "Thôn 3",
    "thon 4": "Thôn 4",
    "thon 5": "Thôn 5",
    "to 1": "Tổ 1",
    "to 2": "Tổ 2",
    "to 3": "Tổ 3",
    "to 4": "Tổ 4",
    "to 5": "Tổ 5",
    "to 6": "Tổ 6",
    "to 7": "Tổ 7",
    "to 8": "Tổ 8",
    "to 9": "Tổ 9",
    "to 10": "Tổ 10",
}


class VietnameseAdministrativeRestorer:
    """
    Generic Data-Driven Vietnamese Administrative Diacritics Restorer.
    Uses official GSO gazetteer dictionaries and administrative prefix rules
    to restore missing diacritics on address components without hardcoding individual user records.
    """

    @classmethod
    def restore_address_diacritics(cls, address: Optional[str]) -> Optional[str]:
        if not address:
            return address

        # Step 1: Pre-clean spacing and punctuation
        norm = UnicodeNormalizer.normalize(address)
        if not norm:
            return norm

        # Normalize unit prefixes like 'T09', 'T.9', 'To 9' -> 'Tổ 9'
        norm = re.sub(r'\bT0?(\d+)\b', r'Tổ \1', norm)
        norm = re.sub(r'\bTo\s+(\d+)\b', r'Tổ \1', norm)

        # Split into comma-separated segments or whitespace-separated clauses
        segments = [s.strip() for s in norm.split(',') if s.strip()]

        restored_segments: List[str] = []
        for segment in segments:
            restored = cls._restore_segment(segment)
            restored_segments.append(restored)

        result = ", ".join(restored_segments)
        return UnicodeNormalizer.normalize(result)

    @classmethod
    def _restore_segment(cls, segment: str) -> str:
        if not segment:
            return segment

        clean_seg = remove_vietnamese_accents(segment).lower().strip()

        # 1. Exact match with province gazetteer
        if clean_seg in PROVINCES_GAZETTEER:
            return PROVINCES_GAZETTEER[clean_seg]

        # 2. Exact match with common administrative gazetteer
        if clean_seg in COMMON_ADMIN_NAMES_GAZETTEER:
            return COMMON_ADMIN_NAMES_GAZETTEER[clean_seg]

        # 3. Phrasal Trie / N-gram restoration for compound segments (e.g. 'Ap Phu Binh', 'Tan Phu Trung')
        words = segment.split()
        if not words:
            return segment

        # Check for administrative prefixes at the start of the segment
        # e.g., 'Ap Phu Binh' -> prefix='Ấp', remainder='Phu Binh' -> 'Ấp Phú Bình'
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
            if remainder_words:
                remainder_str = " ".join(remainder_words)
                remainder_clean = remove_vietnamese_accents(remainder_str).lower()

                # Try matching remainder against gazetteer
                if remainder_clean in COMMON_ADMIN_NAMES_GAZETTEER:
                    restored_remainder = COMMON_ADMIN_NAMES_GAZETTEER[remainder_clean]
                elif remainder_clean in PROVINCES_GAZETTEER:
                    restored_remainder = PROVINCES_GAZETTEER[remainder_clean]
                else:
                    # Fallback to word-by-word restoration
                    restored_remainder = cls._restore_phrase_words(remainder_words)

                return f"{prefix} {restored_remainder}".strip()
            else:
                return prefix

        # No direct prefix: try multi-word gazetteer lookup on the entire phrase
        return cls._restore_phrase_words(words)

    @classmethod
    def _restore_phrase_words(cls, words: List[str]) -> str:
        """
        Scans words using sliding windows to match known administrative place names.
        """
        n = len(words)
        result: List[str] = []
        i = 0

        while i < n:
            matched = False
            # Try 4-gram, 3-gram, 2-gram down to 1-gram
            for l in range(min(4, n - i), 0, -1):
                sub_phrase = " ".join(words[i:i + l])
                sub_clean = remove_vietnamese_accents(sub_phrase).lower()

                if sub_clean in COMMON_ADMIN_NAMES_GAZETTEER:
                    result.append(COMMON_ADMIN_NAMES_GAZETTEER[sub_clean])
                    i += l
                    matched = True
                    break
                elif sub_clean in PROVINCES_GAZETTEER:
                    result.append(PROVINCES_GAZETTEER[sub_clean])
                    i += l
                    matched = True
                    break
                elif sub_clean in ADMINISTRATIVE_PREFIXES:
                    result.append(ADMINISTRATIVE_PREFIXES[sub_clean])
                    i += l
                    matched = True
                    break

            if not matched:
                result.append(words[i])
                i += 1

        return " ".join(result)
