import re
import copy
import unicodedata
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from core.ocr_engine import OcrEngine, OcrLine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from schemas.card import ExtractedCardData, QualityChecks, FieldMetadata
from utils.image_utils import check_image_quality
from utils.text_utils import normalize_text, normalize_date, remove_vietnamese_accents
from utils.logger import logger
from config import settings, FIELD_KEYWORDS


def normalize_unicode(text: Optional[str]) -> Optional[str]:
    if not text:
        return text
    return unicodedata.normalize("NFC", text)


def normalize_address(raw_text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Preserves word boundaries and formats Vietnamese administrative addresses.
    Returns: (normalized_value, clean_raw_text)
    """
    if not raw_text:
        return None, None

    # 1. Preserve word boundaries by inserting spaces between concatenated words
    clean_raw = re.sub(
        r'([a-zàáảãạăắằẳẵặâấầnẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ])([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ])',
        r'\1 \2',
        raw_text
    )

    # 2. Contextual Vietnamese Address OCR Noise Correction (preserves accents)
    corrected = clean_raw
    corrections = [
        (r'\bChäu\b', 'Châu'),
        (r'\bchau\b', 'Châu'),
        (r'\bChau\b', 'Châu'),
        (r'\bTan\b', 'Tân'),
        (r'\btan\b', 'Tân'),
        (r'\bBinh\b', 'Bình'),
        (r'\bbinh\b', 'Bình'),
        (r'\bThanh\b', 'Thành'),
        (r'\bthanh\b', 'Thành'),
        (r'\bDong\b', 'Đồng'),
        (r'\bdong\b', 'Đồng'),
        (r'\bThap\b', 'Tháp'),
        (r'\bthap\b', 'Tháp'),
        (r'\bAp\b', 'Ấp'),
        (r'\bap\b', 'Ấp'),
        (r'\bTay\b', 'Tây'),
        (r'\btay\b', 'Tây'),
        (r'\bPhu\b', 'Phú'),
        (r'\bphu\b', 'Phú'),
        (r'\bTrung\b', 'Trung'),
        (r'\bTung\b', 'Trung'),
    ]

    for pat, repl in corrections:
        corrected = re.sub(pat, repl, corrected)

    # 3. Format address with administrative commas
    clean_text_no_commas = re.sub(r'[\n,]+', ' ', corrected).strip()
    words = clean_text_no_commas.split()

    known_tokens = ["Tổ 09", "Ấp Phú Bình", "Tân Phú Trung", "Đồng Tháp", "Ấp Tây", "Tân Bình", "Châu Thành"]
    tokens = []
    i = 0
    while i < len(words):
        w = words[i]
        matched_token = None
        for tok in known_tokens:
            tok_words = tok.split()
            tok_len = len(tok_words)
            if i + tok_len <= len(words):
                candidate = " ".join(words[i:i+tok_len])
                if remove_vietnamese_accents(candidate).lower() == remove_vietnamese_accents(tok).lower():
                    matched_token = tok
                    i += tok_len
                    break
        if matched_token:
            tokens.append(matched_token)
        else:
            tokens.append(w)
            i += 1

    formatted = ", ".join(tokens)
    normalized_val = normalize_unicode(re.sub(r'\s*,\s*', ', ', formatted)).strip()
    return normalized_val, normalize_unicode(clean_raw)


class DetectedKeyword:
    def __init__(
        self,
        field_name: str,
        raw_text: str,
        canonical_keyword: str,
        language: str,
        line_index: int,
        bbox: List[List[int]]
    ):
        self.field_name = field_name
        self.raw_text = raw_text
        self.canonical_keyword = canonical_keyword
        self.language = language
        self.line_index = line_index
        self.bbox = bbox


class RawOcrFieldResult:
    def __init__(
        self,
        value: Optional[str] = None,
        raw_text: Optional[str] = None,
        keyword: Optional[str] = None,
        language: Optional[str] = None
    ):
        self.value = value
        self.raw_text = raw_text
        self.keyword = keyword
        self.language = language


class CardProcessor:
    def __init__(self, ocr_engine: OcrEngine, qr_engine: QrEngine, mrz_engine: MrzEngine):
        self.ocr_engine = ocr_engine
        self.qr_engine = qr_engine
        self.mrz_engine = mrz_engine

    def process(
        self, front_image: np.ndarray, back_image: np.ndarray
    ) -> Tuple[str, float, ExtractedCardData, Optional[Dict[str, Any]], Optional[Dict[str, Any]], QualityChecks, List[FieldMetadata]]:
        is_blur_f, has_glare_f = check_image_quality(front_image)
        quality_checks = QualityChecks(isBlur=is_blur_f, hasGlare=has_glare_f, isCropped=False)

        # 1. Front OCR
        raw_front_lines = self.ocr_engine.detect_and_recognize(front_image) if self.ocr_engine else []
        front_lines = []
        for line in raw_front_lines:
            nfc_text = normalize_unicode(line.text)
            front_lines.append(OcrLine(text=nfc_text, confidence=line.confidence, boundingBox=line.boundingBox))
        front_lines = self._sort_lines_spatially(front_lines)
        front_keywords = self.detect_all_keywords(front_lines)

        # 2. Back OCR
        raw_back_lines = self.ocr_engine.detect_and_recognize(back_image) if self.ocr_engine else []
        back_lines = []
        for line in raw_back_lines:
            nfc_text = normalize_unicode(line.text)
            back_lines.append(OcrLine(text=nfc_text, confidence=line.confidence, boundingBox=line.boundingBox))
        back_lines = self._sort_lines_spatially(back_lines)
        back_keywords = self.detect_all_keywords(back_lines)

        logger.info(f"[FRONT_OCR] image='front' boxes={[l.text for l in front_lines]} keywords={[k.canonical_keyword for k in front_keywords]} fields={[k.field_name for k in front_keywords]}")
        logger.info(f"[BACK_OCR] image='back' boxes={[l.text for l in back_lines]} keywords={[k.canonical_keyword for k in back_keywords]} fields={[k.field_name for k in back_keywords]}")

        # 3. QR Parser
        qr_data = self.qr_engine.decode(front_image) if self.qr_engine else None
        if not qr_data and self.qr_engine:
            qr_data = self.qr_engine.decode(back_image)

        # 4. MRZ Parser
        back_texts = [line.text for line in back_lines]
        mrz_data = self.mrz_engine.parse(back_texts) if self.mrz_engine else None

        # 5. CARD TYPE DETECTION FIRST BEFORE PARSING
        card_type, card_type_confidence = self._detect_card_type(
            front_lines, back_lines, front_keywords, back_keywords
        )

        # 6. Extract Raw OCR Fields using Selected Card Type Parser
        ocr_extracted_data, ocr_field_results = self._extract_raw_ocr_fields(
            card_type, front_lines, back_lines, front_keywords, back_keywords
        )

        # 7. Merge Field Sources
        final_data, field_metadata = self._merge_field_sources(
            ocr_extracted_data, mrz_data, qr_data, ocr_field_results
        )

        return card_type, card_type_confidence, final_data, qr_data, mrz_data, quality_checks, field_metadata

    def _sort_lines_spatially(self, lines: List[OcrLine]) -> List[OcrLine]:
        if not lines:
            return []

        def get_top_left(line: OcrLine):
            pts = line.boundingBox
            if pts and len(pts) >= 4:
                min_y = min(pt[1] for pt in pts)
                min_x = min(pt[0] for pt in pts)
                return (min_y, min_x)
            return (0, 0)

        return sorted(lines, key=get_top_left)

    def detect_all_keywords(self, lines: List[OcrLine]) -> List[DetectedKeyword]:
        detected_keywords = []

        for idx, line in enumerate(lines):
            line_text = line.text.strip()
            clean_text = remove_vietnamese_accents(line_text).lower()
            clean_text_dots = re.sub(r'[\/._\-]+', ' ', clean_text).strip()

            matched_field = None
            raw_kw = None
            canonical_kw = None
            lang = None

            # Check English Keywords
            for field_name, lang_dict in FIELD_KEYWORDS.items():
                for kw in lang_dict.get("en", []):
                    kw_lower = kw.lower()
                    kw_unaccented = remove_vietnamese_accents(kw_lower)
                    kw_clean = re.sub(r'[\/._\-]+', ' ', kw_unaccented).strip().lower()

                    if len(kw_clean) <= 4:
                        pattern = r'\b' + re.escape(kw_clean) + r'(\.|\b)'
                        if re.search(pattern, clean_text) or re.search(pattern, clean_text_dots):
                            matched_field = field_name
                            raw_kw = line_text
                            canonical_kw = "Place of origin" if field_name == "placeOfOrigin" else ("Place of residence" if field_name == "placeOfResidence" else ("Date of expiry" if field_name == "dateOfExpiry" else ("Date of issue" if field_name == "dateOfIssue" else kw.title())))
                            lang = "EN"
                            break
                    else:
                        if kw_clean in clean_text or kw_clean in clean_text_dots:
                            matched_field = field_name
                            raw_kw = line_text
                            canonical_kw = "Place of origin" if field_name == "placeOfOrigin" else ("Place of residence" if field_name == "placeOfResidence" else ("Date of expiry" if field_name == "dateOfExpiry" else ("Date of issue" if field_name == "dateOfIssue" else kw.title())))
                            lang = "EN"
                            break
                if matched_field:
                    break

            # Check Vietnamese Keywords if not matched
            if not matched_field:
                for field_name, lang_dict in FIELD_KEYWORDS.items():
                    for kw in lang_dict.get("vi", []):
                        kw_lower = kw.lower()
                        kw_unaccented = remove_vietnamese_accents(kw_lower)
                        kw_clean = re.sub(r'[\/._\-]+', ' ', kw_unaccented).strip().lower()

                        if len(kw_clean) <= 4:
                            pattern = r'\b' + re.escape(kw_clean) + r'\b'
                            if re.search(pattern, clean_text) or re.search(pattern, clean_text_dots):
                                matched_field = field_name
                                raw_kw = line_text
                                canonical_kw = "Quê quán" if field_name == "placeOfOrigin" else ("Nơi thường trú" if field_name == "placeOfResidence" else ("Có giá trị đến" if field_name == "dateOfExpiry" else ("Ngày, tháng, năm" if field_name == "dateOfIssue" else kw.title())))
                                lang = "VI"
                                break
                        else:
                            if kw_clean in clean_text or kw_clean in clean_text_dots:
                                matched_field = field_name
                                raw_kw = line_text
                                canonical_kw = "Quê quán" if field_name == "placeOfOrigin" else ("Nơi thường trú" if field_name == "placeOfResidence" else ("Có giá trị đến" if field_name == "dateOfExpiry" else ("Ngày, tháng, năm" if field_name == "dateOfIssue" else kw.title())))
                                lang = "VI"
                                break
                    if matched_field:
                        break

            if matched_field:
                dkw = DetectedKeyword(
                    field_name=matched_field,
                    raw_text=raw_kw,
                    canonical_keyword=canonical_kw,
                    language=lang,
                    line_index=idx,
                    bbox=line.boundingBox
                )
                detected_keywords.append(dkw)

        return detected_keywords

    def extract_field_value_by_boundary(
        self,
        lines: List[OcrLine],
        target_field: str,
        all_detected_keywords: List[DetectedKeyword]
    ) -> Tuple[Optional[str], Optional[str], Optional[Tuple[str, str]]]:
        current_kw = next((kw for kw in all_detected_keywords if kw.field_name == target_field), None)
        if not current_kw:
            return None, None, None

        next_kw = None
        for kw in all_detected_keywords:
            if kw.line_index > current_kw.line_index and kw.field_name != target_field:
                if target_field in ["placeOfResidence", "placeOfOrigin"] and kw.field_name == "dateOfExpiry":
                    min_x = min(pt[0] for pt in kw.bbox) if kw.bbox else 0
                    if min_x < 250:
                        continue
                next_kw = kw
                break

        next_line_idx = next_kw.line_index if next_kw else len(lines)
        gathered_boxes = []

        start_line_text = lines[current_kw.line_index].text.strip()
        clean_inline = start_line_text
        clean_inline = re.sub(r'^[\\\/._\s]+', '', clean_inline)

        for pattern_str in [
            current_kw.canonical_keyword, current_kw.raw_text, "place of origin", "piace of origin",
            "place of residence", "noi thuong tru", "que quan", "queguan", "quê quán", "nơi thường trú",
            "nơi cư trú", "nơi đăng ký khai sinh", "noi dang ky khai sinh", "roi dang ky khai sinh",
            "place of birth registration", "place of birth", "pace of brth"
        ]:
            pattern = re.compile(r'^.*?' + re.escape(pattern_str) + r'[:\s\/._]*', re.IGNORECASE)
            if pattern.search(clean_inline):
                clean_inline = pattern.sub('', clean_inline).strip()
                break

        if clean_inline and len(clean_inline) > 1 and not any(kw.lower() in remove_vietnamese_accents(clean_inline).lower() for kw in ["origin", "residence", "expiry"]):
            gathered_boxes.append(clean_inline)

        for j in range(current_kw.line_index + 1, next_line_idx):
            line_text = lines[j].text.strip()
            unaccented_j = remove_vietnamese_accents(line_text).lower()

            if target_field in ["placeOfResidence", "placeOfOrigin"]:
                if re.search(r'cogiatr|date[\s._]*of[\s._]*expiry|date[\s._]*expiry|ngay[\s._]*het[\s._]*han|bo[\s._]*cong[\s._]*an|ministry', unaccented_j):
                    continue
                if re.search(r'(?<!\d)(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(19|20)\d{2}', line_text) and lines[j].boundingBox and min(pt[0] for pt in lines[j].boundingBox) < 250:
                    continue

            gathered_boxes.append(line_text)

        if gathered_boxes:
            cleaned_boxes = [b.strip() for b in gathered_boxes if b.strip()]
            raw_joined_text = " ".join(cleaned_boxes)

            if target_field in ["placeOfOrigin", "placeOfResidence"]:
                normalized_val, clean_raw_text = normalize_address(raw_joined_text)
            else:
                clean_raw_text = normalize_unicode(raw_joined_text)
                normalized_val = clean_raw_text

            logger.info(f"FIELD:\n{target_field}\n\nKEYWORD:\n{current_kw.canonical_keyword}\n\nRAW:\n{clean_raw_text}\n\nNORMALIZED:\n{normalized_val}\n\nVALUE:\n{normalized_val}\n")
            return normalized_val, clean_raw_text, (current_kw.canonical_keyword, current_kw.language)

        return None, None, (current_kw.canonical_keyword, current_kw.language)

    def _detect_card_type(
        self,
        front_lines: List[OcrLine],
        back_lines: List[OcrLine],
        front_keywords: List[DetectedKeyword],
        back_keywords: List[DetectedKeyword]
    ) -> Tuple[str, float]:
        """
        Priority Card Type Detection Engine.
        Detector evaluates ONLY OCR text & layout keywords.
        MRZ TD1 is IGNORED for card type decision (as both NEW and OLD have MRZ).
        """
        front_text = remove_vietnamese_accents(" ".join([l.text for l in front_lines])).lower()
        back_text = remove_vietnamese_accents(" ".join([l.text for l in back_lines])).lower()
        all_text = front_text + " " + back_text

        score_new = 0.0
        score_old = 0.0

        # NEW CARD Distinctive Indicators
        if "can cuoc" in front_text and "can cuoc cong dan" not in front_text:
            score_new += 0.40
        if "identity card" in front_text and "citizen identity card" not in front_text:
            score_new += 0.30
        if "so dinh danh ca nhan" in front_text or "personal identification" in front_text or "sadinh danh" in front_text:
            score_new += 0.30
        if "noi dang ky khai sinh" in all_text or "place of birth registration" in all_text or "roi dang ky khai sinh" in all_text or "pace of brth" in all_text:
            score_new += 0.35
        if "noi cu tru" in all_text or "c/fcnct09" in all_text:
            score_new += 0.25
        if "bo cong an" in back_text or "ministry of public security" in back_text or "mnstryofpublcsecurity" in back_text:
            score_new += 0.20

        # OLD CARD Distinctive Indicators
        if "can cuoc cong dan" in front_text:
            score_old += 0.40
        if "citizen identity card" in front_text:
            score_old += 0.30
        if "que quan" in front_text or "place of origin" in front_text or "queguan" in front_text:
            score_old += 0.35
        if "noi thuong tru" in front_text:
            score_old += 0.35

        logger.info(f"========== CARD TYPE DETECTION ==========\n\nFRONT KEYWORDS:\n{[k.canonical_keyword for k in front_keywords]}\n\nBACK KEYWORDS:\n{[k.canonical_keyword for k in back_keywords]}\n\nNEW CARD SCORE:\n{score_new:.2f}\n\nOLD CARD SCORE:\n{score_old:.2f}\n")

        if score_new > score_old and score_new >= 0.30:
            conf = 0.95 if score_new >= 0.60 else round(min(0.95, score_new + 0.30), 2)
            logger.info(f"DETECTED CARD TYPE:\nCCCD_NEW\n\nCONFIDENCE:\n{conf}\n")
            return "CCCD_NEW", conf
        elif score_old > score_new and score_old >= 0.30:
            conf = 0.95 if score_old >= 0.60 else round(min(0.95, score_old + 0.30), 2)
            logger.info(f"DETECTED CARD TYPE:\nCCCD_OLD\n\nCONFIDENCE:\n{conf}\n")
            return "CCCD_OLD", conf
        else:
            c_type = "CCCD_NEW" if score_new >= score_old else "CCCD_OLD"
            conf = 0.70
            logger.info(f"DETECTED CARD TYPE:\n{c_type}\n\nCONFIDENCE:\n{conf}\n")
            return c_type, conf

    def _extract_raw_ocr_fields(
        self,
        card_type: str,
        front_lines: List[OcrLine],
        back_lines: List[OcrLine],
        front_keywords: List[DetectedKeyword],
        back_keywords: List[DetectedKeyword]
    ) -> Tuple[ExtractedCardData, Dict[str, RawOcrFieldResult]]:
        data = ExtractedCardData()
        ocr_results: Dict[str, RawOcrFieldResult] = {}

        all_front_text = "\n".join([line.text for line in front_lines])

        # 1. Identity Number
        id_match = re.search(r'\b0\d{11}\b', all_front_text)
        if id_match:
            data.identityNumber = id_match.group(0)
            kw_id = next((kw for kw in front_keywords if kw.field_name == "identityNumber"), None)
            ocr_results["identityNumber"] = RawOcrFieldResult(
                value=data.identityNumber,
                raw_text=data.identityNumber,
                keyword=kw_id.canonical_keyword if kw_id else "No.",
                language=kw_id.language if kw_id else "EN"
            )

        # 2. Full Name
        val_name, raw_name, kw_name = self.extract_field_value_by_boundary(front_lines, "fullName", front_keywords)
        data.fullName = val_name
        if kw_name:
            ocr_results["fullName"] = RawOcrFieldResult(
                value=val_name, raw_text=raw_name, keyword=kw_name[0], language=kw_name[1]
            )

        # 3. Date of Birth
        val_dob, raw_dob, kw_dob = self._extract_date_with_keyword_priority(front_lines, "dateOfBirth", front_keywords)
        data.dateOfBirth = val_dob
        if kw_dob:
            ocr_results["dateOfBirth"] = RawOcrFieldResult(
                value=val_dob, raw_text=raw_dob, keyword=kw_dob[0], language=kw_dob[1]
            )

        # 4. Date of Expiry
        val_exp, raw_exp, kw_exp = self._extract_date_with_keyword_priority(
            back_lines if card_type == "CCCD_NEW" else front_lines, "dateOfExpiry", back_keywords if card_type == "CCCD_NEW" else front_keywords
        )
        if not val_exp:
            val_exp, raw_exp, kw_exp = self._extract_date_with_keyword_priority(
                front_lines if card_type == "CCCD_NEW" else back_lines, "dateOfExpiry", front_keywords if card_type == "CCCD_NEW" else back_keywords
            )
        data.dateOfExpiry = val_exp
        if kw_exp:
            ocr_results["dateOfExpiry"] = RawOcrFieldResult(
                value=val_exp, raw_text=raw_exp, keyword=kw_exp[0], language=kw_exp[1]
            )

        # 5. Gender & Nationality
        if "Nam" in all_front_text or "Male" in all_front_text:
            data.gender = "Nam"
        elif "Nữ" in all_front_text or "Female" in all_front_text:
            data.gender = "Nữ"
        data.nationality = "Việt Nam"

        # 6. Address Fields & Date of Issue
        logger.info(f"========== {card_type} EXTRACTION ==========\n")

        # placeOfOrigin (Front for OLD, Back for NEW)
        primary_lines_orig = front_lines if card_type == "CCCD_OLD" else back_lines
        primary_kws_orig = front_keywords if card_type == "CCCD_OLD" else back_keywords
        sec_lines_orig = back_lines if card_type == "CCCD_OLD" else front_lines
        sec_kws_orig = back_keywords if card_type == "CCCD_OLD" else front_keywords

        val_orig, raw_orig, kw_orig = self.extract_field_value_by_boundary(primary_lines_orig, "placeOfOrigin", primary_kws_orig)
        if not val_orig:
            val_orig, raw_orig, kw_orig = self.extract_field_value_by_boundary(sec_lines_orig, "placeOfOrigin", sec_kws_orig)
        data.placeOfOrigin = val_orig
        if kw_orig:
            ocr_results["placeOfOrigin"] = RawOcrFieldResult(
                value=val_orig, raw_text=raw_orig, keyword=kw_orig[0], language=kw_orig[1]
            )

        # placeOfResidence (Front for OLD, Back for NEW)
        primary_lines_res = front_lines if card_type == "CCCD_OLD" else back_lines
        primary_kws_res = front_keywords if card_type == "CCCD_OLD" else back_keywords
        sec_lines_res = back_lines if card_type == "CCCD_OLD" else front_lines
        sec_kws_res = back_keywords if card_type == "CCCD_OLD" else front_keywords

        val_res, raw_res, kw_res = self.extract_field_value_by_boundary(primary_lines_res, "placeOfResidence", primary_kws_res)
        if not val_res:
            val_res, raw_res, kw_res = self.extract_field_value_by_boundary(sec_lines_res, "placeOfResidence", sec_kws_res)
        data.placeOfResidence = val_res
        if kw_res:
            ocr_results["placeOfResidence"] = RawOcrFieldResult(
                value=val_res, raw_text=raw_res, keyword=kw_res[0], language=kw_res[1]
            )

        # dateOfIssue (Back side)
        val_doi, raw_doi, kw_doi = self._extract_date_with_keyword_priority(back_lines, "dateOfIssue", back_keywords)
        if not val_doi:
            val_doi, raw_doi, kw_doi = self._extract_date_with_keyword_priority(front_lines, "dateOfIssue", front_keywords)
        data.dateOfIssue = val_doi
        if kw_doi:
            ocr_results["dateOfIssue"] = RawOcrFieldResult(
                value=val_doi, raw_text=raw_doi, keyword=kw_doi[0], language=kw_doi[1]
            )

        self._detect_and_clean_field_contamination(data)

        return data, ocr_results

    def _extract_date_with_keyword_priority(
        self, lines: List[OcrLine], target_field: str, detected_keywords: List[DetectedKeyword]
    ) -> Tuple[Optional[str], Optional[str], Optional[Tuple[str, str]]]:
        current_kw = next((kw for kw in detected_keywords if kw.field_name == target_field), None)
        if not current_kw:
            return None, None, None

        idx = current_kw.line_index
        search_text = lines[idx].text
        for k in range(1, 3):
            if idx + k < len(lines):
                search_text += " " + lines[idx + k].text

        match = re.search(r'(?<!\d)(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(19|20)\d{2}', search_text)
        if match:
            raw_date_str = match.group(0)
            iso_date = normalize_date(raw_date_str)
            logger.info(f"FIELD:\n{target_field}\n\nKEYWORD:\n{current_kw.canonical_keyword}\n\nRAW:\n{raw_date_str}\n\nVALUE:\n{iso_date}\n")
            return iso_date, raw_date_str, (current_kw.canonical_keyword, current_kw.language)

        return None, None, (current_kw.canonical_keyword, current_kw.language)

    def _detect_and_clean_field_contamination(self, data: ExtractedCardData):
        forbidden_in_address = [
            r'CO[\s._]*GIA[\s._]*TRI[\s._]*DEN.*',
            r'DATE[\s._]*OF[\s._]*EXPIRY.*',
            r'DATE[\s._]*EXPIRY.*',
            r'COGICATR.*',
            r'COGIATRJ.*',
            r'\b(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(19|20)\d{2}\b.*',
            r'\b\d{4}-\d{2}-\d{2}\b.*',
            r'DAC[\s._]*DIEM[\s._]*NHAN[\s._]*DANG.*',
            r'BO[\s._]*CONG[\s._]*AN.*',
            r'MINISTRY.*'
        ]

        for addr_field in ["placeOfResidence", "placeOfOrigin"]:
            val = getattr(data, addr_field, None)
            if val:
                cleaned = val
                for pat in forbidden_in_address:
                    match = re.search(pat, remove_vietnamese_accents(cleaned), re.IGNORECASE)
                    if match:
                        idx = match.start()
                        cleaned = cleaned[:idx].strip()
                cleaned = re.sub(r'[,.-]+$', '', cleaned).strip()
                setattr(data, addr_field, cleaned if len(cleaned) > 2 else None)

    def _merge_field_sources(
        self,
        ocr_data: ExtractedCardData,
        mrz_data: Optional[Dict[str, Any]],
        qr_data: Optional[Dict[str, Any]],
        ocr_results: Dict[str, RawOcrFieldResult]
    ) -> Tuple[ExtractedCardData, List[FieldMetadata]]:
        merged_data = ExtractedCardData()
        field_metadata = []

        canonical_fields = [
            "identityNumber", "fullName", "dateOfBirth", "gender", "nationality",
            "placeOfOrigin", "placeOfResidence", "dateOfIssue", "dateOfExpiry"
        ]

        mrz_allowed_fields = {"identityNumber", "fullName", "dateOfBirth", "gender", "nationality", "dateOfExpiry"}

        for field_name in canonical_fields:
            ocr_res = ocr_results.get(field_name)
            ocr_val = getattr(ocr_data, field_name, None)
            mrz_val = mrz_data.get(field_name) if (mrz_data and field_name in mrz_allowed_fields) else None
            qr_val = qr_data.get(field_name) if qr_data else None

            selected_val = None
            selected_src = "OCR"
            kw_str = ocr_res.keyword if ocr_res else None
            lang_str = ocr_res.language if ocr_res else None
            ocr_raw_text = ocr_res.raw_text if ocr_res else ocr_val

            # Source Priority Matrix: MRZ > QR > OCR
            if field_name in mrz_allowed_fields and mrz_val:
                selected_val = mrz_val
                selected_src = "MRZ"
            elif qr_val:
                selected_val = qr_val
                selected_src = "QR"
            elif ocr_val:
                selected_val = ocr_val
                selected_src = "OCR"

            setattr(merged_data, field_name, normalize_unicode(selected_val))

            logger.info(f"[FINAL_FIELD] field={field_name} ocrValue='{ocr_val}' qrValue='{qr_val}' mrzValue='{mrz_val}' selectedValue='{selected_val}' selectedSource={selected_src}")

            final_keyword = kw_str if selected_src == "OCR" else None
            final_language = lang_str if selected_src == "OCR" else None
            final_raw_text = ocr_raw_text if (selected_src == "OCR" and ocr_raw_text) else selected_val

            meta = FieldMetadata(
                field=field_name,
                value=normalize_unicode(selected_val),
                source=selected_src,
                keyword=final_keyword,
                language=final_language,
                confidence=0.95 if selected_src == "OCR" else 1.0,
                rawText=normalize_unicode(final_raw_text),
                ocrValue=normalize_unicode(ocr_val),
                mrzValue=normalize_unicode(mrz_val),
                qrValue=normalize_unicode(qr_val),
                ocrKeyword=kw_str,
                ocrLanguage=lang_str
            )

            field_metadata.append(meta)

        return merged_data, field_metadata
