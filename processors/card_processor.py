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
    """
    Normalizes string to Unicode NFC format.
    Preserves all Vietnamese accents and characters.
    """
    if not text:
        return text
    return unicodedata.normalize("NFC", text)


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
    """
    Unified Card Processor with Keyword Bounding Box Field Boundary Extraction,
    Independent Field Metadata State (Zero cross-field pollution),
    Full Unicode Vietnamese Accent Preservation (NFC),
    and Fallback Image Side Extraction (prevents placeOfOrigin/placeOfResidence/dateOfIssue from turning null).
    """

    def __init__(self, ocr_engine: OcrEngine, qr_engine: QrEngine, mrz_engine: MrzEngine):
        self.ocr_engine = ocr_engine
        self.qr_engine = qr_engine
        self.mrz_engine = mrz_engine

    def process(
        self, front_image: np.ndarray, back_image: np.ndarray
    ) -> Tuple[str, float, ExtractedCardData, Optional[Dict[str, Any]], Optional[Dict[str, Any]], QualityChecks, List[FieldMetadata]]:
        """
        Executes pipeline: front OCR -> back OCR -> MRZ -> QR -> boundary extraction -> merge sources -> metadata.
        """
        is_blur_f, has_glare_f = check_image_quality(front_image)
        quality_checks = QualityChecks(isBlur=is_blur_f, hasGlare=has_glare_f, isCropped=False)

        # 1. Front OCR & Box Logging (Normalize Unicode NFC)
        raw_front_lines = self.ocr_engine.detect_and_recognize(front_image) if self.ocr_engine else []
        front_lines = []
        for line in raw_front_lines:
            nfc_text = normalize_unicode(line.text)
            front_lines.append(OcrLine(text=nfc_text, confidence=line.confidence, boundingBox=line.boundingBox))
        front_lines = self._sort_lines_spatially(front_lines)

        # 2. Back OCR & Box Logging (Normalize Unicode NFC)
        raw_back_lines = self.ocr_engine.detect_and_recognize(back_image) if self.ocr_engine else []
        back_lines = []
        for line in raw_back_lines:
            nfc_text = normalize_unicode(line.text)
            back_lines.append(OcrLine(text=nfc_text, confidence=line.confidence, boundingBox=line.boundingBox))
        back_lines = self._sort_lines_spatially(back_lines)

        logger.info(f"[OCR_RAW] front='{' | '.join([l.text for l in front_lines])}'")
        logger.info(f"[OCR_RAW] back='{' | '.join([l.text for l in back_lines])}'")

        # 3. QR Parser (Front or Back)
        qr_data = self.qr_engine.decode(front_image) if self.qr_engine else None
        if not qr_data and self.qr_engine:
            qr_data = self.qr_engine.decode(back_image)

        # 4. MRZ Parser (Back side only)
        back_texts = [line.text for line in back_lines]
        mrz_data = self.mrz_engine.parse(back_texts) if self.mrz_engine else None

        # 5. Detect Card Type & Calculate Confidence Score
        front_texts = [line.text for line in front_lines]
        card_type, card_type_confidence, indicators = self._detect_card_type(
            front_texts + back_texts,
            has_mrz=mrz_data is not None,
            has_qr_front=qr_data is not None
        )
        logger.info(f"[CARD_TYPE] type={card_type} confidence={card_type_confidence:.2f} indicators={indicators}")

        # 6. Extract raw OCR fields using Keyword Boundary Extraction Engine
        ocr_extracted_data, ocr_field_results = self._extract_raw_ocr_fields(card_type, front_lines, back_lines)

        # 7. Merge Field Sources (MRZ > QR > OCR per Source Priority Matrix)
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
        """
        Scans all OCR lines and detects ALL keywords with noisy OCR variants.
        Uses normalizedForMatching internally ONLY for matching.
        Returns list of DetectedKeyword.
        Logs: [OCR_KEYWORD] and [KEYWORDS]
        """
        detected_keywords = []

        for idx, line in enumerate(lines):
            line_text = line.text.strip()
            clean_text = remove_vietnamese_accents(line_text).lower()
            clean_text_dots = re.sub(r'[\/._\-]+', ' ', clean_text).strip()

            matched_field = None
            raw_kw = None
            canonical_kw = None
            lang = None

            # Step 1: Check English Keywords
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

            # Step 2: Check Vietnamese Keywords if not matched
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
                logger.info(f"[OCR_KEYWORD] rawText='{dkw.raw_text}' normalizedText='{clean_text_dots}' field={dkw.field_name} keyword='{dkw.canonical_keyword}' language={dkw.language} bbox={dkw.bbox} confidence={line.confidence:.2f}")
                logger.info(f"[KEYWORDS] field={dkw.field_name} keyword='{dkw.canonical_keyword}' bbox={dkw.bbox}")

        return detected_keywords

    def extract_field_value_by_boundary(
        self,
        lines: List[OcrLine],
        target_field: str,
        all_detected_keywords: List[DetectedKeyword]
    ) -> Tuple[Optional[str], Optional[str], Optional[Tuple[str, str]]]:
        """
        Keyword Bounding Box Field Boundary Extraction Algorithm:
        Returns: (normalized_value, raw_text, (canonical_keyword, language))
        Logs: [BOUNDARY] and [EXTRACT]
        """
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

        start_bbox = current_kw.bbox
        next_bbox = next_kw.bbox if next_kw else None
        next_kw_name = next_kw.canonical_keyword if next_kw else "EOF"
        next_line_idx = next_kw.line_index if next_kw else len(lines)

        logger.info(f"[BOUNDARY] field={target_field} start='{current_kw.canonical_keyword}' stop='{next_kw_name}'")

        gathered_boxes = []

        start_line_text = lines[current_kw.line_index].text.strip()
        clean_inline = start_line_text
        clean_inline = re.sub(r'^[\\\/._\s]+', '', clean_inline)

        # Remove matched keyword text from inline string while preserving Unicode value
        for pattern_str in [current_kw.canonical_keyword, current_kw.raw_text, "place of origin", "piace of origin", "place of residence", "noi thuong tru", "que quan", "queguan", "quê quán", "nơi thường trú", "nơi cư trú"]:
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
                if re.search(r'cogiatr|date[\s._]*of[\s._]*expiry|date[\s._]*expiry', unaccented_j):
                    continue
                if re.search(r'(?<!\d)(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(19|20)\d{2}', line_text) and lines[j].boundingBox and min(pt[0] for pt in lines[j].boundingBox) < 250:
                    continue

            gathered_boxes.append(line_text)

        if gathered_boxes:
            cleaned_boxes = [b.strip() for b in gathered_boxes if b.strip()]
            
            # Format multiline address boxes with commas if not present
            formatted_parts = []
            for b in cleaned_boxes:
                if formatted_parts and not formatted_parts[-1].endswith(','):
                    formatted_parts.append(', ' + b)
                elif formatted_parts:
                    formatted_parts.append(' ' + b)
                else:
                    formatted_parts.append(b)

            raw_text = "".join(formatted_parts)
            raw_text_clean = normalize_unicode(raw_text)

            # OCR Error correction: Chäu -> Châu (preserving Vietnamese accent!)
            raw_text = raw_text.replace('Chäu', 'Châu').replace('chäu', 'châu')

            # Clean duplicate commas/spaces
            normalized_val = normalize_unicode(re.sub(r'\s*,\s*', ', ', raw_text)).strip()

            logger.info(f"[EXTRACT] field={target_field} rawText='{raw_text_clean}' value='{normalized_val}'")
            return normalized_val, raw_text_clean, (current_kw.canonical_keyword, current_kw.language)

        logger.info(f"[EXTRACT] field={target_field} rawText=None value=None")
        return None, None, (current_kw.canonical_keyword, current_kw.language)

    def _detect_card_type(
        self, all_texts: List[str], has_mrz: bool = False, has_qr_front: bool = False
    ) -> Tuple[str, float, List[str]]:
        full_text = remove_vietnamese_accents(" ".join(all_texts)).lower()

        score_new = 0.0
        score_old = 0.0
        indicators = []

        if "can cuoc cong dan" in full_text:
            score_old += 0.25
            indicators.append("CĂN CƯỚC CÔNG DÂN")
        if "citizen identity card" in full_text:
            score_old += 0.25
            indicators.append("Citizen Identity Card")

        if "can cuoc" in full_text and "cong dan" not in full_text:
            score_new += 0.25
            indicators.append("CĂN CƯỚC")
        if "identity card" in full_text and "citizen" not in full_text:
            score_new += 0.25
            indicators.append("Identity Card")

        if "que quan" in full_text or "place of origin" in full_text or "queguan" in full_text or "piace of origin" in full_text:
            score_old += 0.15
            indicators.append("Quê quán / Place of origin")

        if "noi dang ky khai sinh" in full_text or "place of birth registration" in full_text:
            score_new += 0.15
            indicators.append("Nơi đăng ký khai sinh / Place of birth registration")

        if "noi thuong tru" in full_text:
            score_old += 0.15
            indicators.append("Nơi thường trú")
        if "place of residence" in full_text or "noi cu tru" in full_text:
            score_old += 0.10
            score_new += 0.10
            indicators.append("Nơi cư trú / Place of Residence")

        if "co gia tri den" in full_text or "date of expiry" in full_text or "cogiatr" in full_text:
            score_old += 0.10
            score_new += 0.10
            indicators.append("Có giá trị đến / Date of expiry")

        if "so dinh danh ca nhan" in full_text or "personal identification number" in full_text:
            score_new += 0.15
            indicators.append("Số định danh cá nhân / Personal identification number")

        if has_mrz:
            score_old += 0.15
            indicators.append("MRZ TD1")

        if score_old > score_new and score_old >= 0.25:
            conf = min(1.0, score_old + 0.30)
            return "CCCD_OLD", round(conf, 2), indicators
        elif score_new > score_old and score_new >= 0.25:
            conf = min(1.0, score_new + 0.30)
            return "CCCD_NEW", round(conf, 2), indicators
        elif score_old > 0 or score_new > 0 or has_mrz:
            c_type = "CCCD_OLD" if score_old >= score_new else "CCCD_NEW"
            conf = min(0.70, max(score_old, score_new) + 0.30)
            return c_type, round(conf, 2), indicators
        else:
            return "UNKNOWN", 0.0, indicators

    def _extract_raw_ocr_fields(
        self, card_type: str, front_lines: List[OcrLine], back_lines: List[OcrLine]
    ) -> Tuple[ExtractedCardData, Dict[str, RawOcrFieldResult]]:
        """
        Pipeline: detect ALL keywords -> extract field regions by boundary -> normalize field values.
        Stores independent RawOcrFieldResult per field to prevent cross-field metadata pollution.
        Implements fallback image side search (prevents placeOfOrigin/placeOfResidence/dateOfIssue from turning null).
        """
        data = ExtractedCardData()
        ocr_results: Dict[str, RawOcrFieldResult] = {}

        front_keywords = self.detect_all_keywords(front_lines)
        back_keywords = self.detect_all_keywords(back_lines)

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

        # 4. Date of Expiry (Front side)
        val_exp, raw_exp, kw_exp = self._extract_date_with_keyword_priority(front_lines, "dateOfExpiry", front_keywords)
        data.dateOfExpiry = val_exp
        if kw_exp:
            ocr_results["dateOfExpiry"] = RawOcrFieldResult(
                value=val_exp, raw_text=raw_exp, keyword=kw_exp[0], language=kw_exp[1]
            )

        # 5. Gender & Nationality (Full Unicode)
        if "Nam" in all_front_text or "Male" in all_front_text:
            data.gender = "Nam"
        elif "Nữ" in all_front_text or "Female" in all_front_text:
            data.gender = "Nữ"
        data.nationality = "Việt Nam"

        # 6. Address Fields & Date of Issue (with Robust Fallback to other image side)
        # placeOfOrigin
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

        # placeOfResidence
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

        # dateOfIssue
        val_doi, raw_doi, kw_doi = self._extract_date_with_keyword_priority(back_lines, "dateOfIssue", back_keywords)
        if not val_doi:
            val_doi, raw_doi, kw_doi = self._extract_date_with_keyword_priority(front_lines, "dateOfIssue", front_keywords)
        data.dateOfIssue = val_doi
        if kw_doi:
            ocr_results["dateOfIssue"] = RawOcrFieldResult(
                value=val_doi, raw_text=raw_doi, keyword=kw_doi[0], language=kw_doi[1]
            )

        # Apply Field Contamination Cleaning
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
            logger.info(f"[EXTRACT] field={target_field} rawText='{raw_date_str}' value='{iso_date}'")
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
            r'DAC[\s._]*DIEM[\s._]*NHAN[\s._]*DANG.*'
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
        """
        Merges field candidates with strict independent FieldMetadata objects.
        Prevents shared object mutation and cross-field rawText pollution.
        Logs: [VALIDATION] and [METADATA]
        """
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

            # Apply Source Priority Selection Matrix
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

            logger.info(f"[VALIDATION] field={field_name} ocrValue='{ocr_val}' qrValue='{qr_val}' mrzValue='{mrz_val}' selectedValue='{selected_val}' selectedSource={selected_src}")

            # Construct INDEPENDENT Field Metadata object (Zero shared state mutation)
            if selected_val is None:
                conf = 0.3 if kw_str else 0.0
                meta = FieldMetadata(
                    field=field_name,
                    value=None,
                    source="OCR",
                    keyword=kw_str if conf == 0.3 else None,
                    language=lang_str if conf == 0.3 else None,
                    confidence=conf,
                    rawText=None
                )
            else:
                # rawText is strictly the raw OCR string for OCR source or exact selected value
                final_raw_text = ocr_raw_text if (selected_src == "OCR" and ocr_raw_text) else selected_val
                meta = FieldMetadata(
                    field=field_name,
                    value=normalize_unicode(selected_val),
                    source=selected_src,
                    keyword=kw_str,
                    language=lang_str,
                    confidence=0.95 if selected_src == "OCR" else 1.0,
                    rawText=normalize_unicode(final_raw_text)
                )

            field_metadata.append(meta)
            logger.info(f"[METADATA] field={field_name} source={meta.source} keyword='{meta.keyword}' rawText='{meta.rawText}' value='{meta.value}' confidence={meta.confidence:.2f}")

        return merged_data, field_metadata
