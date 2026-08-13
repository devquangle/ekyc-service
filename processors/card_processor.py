import re
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from core.ocr_engine import OcrEngine, OcrLine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from schemas.card import ExtractedCardData, QualityChecks
from utils.image_utils import check_image_quality
from utils.text_utils import normalize_text, normalize_date, remove_vietnamese_accents
from utils.logger import logger


class CardProcessor:
    """
    Unified Card Processor for extracting data from front.jpg and back.jpg of Vietnamese ID Cards.
    Calculates Card Type Classification Confidence and applies Fallback hierarchy (OCR -> MRZ -> QR).
    """

    def __init__(self, ocr_engine: OcrEngine, qr_engine: QrEngine, mrz_engine: MrzEngine):
        self.ocr_engine = ocr_engine
        self.qr_engine = qr_engine
        self.mrz_engine = mrz_engine

    def process(
        self, front_image: np.ndarray, back_image: np.ndarray
    ) -> Tuple[str, float, ExtractedCardData, Optional[Dict[str, Any]], Optional[Dict[str, Any]], QualityChecks]:
        """
        Executes OCR, QR decoding (front & back), and MRZ parsing (back).
        Returns: (card_type, card_type_confidence, extracted_data, qr_data, mrz_data, quality_checks)
        """
        # Quality check on front image
        is_blur_f, has_glare_f = check_image_quality(front_image)
        quality_checks = QualityChecks(isBlur=is_blur_f, hasGlare=has_glare_f, isCropped=False)

        # 1. OCR Front and Back
        front_lines = self.ocr_engine.detect_and_recognize(front_image)
        back_lines = self.ocr_engine.detect_and_recognize(back_image)

        front_texts = [line.text for line in front_lines]
        back_texts = [line.text for line in back_lines]

        # 2. Decode QR Code (Try both front and back)
        qr_data = self.qr_engine.decode(front_image)
        if not qr_data:
            qr_data = self.qr_engine.decode(back_image)

        # 3. Parse MRZ Code (Back side)
        mrz_data = self.mrz_engine.parse(back_texts)

        # 4. Detect Card Type using Scoring & Confidence
        card_type, card_type_confidence = self._detect_card_type(
            front_texts + back_texts,
            has_mrz=mrz_data is not None,
            has_qr_front=qr_data is not None
        )

        # 5. Build Extracted Data from OCR
        extracted_data = self._extract_ocr_fields(card_type, front_lines, back_lines)

        # 6. Apply Fallback Hierarchy for Fields (OCR -> MRZ -> QR)
        # Full Name Fallback
        if not extracted_data.fullName:
            if mrz_data and mrz_data.get("fullName"):
                extracted_data.fullName = mrz_data.get("fullName")
            elif qr_data and qr_data.get("fullName"):
                extracted_data.fullName = qr_data.get("fullName")

        # Identity Number Fallback
        if not extracted_data.identityNumber:
            if qr_data and qr_data.get("identityNumber"):
                extracted_data.identityNumber = qr_data.get("identityNumber")
            elif mrz_data and mrz_data.get("identityNumber"):
                extracted_data.identityNumber = mrz_data.get("identityNumber")

        # Date of Birth Fallback
        if not extracted_data.dateOfBirth:
            if mrz_data and mrz_data.get("dateOfBirth"):
                extracted_data.dateOfBirth = mrz_data.get("dateOfBirth")
            elif qr_data and qr_data.get("dateOfBirth"):
                extracted_data.dateOfBirth = qr_data.get("dateOfBirth")

        # Gender Fallback
        if not extracted_data.gender:
            if mrz_data and mrz_data.get("gender"):
                extracted_data.gender = mrz_data.get("gender")
            elif qr_data and qr_data.get("gender"):
                extracted_data.gender = qr_data.get("gender")

        # Date of Expiry Fallback
        if not extracted_data.dateOfExpiry:
            if mrz_data and mrz_data.get("dateOfExpiry"):
                extracted_data.dateOfExpiry = mrz_data.get("dateOfExpiry")

        # Date of Issue Fallback
        if not extracted_data.dateOfIssue:
            if qr_data and qr_data.get("dateOfIssue"):
                extracted_data.dateOfIssue = qr_data.get("dateOfIssue")

        # Residence Fallback
        if not extracted_data.placeOfResidence:
            if qr_data and qr_data.get("placeOfResidence"):
                extracted_data.placeOfResidence = qr_data.get("placeOfResidence")

        return card_type, card_type_confidence, extracted_data, qr_data, mrz_data, quality_checks

    def _detect_card_type(self, all_texts: List[str], has_mrz: bool = False, has_qr_front: bool = False) -> Tuple[str, float]:
        """
        Calculates keyword signature score to classify card type into CCCD_NEW or CCCD_OLD.
        Returns: (card_type, confidence_score)
        """
        full_text = remove_vietnamese_accents(" ".join(all_texts))

        score_new = 0.0
        score_old = 0.0
        max_possible = 10.0

        if "CAN CUOC" in full_text and "CONG DAN" not in full_text:
            score_new += 3.0
        if "CAN CUOC CONG DAN" in full_text:
            score_old += 3.0

        if "SO DINH DANH CA NHAN" in full_text:
            score_new += 2.0
        if "SO / NO" in full_text or "SO:" in full_text or "CITIZEN IDENTITY CARD" in full_text:
            score_old += 2.0

        if "NOI DANG KY KHAI SINH" in full_text:
            score_new += 2.0
        if "QUE QUAN" in full_text or "PLACE OF ORIGIN" in full_text:
            score_old += 2.0

        if "NOI CU TRU" in full_text:
            score_new += 2.0
        if "NOI THUONG TRU" in full_text or "PLACE OF RESIDENCE" in full_text:
            score_old += 2.0

        if "HO, CHU DEM VA TEN KHAI SINH" in full_text:
            score_new += 2.0
        if "HO VA TEN" in full_text or "FULL NAME" in full_text:
            score_old += 2.0

        # Secondary evidence signals
        if has_mrz:
            score_old += 1.0
            score_new += 1.0

        if score_new > score_old and score_new >= 2.0:
            confidence = min(1.0, score_new / max_possible)
            return "CCCD_NEW", round(confidence, 2)
        elif score_old > score_new and score_old >= 2.0:
            confidence = min(1.0, score_old / max_possible)
            return "CCCD_OLD", round(confidence, 2)
        elif score_old > 0 or score_new > 0 or has_mrz:
            # Moderate evidence fallback
            c_type = "CCCD_OLD" if score_old >= score_new else "CCCD_NEW"
            conf = min(0.70, (max(score_old, score_new) + 2.0) / max_possible)
            return c_type, round(conf, 2)
        else:
            return "UNKNOWN", 0.0

    def _extract_ocr_fields(
        self, card_type: str, front_lines: List[OcrLine], back_lines: List[OcrLine]
    ) -> ExtractedCardData:
        """
        Extracts card fields based on card type spatial label alignment and regex parsing.
        """
        data = ExtractedCardData()
        all_front_text = "\n".join([line.text for line in front_lines])
        all_back_text = "\n".join([line.text for line in back_lines])

        # 1. Identity Number (12 digits starting with 0)
        id_match = re.search(r'\b0\d{11}\b', all_front_text)
        if id_match:
            data.identityNumber = id_match.group(0)

        # 2. Date of Birth, Date of Issue, Date of Expiry (Regex DD/MM/YYYY)
        dates = re.findall(r'\b(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/(19|20)\d{2}\b', all_front_text + "\n" + all_back_text)
        formatted_dates = []
        for d in dates:
            formatted = normalize_date(f"{d[0]}/{d[1]}/{d[2]}")
            if formatted:
                formatted_dates.append(formatted)

        if formatted_dates:
            data.dateOfBirth = formatted_dates[0]
            if len(formatted_dates) > 1:
                data.dateOfExpiry = formatted_dates[1]
            if len(formatted_dates) > 2:
                data.dateOfIssue = formatted_dates[2]

        # 3. Gender & Nationality
        if "Nam" in all_front_text:
            data.gender = "Nam"
        elif "Nữ" in all_front_text:
            data.gender = "Nữ"
        data.nationality = "Việt Nam"

        # 4. Full Name (Find line after Full Name label)
        for idx, line in enumerate(front_lines):
            txt = line.text.upper()
            if "HỌ VÀ TÊN" in txt or "HỌ, CHỮ ĐỆM" in txt:
                if idx + 1 < len(front_lines):
                    next_txt = front_lines[idx + 1].text.strip()
                    if next_txt.isupper() and len(next_txt) > 3:
                        data.fullName = normalize_text(next_txt)
                        break

        # 5. Address Parsing (Place of origin, place of birth, place of residence)
        if card_type == "CCCD_OLD":
            data.placeOfOrigin = self._find_field_after_label(front_lines, ["QUÊ QUÁN", "PLACE OF ORIGIN"])
            data.placeOfResidence = self._find_field_after_label(front_lines, ["NƠI THƯỜNG TRÚ", "PLACE OF RESIDENCE"])
        else:  # CCCD_NEW
            data.placeOfBirth = self._find_field_after_label(back_lines, ["NƠI ĐĂNG KÝ KHAI SINH", "PLACE OF BIRTH"])
            data.placeOfResidence = self._find_field_after_label(back_lines, ["NƠI CƯ TRÚ", "PLACE OF RESIDENCE"])

        return data

    def _find_field_after_label(self, lines: List[OcrLine], labels: List[str]) -> Optional[str]:
        for idx, line in enumerate(lines):
            txt = line.text.upper()
            if any(lbl in txt for lbl in labels):
                value_parts = []
                for j in range(idx + 1, min(idx + 3, len(lines))):
                    next_txt = lines[j].text.strip()
                    if any(kw in next_txt.upper() for kw in ["NGÀY", "BỘ CÔNG AN", "CỤC TRƯỜNG", "SIGNATURE"]):
                        break
                    value_parts.append(next_txt)
                if value_parts:
                    return normalize_text(" ".join(value_parts))
        return None
