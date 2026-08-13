from typing import Tuple, Dict, Any, List, Optional
from pydantic import BaseModel
from ocr.detector import OCRText
from ocr.field_extractor import ExtractedField
from ocr.layout_parser import LayoutLine
from schemas.card import ExtractedCardData, CrossValidationResult, CrossValidationDetail, FieldMetadata
from utils.text_utils import compare_names, remove_vietnamese_accents
from utils.logger import logger
from config import settings


class CardTypeClassifier:
    """
    Card Type Classifier based on visual layout keywords, chip presence,
    document structure, MRZ, and QR indicators.
    """

    def classify(
        self,
        front_tokens: List[OCRText],
        back_tokens: List[OCRText],
        ocr_fields: Dict[str, ExtractedField]
    ) -> Tuple[str, float]:
        front_text = remove_vietnamese_accents(" ".join([t.text for t in front_tokens])).lower()
        back_text = remove_vietnamese_accents(" ".join([t.text for t in back_tokens])).lower()
        all_text = front_text + " " + back_text

        score_new = 0.0
        score_old = 0.0

        # NEW CARD Distinctive Indicators (Chip Card - 2021+)
        if "can cuoc" in front_text and "can cuoc cong dan" not in front_text:
            score_new += 0.40
        if "identity card" in front_text and "citizen identity card" not in front_text:
            score_new += 0.30
        if "so dinh danh ca nhan" in front_text or "personal identification" in front_text or "sadinh danh" in front_text:
            score_new += 0.30
        if "noi dang ky khai sinh" in all_text or "place of birth registration" in all_text or "roi dang ky khai sinh" in all_text:
            score_new += 0.35
        if "noi cu tru" in all_text:
            score_new += 0.25
        if "bo cong an" in back_text or "ministry of public security" in back_text:
            score_new += 0.20

        # OLD CARD Distinctive Indicators (CCCD 12-digit without chip / 9-digit CMND)
        if "can cuoc cong dan" in front_text:
            score_old += 0.40
        if "citizen identity card" in front_text:
            score_old += 0.30
        if "que quan" in front_text or "place of origin" in front_text:
            score_old += 0.35
        if "noi thuong tru" in front_text:
            score_old += 0.35

        logger.info(f"[CARD_CLASSIFIER] New Score: {score_new:.2f}, Old Score: {score_old:.2f}")

        if score_new > score_old and score_new >= 0.30:
            conf = 0.95 if score_new >= 0.60 else round(min(0.95, score_new + 0.30), 2)
            return "CCCD_NEW", conf
        elif score_old > score_new and score_old >= 0.30:
            conf = 0.95 if score_old >= 0.60 else round(min(0.95, score_old + 0.30), 2)
            return "CCCD_OLD", conf
        else:
            if not front_tokens and not back_tokens:
                return "UNKNOWN", 0.0
            c_type = "CCCD_NEW" if score_new >= score_old else "CCCD_OLD"
            return c_type, 0.70


class CrossValidator:
    """
    Multi-source Priority Merger & 3-Way Cross Validation Engine (OCR vs QR vs MRZ).
    Enforces field source priority matrices and detailed field status logging.
    """

    FIELD_PRIORITY = {
        "identityNumber": ["QR", "MRZ", "OCR"],
        "fullName": ["QR", "OCR", "MRZ"],
        "dateOfBirth": ["QR", "OCR", "MRZ"],
        "gender": ["QR", "OCR", "MRZ"],
        "nationality": ["QR", "OCR"],
        "placeOfOrigin": ["QR", "OCR"],
        "placeOfResidence": ["QR", "OCR"],
        "dateOfIssue": ["QR", "OCR"],
        "dateOfExpiry": ["QR", "OCR", "MRZ"],
    }

    def merge_and_validate(
        self,
        ocr_fields: Dict[str, ExtractedField],
        qr_data: Optional[Dict[str, Any]],
        mrz_data: Optional[Dict[str, Any]],
        card_type: str
    ) -> Tuple[bool, ExtractedCardData, CrossValidationResult, List[FieldMetadata], List[str]]:
        errors = []
        details = []
        field_metadata = []
        final_card_data = ExtractedCardData()

        canonical_fields = [
            "identityNumber", "fullName", "dateOfBirth", "gender", "nationality",
            "placeOfOrigin", "placeOfResidence", "dateOfIssue", "dateOfExpiry"
        ]

        mrz_allowed_fields = {"identityNumber", "fullName", "dateOfBirth", "gender", "nationality", "dateOfExpiry"}

        for field_name in canonical_fields:
            ocr_ext = ocr_fields.get(field_name)
            ocr_val = ocr_ext.value if ocr_ext else None
            qr_val = qr_data.get(field_name) if qr_data else None
            mrz_val = mrz_data.get(field_name) if (mrz_data and field_name in mrz_allowed_fields) else None

            # Source priority selection
            priority = self.FIELD_PRIORITY.get(field_name, ["QR", "OCR", "MRZ"])
            selected_val = None
            selected_src = "OCR"

            for src in priority:
                if src == "QR" and qr_val:
                    selected_val = qr_val
                    selected_src = "QR"
                    break
                elif src == "MRZ" and mrz_val:
                    selected_val = mrz_val
                    selected_src = "MRZ"
                    break
                elif src == "OCR" and ocr_val:
                    selected_val = ocr_val
                    selected_src = "OCR"
                    break

            setattr(final_card_data, field_name, selected_val)

            # Evaluate Field Status
            status = self._evaluate_field_status(ocr_val, qr_val, mrz_val, field_name)

            detail = CrossValidationDetail(
                fieldName=field_name,
                ocrValue=ocr_val,
                qrValue=qr_val,
                mrzValue=mrz_val,
                status=status
            )
            details.append(detail)

            if status in ["MISMATCH", "CONFLICT"]:
                errors.append(f"CARD_DATA_{status}_{field_name.upper()}")

            # Compute real field metadata confidence
            conf = 0.95 if selected_src == "OCR" else 1.0
            if ocr_val and mrz_val and ocr_val == mrz_val:
                conf = min(0.99, conf + 0.04)

            meta = FieldMetadata(
                field=field_name,
                value=selected_val,
                source=selected_src,
                keyword=ocr_ext.keyword if ocr_ext else None,
                language=ocr_ext.language if ocr_ext else None,
                confidence=round(conf, 2),
                rawText=ocr_ext.rawText if ocr_ext else selected_val,
                ocrValue=ocr_val,
                mrzValue=mrz_val,
                qrValue=qr_val,
                ocrKeyword=ocr_ext.keyword if ocr_ext else None,
                ocrLanguage=ocr_ext.language if ocr_ext else None
            )
            field_metadata.append(meta)

            logger.info(
                f"[CROSS_VAL] field={field_name} ocr='{ocr_val}' qr='{qr_val}' mrz='{mrz_val}' -> status={status} selected='{selected_val}' ({selected_src})"
            )

        # Global Pairwise Flags
        ocr_match_qr = self._evaluate_pairwise(ocr_fields, qr_data)
        ocr_match_mrz = self._evaluate_pairwise(ocr_fields, mrz_data)

        mrz_valid = mrz_data.get("isMrzValid", True) if mrz_data else True

        cross_val_result = CrossValidationResult(
            ocrMatchQr=ocr_match_qr,
            ocrMatchMrz=ocr_match_mrz,
            mrzCheckDigitValid=mrz_valid,
            isExpired=False,
            details=details
        )

        card_verified = (
            card_type != "UNKNOWN"
            and mrz_valid
            and final_card_data.identityNumber is not None
            and final_card_data.fullName is not None
            and ocr_match_qr is not False
            and ocr_match_mrz is not False
        )

        return card_verified, final_card_data, cross_val_result, field_metadata, errors

    def _evaluate_field_status(
        self, ocr_val: Optional[str], qr_val: Optional[str], mrz_val: Optional[str], field_name: str
    ) -> str:
        vals = {"OCR": ocr_val, "QR": qr_val, "MRZ": mrz_val}
        non_nulls = {k: v for k, v in vals.items() if v is not None}

        if len(non_nulls) == 0:
            return "MISSING"
        elif len(non_nulls) == 1:
            k = list(non_nulls.keys())[0]
            return f"{k}_ONLY"

        val_list = list(non_nulls.values())
        if field_name == "fullName":
            for i in range(len(val_list)):
                for j in range(i + 1, len(val_list)):
                    if compare_names(val_list[i], val_list[j]) < settings.FULLNAME_FUZZY_THRESHOLD:
                        return "CONFLICT" if field_name == "gender" else "MISMATCH"
            return "MATCH"
        else:
            for i in range(len(val_list)):
                for j in range(i + 1, len(val_list)):
                    if val_list[i] != val_list[j]:
                        return "CONFLICT" if field_name == "gender" else "MISMATCH"
            return "MATCH"

    def _evaluate_pairwise(self, ocr_fields: Dict[str, ExtractedField], other_data: Optional[Dict[str, Any]]) -> Optional[bool]:
        if not other_data:
            return None

        compared = False
        for fname in ["identityNumber", "fullName", "dateOfBirth", "dateOfExpiry"]:
            ocr_ext = ocr_fields.get(fname)
            v1 = ocr_ext.value if ocr_ext else None
            v2 = other_data.get(fname)

            if v1 is not None and v2 is not None:
                compared = True
                if fname == "fullName":
                    if compare_names(v1, v2) < settings.FULLNAME_FUZZY_THRESHOLD:
                        return False
                else:
                    if v1 != v2:
                        return False

        return True if compared else None
