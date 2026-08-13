from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from config import settings
from schemas.card import ExtractedCardData, CrossValidationResult, CrossValidationDetail
from utils.text_utils import compare_names
from utils.logger import logger


class CardValidator:
    """
    Card Validator Engine performing 3-way Cross-Validation (OCR ↔ QR ↔ MRZ)
    with strict status evaluation (MATCH, MISMATCH, NOT_AVAILABLE).
    """

    def validate(
        self,
        ocr_data: ExtractedCardData,
        qr_data: Optional[Dict[str, Any]],
        mrz_data: Optional[Dict[str, Any]],
        card_type: str = "UNKNOWN"
    ) -> Tuple[bool, CrossValidationResult, List[str]]:
        """
        Executes cross validation matrix audit and card expiry check.
        Returns: (card_verified, cross_val_result, errors)
        """
        details: List[CrossValidationDetail] = []
        errors: List[str] = []

        # 1. Check Identity Number (OCR vs QR vs MRZ)
        id_ocr = ocr_data.identityNumber
        id_qr = qr_data.get("identityNumber") if qr_data else None
        id_mrz = mrz_data.get("identityNumber") if mrz_data else None

        id_status = self._evaluate_field_status(id_ocr, id_qr, id_mrz)
        details.append(CrossValidationDetail(
            fieldName="identityNumber",
            ocrValue=id_ocr,
            qrValue=id_qr,
            mrzValue=id_mrz,
            status=id_status
        ))

        # 2. Check Full Name (OCR vs QR vs MRZ)
        name_ocr = ocr_data.fullName
        name_qr = qr_data.get("fullName") if qr_data else None
        name_mrz = mrz_data.get("fullName") if mrz_data else None

        name_status = self._evaluate_fuzzy_name_status(name_ocr, name_qr, name_mrz)
        details.append(CrossValidationDetail(
            fieldName="fullName",
            ocrValue=name_ocr,
            qrValue=name_qr,
            mrzValue=name_mrz,
            status=name_status
        ))

        # 3. Check Date of Birth
        dob_ocr = ocr_data.dateOfBirth
        dob_qr = qr_data.get("dateOfBirth") if qr_data else None
        dob_mrz = mrz_data.get("dateOfBirth") if mrz_data else None

        dob_status = self._evaluate_field_status(dob_ocr, dob_qr, dob_mrz)
        details.append(CrossValidationDetail(
            fieldName="dateOfBirth",
            ocrValue=dob_ocr,
            qrValue=dob_qr,
            mrzValue=dob_mrz,
            status=dob_status
        ))

        # 4. Check Date of Expiry (OCR vs MRZ)
        exp_ocr = ocr_data.dateOfExpiry
        exp_mrz = mrz_data.get("dateOfExpiry") if mrz_data else None

        exp_status = self._evaluate_field_status(exp_ocr, None, exp_mrz)
        details.append(CrossValidationDetail(
            fieldName="dateOfExpiry",
            ocrValue=exp_ocr,
            qrValue=None,
            mrzValue=exp_mrz,
            status=exp_status
        ))

        # --- Evaluate pairwise booleans: ocrMatchQr & ocrMatchMrz ---
        ocr_match_qr = self._evaluate_pairwise_boolean(id_ocr, id_qr, name_ocr, name_qr, dob_ocr, dob_qr)
        ocr_match_mrz = self._evaluate_pairwise_boolean(id_ocr, id_mrz, name_ocr, name_mrz, dob_ocr, dob_mrz, exp_ocr, exp_mrz)

        # Collect critical mismatch errors
        if id_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_IDENTITY_NUMBER")
        if name_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_FULL_NAME")
        if dob_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_DATE_OF_BIRTH")
        if exp_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_DATE_OF_EXPIRY")

        # 5. Check Expiry Logic (dateOfExpiry >= current_date)
        is_expired = False
        target_expiry = exp_ocr or exp_mrz
        if target_expiry:
            try:
                exp_date = datetime.strptime(target_expiry, "%Y-%m-%d").date()
                if exp_date < datetime.now().date():
                    is_expired = True
                    errors.append("CARD_EXPIRED")
            except Exception:
                pass

        # 6. Check MRZ Check Digits
        mrz_check_digit_valid = True
        if mrz_data and not mrz_data.get("isMrzValid", True):
            mrz_check_digit_valid = False
            errors.append("MRZ_CHECK_DIGIT_FAILED")

        # 7. Core field completeness check
        if not id_ocr and not id_qr and not id_mrz:
            errors.append("MISSING_IDENTITY_NUMBER")
        if not name_ocr and not name_qr and not (mrz_data and mrz_data.get("fullName")):
            errors.append("MISSING_FULL_NAME")

        if card_type == "UNKNOWN":
            errors.append("UNKNOWN_CARD_TYPE")

        cross_val_result = CrossValidationResult(
            ocrMatchQr=ocr_match_qr,
            ocrMatchMrz=ocr_match_mrz,
            mrzCheckDigitValid=mrz_check_digit_valid,
            isExpired=is_expired,
            details=details
        )

        # cardVerified requires NO critical mismatches, valid ID, valid MRZ, cardType known, not expired
        card_verified = (
            card_type != "UNKNOWN"
            and not is_expired
            and mrz_check_digit_valid
            and (ocr_match_qr is not False)
            and (ocr_match_mrz is not False)
            and len(errors) == 0
        )

        return card_verified, cross_val_result, errors

    def _evaluate_field_status(self, v1: Optional[str], v2: Optional[str], v3: Optional[str]) -> str:
        """
        Evaluates cross-validation status for exact match fields.
        Rules:
        - If 2 or more values exist: if equal -> MATCH, else MISMATCH.
        - If 0 or 1 value exists -> NOT_AVAILABLE.
        """
        values = [v for v in [v1, v2, v3] if v is not None]
        if len(values) < 2:
            return "NOT_AVAILABLE"

        first = values[0]
        for val in values[1:]:
            if val != first:
                return "MISMATCH"
        return "MATCH"

    def _evaluate_fuzzy_name_status(self, n1: Optional[str], n2: Optional[str], n3: Optional[str]) -> str:
        """
        Evaluates cross-validation status for fuzzy name matching.
        """
        names = [n for n in [n1, n2, n3] if n is not None]
        if len(names) < 2:
            return "NOT_AVAILABLE"

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                sim = compare_names(names[i], names[j])
                if sim < settings.FULLNAME_FUZZY_THRESHOLD:
                    return "MISMATCH"
        return "MATCH"

    def _evaluate_pairwise_boolean(self, *pairs) -> Optional[bool]:
        """
        Evaluates boolean pairwise match between two sources.
        Returns:
        - True if at least one comparison was made and all compared pairs matched.
        - False if any comparison was a mismatch.
        - None if no pairs were available to compare (NOT_AVAILABLE).
        """
        has_comparison = False
        # Iterate in pairs (v1, v2)
        for i in range(0, len(pairs) - 1, 2):
            val1 = pairs[i]
            val2 = pairs[i + 1]
            if val1 is not None and val2 is not None:
                has_comparison = True
                if val1 != val2 and compare_names(str(val1), str(val2)) < settings.FULLNAME_FUZZY_THRESHOLD:
                    return False

        if not has_comparison:
            return None
        return True
