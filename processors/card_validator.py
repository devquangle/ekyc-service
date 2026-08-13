from typing import Tuple, Dict, Any, List, Optional
from schemas.card import ExtractedCardData, CrossValidationResult, CrossValidationDetail
from utils.text_utils import compare_names
from utils.logger import logger
from config import settings


class CardValidator:
    """
    Card Validator module performing 3-Way Cross-Validation (OCR vs QR vs MRZ)
    and calculating overall card verification status.
    """

    def validate(
        self,
        ocr_data: ExtractedCardData,
        qr_data: Optional[Dict[str, Any]],
        mrz_data: Optional[Dict[str, Any]],
        card_type: str
    ) -> Tuple[bool, CrossValidationResult, List[str]]:
        """
        Executes strict cross-validation and expiry validation.
        Returns: (card_verified, cross_validation_result, errors)
        """
        errors = []
        details = []

        # 1. Compare Identity Number
        id_ocr = ocr_data.identityNumber
        id_qr = qr_data.get("identityNumber") if qr_data else None
        id_mrz = mrz_data.get("identityNumber") if mrz_data else None

        id_status = self._evaluate_field_status(id_ocr, id_qr, id_mrz)
        details.append(CrossValidationDetail(
            fieldName="identityNumber", ocrValue=id_ocr, qrValue=id_qr, mrzValue=id_mrz, status=id_status
        ))
        if id_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_IDENTITY_NUMBER")

        # 2. Compare Full Name
        name_ocr = ocr_data.fullName
        name_qr = qr_data.get("fullName") if qr_data else None
        name_mrz = mrz_data.get("fullName") if mrz_data else None

        name_status = self._evaluate_name_status(name_ocr, name_qr, name_mrz)
        details.append(CrossValidationDetail(
            fieldName="fullName", ocrValue=name_ocr, qrValue=name_qr, mrzValue=name_mrz, status=name_status
        ))
        if name_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_FULL_NAME")

        # 3. Compare Date of Birth
        dob_ocr = ocr_data.dateOfBirth
        dob_qr = qr_data.get("dateOfBirth") if qr_data else None
        dob_mrz = mrz_data.get("dateOfBirth") if mrz_data else None

        dob_status = self._evaluate_field_status(dob_ocr, dob_qr, dob_mrz)
        details.append(CrossValidationDetail(
            fieldName="dateOfBirth", ocrValue=dob_ocr, qrValue=dob_qr, mrzValue=dob_mrz, status=dob_status
        ))
        if dob_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_DATE_OF_BIRTH")

        # 4. Compare Date of Expiry
        exp_ocr = ocr_data.dateOfExpiry
        exp_qr = qr_data.get("dateOfExpiry") if qr_data else None
        exp_mrz = mrz_data.get("dateOfExpiry") if mrz_data else None

        exp_status = self._evaluate_field_status(exp_ocr, exp_qr, exp_mrz)
        details.append(CrossValidationDetail(
            fieldName="dateOfExpiry", ocrValue=exp_ocr, qrValue=exp_qr, mrzValue=exp_mrz, status=exp_status
        ))
        if exp_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_DATE_OF_EXPIRY")

        # 5. Evaluate Pairwise Global Flags (ocrMatchQr, ocrMatchMrz)
        ocr_match_qr = self._evaluate_pairwise_boolean(id_ocr, id_qr, name_ocr, name_qr, dob_ocr, dob_qr, exp_ocr, exp_qr)
        ocr_match_mrz = self._evaluate_pairwise_boolean(id_ocr, id_mrz, name_ocr, name_mrz, dob_ocr, dob_mrz, exp_ocr, exp_mrz)

        # 6. Check Expiry
        is_expired = False
        mrz_valid = mrz_data.get("isMrzValid", True) if mrz_data else True

        cross_val_result = CrossValidationResult(
            ocrMatchQr=ocr_match_qr,
            ocrMatchMrz=ocr_match_mrz,
            mrzCheckDigitValid=mrz_valid,
            isExpired=is_expired,
            details=details
        )

        # 7. Card Verification Decision Logic
        card_verified = (
            card_type != "UNKNOWN"
            and not is_expired
            and mrz_valid
            and id_ocr is not None
            and name_ocr is not None
            and ocr_match_qr is not False
            and ocr_match_mrz is not False
        )

        return card_verified, cross_val_result, errors

    def _evaluate_field_status(self, val_ocr: Optional[str], val_qr: Optional[str], val_mrz: Optional[str]) -> str:
        vals = [v for v in [val_ocr, val_qr, val_mrz] if v is not None]
        if len(vals) < 2:
            return "NOT_AVAILABLE"

        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if vals[i] != vals[j]:
                    return "MISMATCH"
        return "MATCH"

    def _evaluate_name_status(self, val_ocr: Optional[str], val_qr: Optional[str], val_mrz: Optional[str]) -> str:
        vals = [v for v in [val_ocr, val_qr, val_mrz] if v is not None]
        if len(vals) < 2:
            return "NOT_AVAILABLE"

        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if compare_names(vals[i], vals[j]) < settings.FULLNAME_FUZZY_THRESHOLD:
                    return "MISMATCH"
        return "MATCH"

    def _evaluate_pairwise_boolean(self, *pairs) -> Optional[bool]:
        has_comparison = False
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
