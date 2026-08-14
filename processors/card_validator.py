from datetime import datetime
from typing import Tuple, Dict, Any, List, Optional
from schemas.card import ExtractedCardData, CrossValidationResult, CrossValidationDetail
from ocr.normalizer import (
    normalize_text_for_compare,
    normalize_identity_number,
    normalize_gender,
    parse_date,
    normalize_address_for_compare,
)
from utils.text_utils import compare_names
from utils.logger import logger
from config import settings


class CardValidator:
    """
    Card Cross-Validation Engine for Vietnamese Identity Cards.
    Performs type-specialized multi-source cross-validation across OCR, QR, and MRZ data sources:
    - fullName: Normalized Vietnamese phonetic & token fuzzy matching (FULLNAME_FUZZY_THRESHOLD).
    - identityNumber: Strict normalized digit sequence equality (9 or 12 digits).
    - dateOfBirth / dateOfExpiry / dateOfIssue: Strict ISO YYYY-MM-DD calendar date equality.
    - gender: Canonical 'Nam' / 'Nữ' equality.
    - addresses: Administrative token & gazetteer comparison.
    """

    def validate(
        self,
        ocr_data: ExtractedCardData,
        qr_data: Optional[Dict[str, Any]],
        mrz_data: Optional[Dict[str, Any]],
        card_type: str
    ) -> Tuple[bool, CrossValidationResult, List[str]]:
        """
        Executes strict cross-validation and expiry checking across all card fields.

        Returns:
            Tuple[bool, CrossValidationResult, List[str]]:
                - card_verified (bool): Overall verification decision.
                - cross_validation_result (CrossValidationResult): Detailed comparison status per field.
                - errors (List[str]): List of error codes or warnings.
        """
        errors: List[str] = []
        details: List[CrossValidationDetail] = []

        # 1. Identity Number Cross-Validation (Strict Digit Equality)
        id_ocr = ocr_data.identityNumber
        id_qr = qr_data.get("identityNumber") if qr_data else None
        id_mrz = mrz_data.get("identityNumber") if mrz_data else None
        id_status = self._evaluate_identity_number_status(id_ocr, id_qr, id_mrz)
        details.append(CrossValidationDetail(
            fieldName="identityNumber", ocrValue=id_ocr, qrValue=id_qr, mrzValue=id_mrz, status=id_status
        ))
        if id_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_IDENTITY_NUMBER")

        # 2. Full Name Cross-Validation (Fuzzy Name Comparison)
        name_ocr = ocr_data.fullName
        name_qr = qr_data.get("fullName") if qr_data else None
        name_mrz = mrz_data.get("fullName") if mrz_data else None
        name_status = self._evaluate_name_status(name_ocr, name_qr, name_mrz)
        details.append(CrossValidationDetail(
            fieldName="fullName", ocrValue=name_ocr, qrValue=name_qr, mrzValue=name_mrz, status=name_status
        ))
        if name_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_FULL_NAME")

        # 3. Date of Birth Cross-Validation (Strict ISO Date Equality)
        dob_ocr = ocr_data.dateOfBirth
        dob_qr = qr_data.get("dateOfBirth") if qr_data else None
        dob_mrz = mrz_data.get("dateOfBirth") if mrz_data else None
        dob_status = self._evaluate_date_status(dob_ocr, dob_qr, dob_mrz, is_expiry=False)
        details.append(CrossValidationDetail(
            fieldName="dateOfBirth", ocrValue=dob_ocr, qrValue=dob_qr, mrzValue=dob_mrz, status=dob_status
        ))
        if dob_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_DATE_OF_BIRTH")

        # 4. Gender Cross-Validation (Canonical Gender Equality)
        gender_ocr = ocr_data.gender
        gender_qr = qr_data.get("gender") if qr_data else None
        gender_mrz = mrz_data.get("gender") if mrz_data else None
        gender_status = self._evaluate_gender_status(gender_ocr, gender_qr, gender_mrz)
        details.append(CrossValidationDetail(
            fieldName="gender", ocrValue=gender_ocr, qrValue=gender_qr, mrzValue=gender_mrz, status=gender_status
        ))
        if gender_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_GENDER")

        # 5. Date of Expiry Cross-Validation (Strict ISO Date Equality)
        exp_ocr = ocr_data.dateOfExpiry
        exp_qr = qr_data.get("dateOfExpiry") if qr_data else None
        exp_mrz = mrz_data.get("dateOfExpiry") if mrz_data else None
        exp_status = self._evaluate_date_status(exp_ocr, exp_qr, exp_mrz, is_expiry=True)
        details.append(CrossValidationDetail(
            fieldName="dateOfExpiry", ocrValue=exp_ocr, qrValue=exp_qr, mrzValue=exp_mrz, status=exp_status
        ))
        if exp_status == "MISMATCH":
            errors.append("CARD_DATA_MISMATCH_DATE_OF_EXPIRY")

        # 6. Global Pairwise Match Flags (ocrMatchQr, ocrMatchMrz)
        ocr_match_qr = self._evaluate_ocr_pairwise_match(ocr_data, qr_data)
        ocr_match_mrz = self._evaluate_ocr_pairwise_match(ocr_data, mrz_data, mrz_allowed_only=True)

        # 7. Card Expiry Validation against Current Date
        is_expired = False
        exp_str = exp_ocr or exp_qr or exp_mrz
        if exp_str:
            try:
                parsed_exp = parse_date(exp_str, is_expiry=True)
                if parsed_exp:
                    exp_date = datetime.strptime(parsed_exp, "%Y-%m-%d").date()
                    if exp_date < datetime.now().date():
                        is_expired = True
                        errors.append("CARD_EXPIRED")
            except ValueError:
                pass

        # 8. MRZ Checksum Status
        mrz_valid = mrz_data.get("isMrzValid", True) if mrz_data else True
        if mrz_data and not mrz_valid:
            errors.append("CARD_DATA_WARNING_MRZ_CHECKSUM_INVALID")

        cross_val_result = CrossValidationResult(
            ocrMatchQr=ocr_match_qr,
            ocrMatchMrz=ocr_match_mrz,
            mrzCheckDigitValid=mrz_valid,
            isExpired=is_expired,
            details=details
        )

        # 9. Verification Decision Matrix
        is_cccd_new = (card_type == "CCCD_NEW" or mrz_data is None)
        has_valid_qr = (qr_data is not None and ocr_match_qr is True)
        has_valid_ocr = (ocr_data.identityNumber is not None and ocr_data.fullName is not None)

        if is_cccd_new:
            card_verified = (
                card_type != "UNKNOWN"
                and not is_expired
                and has_valid_ocr
                and ocr_match_qr is not False
            )
            logger.info(f"[CARD_VALIDATOR] CCCD_NEW: verified={card_verified} (qr_match={ocr_match_qr})")
        else:
            mrz_pass_or_qr_override = mrz_valid or has_valid_qr
            card_verified = (
                card_type != "UNKNOWN"
                and not is_expired
                and has_valid_ocr
                and mrz_pass_or_qr_override
                and ocr_match_qr is not False
                and (has_valid_qr or ocr_match_mrz is not False)
            )
            logger.info(f"[CARD_VALIDATOR] CCCD_OLD: verified={card_verified} (mrz_valid={mrz_valid}, qr_match={ocr_match_qr}, mrz_match={ocr_match_mrz})")

        return card_verified, cross_val_result, errors

    # ==================== Type-Specific Field Evaluators ====================

    def _evaluate_identity_number_status(
        self, val_ocr: Optional[str], val_qr: Optional[str], val_mrz: Optional[str]
    ) -> str:
        vals = [v for v in [val_ocr, val_qr, val_mrz] if v is not None]
        if len(vals) < 2:
            return "NOT_AVAILABLE"

        norm_ids = [normalize_identity_number(v) or str(v).strip() for v in vals]
        for i in range(len(norm_ids)):
            for j in range(i + 1, len(norm_ids)):
                if norm_ids[i] != norm_ids[j]:
                    return "MISMATCH"
        return "MATCH"

    def _evaluate_name_status(
        self, val_ocr: Optional[str], val_qr: Optional[str], val_mrz: Optional[str]
    ) -> str:
        vals = [v for v in [val_ocr, val_qr, val_mrz] if v is not None]
        if len(vals) < 2:
            return "NOT_AVAILABLE"

        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                sim = compare_names(vals[i], vals[j])
                if sim < settings.FULLNAME_FUZZY_THRESHOLD:
                    return "MISMATCH"
        return "MATCH"

    def _evaluate_date_status(
        self, val_ocr: Optional[str], val_qr: Optional[str], val_mrz: Optional[str], is_expiry: bool = False
    ) -> str:
        vals = [v for v in [val_ocr, val_qr, val_mrz] if v is not None]
        if len(vals) < 2:
            return "NOT_AVAILABLE"

        parsed_dates = [parse_date(v, is_expiry=is_expiry) for v in vals]
        if any(d is None for d in parsed_dates):
            # If any date could not be parsed into ISO format, compare normalized strings
            comp_keys = [normalize_text_for_compare(v) for v in vals]
            for i in range(len(comp_keys)):
                for j in range(i + 1, len(comp_keys)):
                    if comp_keys[i] != comp_keys[j]:
                        return "MISMATCH"
            return "MATCH"

        for i in range(len(parsed_dates)):
            for j in range(i + 1, len(parsed_dates)):
                if parsed_dates[i] != parsed_dates[j]:
                    return "MISMATCH"
        return "MATCH"

    def _evaluate_gender_status(
        self, val_ocr: Optional[str], val_qr: Optional[str], val_mrz: Optional[str]
    ) -> str:
        vals = [v for v in [val_ocr, val_qr, val_mrz] if v is not None]
        if len(vals) < 2:
            return "NOT_AVAILABLE"

        norm_genders = [normalize_gender(v) for v in vals]
        for i in range(len(norm_genders)):
            for j in range(i + 1, len(norm_genders)):
                if norm_genders[i] is None or norm_genders[j] is None or norm_genders[i] != norm_genders[j]:
                    return "MISMATCH"
        return "MATCH"

    # ==================== Pairwise Cross-Validation ====================

    def _evaluate_ocr_pairwise_match(
        self,
        ocr_data: ExtractedCardData,
        other_data: Optional[Dict[str, Any]],
        mrz_allowed_only: bool = False
    ) -> Optional[bool]:
        """
        Evaluates pairwise match between OCR extracted data and a secondary source (QR or MRZ).
        Applies type-specific comparators for each field.
        """
        if not other_data:
            return None

        comparisons_made = 0

        # 1. Identity Number
        id_ocr = ocr_data.identityNumber
        id_other = other_data.get("identityNumber")
        if id_ocr and id_other:
            comparisons_made += 1
            n_ocr = normalize_identity_number(id_ocr) or str(id_ocr).strip()
            n_other = normalize_identity_number(id_other) or str(id_other).strip()
            if n_ocr != n_other:
                return False

        # 2. Full Name
        name_ocr = ocr_data.fullName
        name_other = other_data.get("fullName")
        if name_ocr and name_other:
            comparisons_made += 1
            if compare_names(name_ocr, name_other) < settings.FULLNAME_FUZZY_THRESHOLD:
                return False

        # 3. Date of Birth
        dob_ocr = ocr_data.dateOfBirth
        dob_other = other_data.get("dateOfBirth")
        if dob_ocr and dob_other:
            comparisons_made += 1
            d_ocr = parse_date(dob_ocr, is_expiry=False)
            d_other = parse_date(dob_other, is_expiry=False)
            if d_ocr != d_other:
                return False

        # 4. Gender
        g_ocr = ocr_data.gender
        g_other = other_data.get("gender")
        if g_ocr and g_other:
            comparisons_made += 1
            gn_ocr = normalize_gender(g_ocr)
            gn_other = normalize_gender(g_other)
            if gn_ocr != gn_other:
                return False

        # 5. Date of Expiry
        exp_ocr = ocr_data.dateOfExpiry
        exp_other = other_data.get("dateOfExpiry")
        if exp_ocr and exp_other:
            comparisons_made += 1
            e_ocr = parse_date(exp_ocr, is_expiry=True)
            e_other = parse_date(exp_other, is_expiry=True)
            if e_ocr != e_other:
                return False

        # 6. Address fields (for QR only)
        if not mrz_allowed_only:
            res_ocr = ocr_data.placeOfResidence
            res_other = other_data.get("placeOfResidence")
            if res_ocr and res_other:
                comparisons_made += 1
                a1 = normalize_address_for_compare(res_ocr)
                a2 = normalize_address_for_compare(res_other)
                # Flexible token overlap for address comparison
                if a1 and a2:
                    toks1 = set(a1.split())
                    toks2 = set(a2.split())
                    inter = toks1.intersection(toks2)
                    if len(toks1) > 0 and len(toks2) > 0:
                        overlap = len(inter) / max(min(len(toks1), len(toks2)), 1)
                        if overlap < 0.35:
                            return False

        if comparisons_made == 0:
            return None

        return True
