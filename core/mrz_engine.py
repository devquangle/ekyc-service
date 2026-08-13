import re
from typing import Optional, Dict, Any, List
from utils.logger import logger
from utils.text_utils import normalize_date, normalize_text
from config import settings


class MrzEngine:
    """
    MRZ Parser Engine strictly for ICAO Doc 9303 TD1 (3-line format x 30 characters).
    Enforces strict positional extraction, Modulo 10 check digit verification,
    and detailed debug logging without arbitrary guessing.
    """

    @staticmethod
    def compute_check_digit(mrz_substr: str) -> int:
        """
        Computes ICAO Doc 9303 Modulo 10 check digit using 7-3-1 weight pattern.
        """
        weights = [7, 3, 1]
        total = 0
        for idx, char in enumerate(mrz_substr):
            if '0' <= char <= '9':
                val = int(char)
            elif 'A' <= char <= 'Z':
                val = ord(char) - ord('A') + 10
            elif char == '<':
                val = 0
            else:
                val = 0
            total += val * weights[idx % 3]
        return total % 10

    def parse(self, ocr_text_lines: List[str]) -> Optional[Dict[str, Any]]:
        """
        Detects and parses 3-line TD1 MRZ lines from OCR text lines list.
        Line 1: I<VNM[10-char Doc Number][12-digit ID Number]<<[CheckDigit]
        Line 2: [dateOfBirth (YYMMDD)][Check1][gender (M/F)][dateOfExpiry (YYMMDD)][Check2][VNM]<<<<<<<<[CompositeCheck]
        Line 3: [fullName (separated by <<)]
        """
        if not ocr_text_lines:
            return None

        # Filter candidate lines
        mrz_candidate_lines = []
        for line in ocr_text_lines:
            clean_line = re.sub(r'\s+', '', line).upper()
            clean_line = clean_line.replace('«', '<').replace('(', '<')
            if len(clean_line) >= 25 and ('VNM' in clean_line or '<' in clean_line or clean_line.startswith('I')):
                mrz_candidate_lines.append(clean_line)

        if len(mrz_candidate_lines) < 3:
            if settings.DEBUG:
                logger.debug(f"[MRZ_RAW] Insufficient candidate lines found: {len(mrz_candidate_lines)}")
            return None

        # Take last 3 lines
        l1 = mrz_candidate_lines[-3]
        l2 = mrz_candidate_lines[-2]
        l3 = mrz_candidate_lines[-1]

        # Pad to 30 chars
        l1 = (l1 + "<" * 30)[:30]
        l2 = (l2 + "<" * 30)[:30]
        l3 = (l3 + "<" * 30)[:30]

        if settings.DEBUG:
            logger.debug(f"[MRZ_RAW] Line 1: {l1}")
            logger.debug(f"[MRZ_RAW] Line 2: {l2}")
            logger.debug(f"[MRZ_RAW] Line 3: {l3}")
            logger.debug(f"[MRZ_FORMAT] TD1 3-line detected")

        # --- Parse Line 1 ---
        # Pos 0..4: "IDVNM" or "I<VNM"
        # Pos 5..14: 10-char document number
        # Pos 15..26: 12-digit Personal Identification Number
        identity_number = None
        raw_id_field = l1[15:27]

        # Strict validation: Vietnamese ID number in MRZ must be 12 digits starting with '0'
        clean_id_field = re.sub(r'[^\d]', '', raw_id_field)
        if len(clean_id_field) == 12 and clean_id_field.startswith("0"):
            identity_number = clean_id_field
            if settings.DEBUG:
                logger.debug(f"[MRZ_IDENTITY_NUMBER] Valid positional match: {identity_number}")
        else:
            if settings.DEBUG:
                logger.debug(f"[MRZ_IDENTITY_NUMBER] Positional field 15:27 '{raw_id_field}' invalid. Set to null (No guessing).")
            identity_number = None

        # --- Parse Line 2 ---
        # Pos 0..5: YYMMDD (DoB)
        # Pos 6: Check digit DoB
        # Pos 7: Gender (M/F)
        # Pos 8..13: YYMMDD (Expiry)
        # Pos 14: Check digit Expiry
        dob_raw = l2[0:6]
        dob_check = l2[6:7]
        sex_raw = l2[7:8]
        expiry_raw = l2[8:14]
        expiry_check = l2[14:15]

        # Verify Check Digits
        valid_dob_check = False
        valid_expiry_check = False
        try:
            if dob_check.isdigit():
                valid_dob_check = (self.compute_check_digit(dob_raw) == int(dob_check))
            if expiry_check.isdigit():
                valid_expiry_check = (self.compute_check_digit(expiry_raw) == int(expiry_check))
        except Exception as e:
            if settings.DEBUG:
                logger.debug(f"[MRZ_FORMAT] Check digit error: {str(e)}")

        is_mrz_valid = valid_dob_check and valid_expiry_check

        date_of_birth = normalize_date(dob_raw)
        date_of_expiry = normalize_date(expiry_raw)
        gender = "Nam" if sex_raw == "M" else ("Nữ" if sex_raw == "F" else None)

        if settings.DEBUG:
            logger.debug(f"[MRZ_DATE_OF_BIRTH] {date_of_birth}")
            logger.debug(f"[MRZ_DATE_OF_EXPIRY] {date_of_expiry}")

        # --- Parse Line 3 ---
        raw_name = l3.replace("<", " ").strip()
        full_name = normalize_text(raw_name)

        if settings.DEBUG:
            logger.debug(f"[MRZ_FULL_NAME] {full_name}")

        return {
            "identityNumber": identity_number,
            "fullName": full_name,
            "dateOfBirth": date_of_birth,
            "gender": gender,
            "nationality": "Việt Nam",
            "dateOfExpiry": date_of_expiry,
            "isMrzValid": is_mrz_valid,
        }
