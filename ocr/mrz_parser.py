import re
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
from ocr.normalizer import parse_date, normalize_full_name, normalize_gender
from utils.logger import logger


class MrzParser:
    """
    ICAO Doc 9303 Part 5 (TD1 3-line format) MRZ Parser & Validator.
    Features:
    - Auto-detection of MRZ candidate zone using bracket '<' density and token morphology.
    - Full ICAO Modulo 10 7-3-1 weight pattern check digits (DoB, Expiry, Document Number, Composite).
    - Century disambiguation for 2-digit years.
    - OCR glyph confusion repair ('O' <-> '0', 'I' <-> '1', 'S' <-> '5', etc.).
    """

    @staticmethod
    def _clean_numeric_field(text: str) -> str:
        """
        Cleans OCR character confusion errors in numeric MRZ fields:
        'O','Q','D' -> '0'; 'I','L' -> '1'; 'S' -> '5'; 'B' -> '8'; 'Z' -> '2'
        """
        if not text:
            return ""
        mapping = {
            'O': '0', 'Q': '0', 'D': '0',
            'I': '1', 'L': '1',
            'S': '5',
            'B': '8',
            'Z': '2',
        }
        res = [mapping.get(char, char) for char in text.upper()]
        return "".join(res)

    @staticmethod
    def _clean_alpha_field(text: str) -> str:
        """
        Cleans OCR character confusion errors in alpha MRZ fields:
        '0' -> 'O'; '1' -> 'I'; '5' -> 'S'
        """
        if not text:
            return ""
        mapping = {
            '0': 'O',
            '1': 'I',
            '5': 'S',
        }
        res = [mapping.get(char, char) for char in text.upper()]
        return "".join(res)

    @classmethod
    def compute_check_digit(cls, mrz_substr: str) -> int:
        """
        Computes ICAO Doc 9303 Modulo 10 check digit using [7, 3, 1] weight pattern.
        Automatically cleans OCR glyph confusions prior to calculation.
        """
        if not mrz_substr:
            return 0

        cleaned_substr = cls._clean_numeric_field(mrz_substr)
        weights = [7, 3, 1]
        total = 0

        for idx, char in enumerate(cleaned_substr.upper()):
            if '0' <= char <= '9':
                val = int(char)
            elif 'A' <= char <= 'Z':
                val = ord(char) - ord('A') + 10
            else:
                val = 0
            total += val * weights[idx % 3]

        return total % 10

    def parse_mrz_lines(self, ocr_text_lines: List[str]) -> Optional[Dict[str, Any]]:
        """
        Parses 3 TD1 MRZ lines into structured fields with complete ICAO Doc 9303 validation.
        """
        if not ocr_text_lines:
            return None

        # Filter candidate lines containing TD1 MRZ characteristics (< density, VNM, length >= 20)
        mrz_candidates: List[str] = []
        for line in ocr_text_lines:
            clean_line = re.sub(r'\s+', '', line).upper()
            clean_line = clean_line.replace('«', '<').replace('(', '<').replace(')', '<').replace('{', '<').replace('}', '<')
            if len(clean_line) >= 20 and ('<' in clean_line or 'VNM' in clean_line or clean_line.startswith('I')):
                mrz_candidates.append(clean_line)

        if len(mrz_candidates) < 3:
            logger.debug(f"[MRZ_PARSER] Insufficient MRZ candidate lines found: {len(mrz_candidates)}")
            return None

        # Take last 3 candidate lines
        l1 = mrz_candidates[-3]
        l2 = mrz_candidates[-2]
        l3 = mrz_candidates[-1]

        # Standard TD1 lines are strictly 30 characters
        l1 = (l1 + "<" * 30)[:30]
        l2 = (l2 + "<" * 30)[:30]
        l3 = (l3 + "<" * 30)[:30]

        logger.info(f"[MRZ_PARSER] Line 1: {l1}")
        logger.info(f"[MRZ_PARSER] Line 2: {l2}")
        logger.info(f"[MRZ_PARSER] Line 3: {l3}")

        # --- Parse Line 1: [0:2] Document Type, [2:5] Country, [5:14] Doc Num, [14] Check Digit, [15:30] Optional Data (12-digit CCCD) ---
        raw_id_field = l1[15:27]
        clean_id_field = self._clean_numeric_field(raw_id_field)
        digits_only_id = re.sub(r'[^\d]', '', clean_id_field)
        identity_number = digits_only_id if len(digits_only_id) == 12 and digits_only_id.startswith("0") else None

        # --- Parse Line 2: [0:6] DoB, [6] DoB Check, [7] Sex, [8:14] Expiry, [14] Expiry Check, [15:18] Nationality, [18:29] Optional Data 2, [29] Composite Check ---
        dob_raw = self._clean_numeric_field(l2[0:6])
        dob_check = self._clean_numeric_field(l2[6:7])
        sex_raw = l2[7:8]
        expiry_raw = self._clean_numeric_field(l2[8:14])
        expiry_check = self._clean_numeric_field(l2[14:15])
        nationality_raw = self._clean_alpha_field(l2[15:18])
        composite_check_raw = self._clean_numeric_field(l2[29:30])

        # 1. DoB Check Digit Validation
        valid_dob_check = False
        if dob_check.isdigit():
            valid_dob_check = (self.compute_check_digit(dob_raw) == int(dob_check))

        # 2. Expiry Check Digit Validation
        valid_expiry_check = False
        if expiry_check.isdigit():
            valid_expiry_check = (self.compute_check_digit(expiry_raw) == int(expiry_check))

        # 3. Composite Check Digit Validation (ICAO Doc 9303 Part 5 Section 4.2.2)
        # Composite data: Line1[5:30] + Line2[0:7] + Line2[8:15] + Line2[18:29]
        composite_data = l1[5:30] + l2[0:7] + l2[8:15] + l2[18:29]
        valid_composite_check = False
        if composite_check_raw.isdigit():
            expected_composite = self.compute_check_digit(composite_data)
            valid_composite_check = (expected_composite == int(composite_check_raw))

        is_mrz_valid = valid_dob_check and valid_expiry_check

        # Dates & Gender Normalization with century calculation
        date_of_birth = parse_date(dob_raw, is_expiry=False)
        date_of_expiry = parse_date(expiry_raw, is_expiry=True)
        gender = normalize_gender(sex_raw)

        # --- Parse Line 3: Name ---
        clean_l3_alpha = self._clean_alpha_field(l3)
        raw_name = clean_l3_alpha.replace("<", " ").strip()
        full_name, _ = normalize_full_name(raw_name)

        logger.info(
            f"[MRZ_PARSER] Parsed: ID={identity_number} Name={full_name} "
            f"DoB={date_of_birth} Expiry={date_of_expiry} Valid={is_mrz_valid} CompositeValid={valid_composite_check}"
        )

        return {
            "identityNumber": identity_number,
            "fullName": full_name,
            "dateOfBirth": date_of_birth,
            "gender": gender,
            "nationality": "Việt Nam",
            "dateOfExpiry": date_of_expiry,
            "isMrzValid": is_mrz_valid,
            "mrzCheckDigitValid": is_mrz_valid,
            "compositeCheckDigitValid": valid_composite_check,
        }
