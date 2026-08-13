import re
import cv2
import numpy as np
from typing import Optional, Dict, Any, List
from ocr.normalizer import parse_date, normalize_full_name, normalize_gender
from utils.logger import logger


class MrzParser:
    """
    MRZ Image preprocessing and TD1 3-line format text parser with Modulo 10 check digit verification.
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

    def preprocess_mrz_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Crops bottom 35% of back card image, converts to grayscale, resizes, and applies Otsu thresholding.
        """
        if image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        crop = image[int(h * 0.60):h, 0:w]

        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        resized = cv2.resize(gray, (1024, int(1024 * (crop.shape[0] / crop.shape[1]))))
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def parse_mrz_lines(self, ocr_text_lines: List[str]) -> Optional[Dict[str, Any]]:
        if not ocr_text_lines:
            return None

        # Filter candidate lines containing TD1 MRZ pattern
        mrz_candidate_lines = []
        for line in ocr_text_lines:
            clean_line = re.sub(r'\s+', '', line).upper()
            clean_line = clean_line.replace('«', '<').replace('(', '<')
            if len(clean_line) >= 20 and ('VNM' in clean_line or '<' in clean_line or clean_line.startswith('I')):
                mrz_candidate_lines.append(clean_line)

        if len(mrz_candidate_lines) < 3:
            logger.debug(f"[MRZ_PARSER] Insufficient MRZ candidate lines: {len(mrz_candidate_lines)}")
            return None

        # Take last 3 lines
        l1 = mrz_candidate_lines[-3]
        l2 = mrz_candidate_lines[-2]
        l3 = mrz_candidate_lines[-1]

        l1 = (l1 + "<" * 30)[:30]
        l2 = (l2 + "<" * 30)[:30]
        l3 = (l3 + "<" * 30)[:30]

        logger.info(f"[MRZ_PARSER] Line 1: {l1}")
        logger.info(f"[MRZ_PARSER] Line 2: {l2}")
        logger.info(f"[MRZ_PARSER] Line 3: {l3}")

        # --- Parse Line 1 ---
        raw_id_field = l1[15:27]
        clean_id_field = re.sub(r'[^\d]', '', raw_id_field)
        identity_number = clean_id_field if len(clean_id_field) == 12 and clean_id_field.startswith("0") else None

        # --- Parse Line 2 ---
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
            logger.error(f"[MRZ_PARSER] Check digit validation error: {str(e)}")

        is_mrz_valid = valid_dob_check and valid_expiry_check

        date_of_birth = parse_date(dob_raw)
        date_of_expiry = parse_date(expiry_raw)
        gender = normalize_gender(sex_raw)

        # --- Parse Line 3 ---
        raw_name = l3.replace("<", " ").strip()
        full_name, _ = normalize_full_name(raw_name)

        logger.info(f"[MRZ_PARSER] Result: ID={identity_number} Name={full_name} DoB={date_of_birth} Expiry={date_of_expiry} Valid={is_mrz_valid}")

        return {
            "identityNumber": identity_number,
            "fullName": full_name,
            "dateOfBirth": date_of_birth,
            "gender": gender,
            "nationality": "Việt Nam",
            "dateOfExpiry": date_of_expiry,
            "isMrzValid": is_mrz_valid,
            "mrzCheckDigitValid": is_mrz_valid,
        }
