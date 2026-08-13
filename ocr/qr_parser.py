import cv2
import numpy as np
from typing import Optional, Dict, Any
from ocr.normalizer import parse_date, normalize_full_name, normalize_gender, normalize_address
from utils.logger import logger


class QrParser:
    """
    Decodes QR Code from Vietnamese CCCD front/back card images.
    Returns structured dict or None. Does not fall back to OCR text.
    """

    def __init__(self):
        self.cv_detector = cv2.QRCodeDetector()

    def decode(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        if image is None or image.size == 0:
            return None

        raw_qr_text = self._try_detect_qr(image)

        if not raw_qr_text:
            logger.info("[QR_PARSER] No QR code detected or unreadable.")
            return None

        logger.info(f"[QR_PARSER] Raw QR text decoded: {raw_qr_text}")
        return self.parse_qr_string(raw_qr_text)

    def _try_detect_qr(self, image: np.ndarray) -> Optional[str]:
        try:
            val, pts, _ = self.cv_detector.detectAndDecode(image)
            if val and len(val.strip()) > 0:
                return val.strip()
        except Exception:
            pass

        # Try pyzbar if available
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            decoded_objects = pyzbar_decode(image)
            for obj in decoded_objects:
                text = obj.data.decode("utf-8", errors="ignore").strip()
                if text:
                    return text
        except Exception:
            pass

        return None

    def parse_qr_string(self, qr_str: str) -> Optional[Dict[str, Any]]:
        parts = qr_str.split("|")

        if len(parts) < 6:
            logger.warning(f"[QR_PARSER] Invalid QR string format (parts={len(parts)})")
            return None

        identity_number = parts[0].strip() if parts[0].strip().isdigit() and len(parts[0].strip()) == 12 else None
        old_id_number = parts[1].strip() if len(parts[1].strip()) > 0 else None
        
        canonical_name, raw_clean_name = normalize_full_name(parts[2].strip()) if len(parts[2].strip()) > 0 else (None, None)
        full_name = raw_clean_name if raw_clean_name else canonical_name
        date_of_birth = parse_date(parts[3].strip()) if len(parts[3].strip()) > 0 else None
        gender = normalize_gender(parts[4].strip()) if len(parts[4].strip()) > 0 else None
        
        raw_address = parts[5].strip() if len(parts[5].strip()) > 0 else None
        place_of_residence, _ = normalize_address(raw_address) if raw_address else (None, None)

        date_of_issue = parse_date(parts[6].strip()) if len(parts) > 6 and len(parts[6].strip()) > 0 else None

        result = {
            "identityNumber": identity_number,
            "oldIdentityNumber": old_id_number,
            "fullName": full_name,
            "dateOfBirth": date_of_birth,
            "gender": gender,
            "placeOfResidence": place_of_residence,
            "dateOfIssue": date_of_issue,
            "nationality": "Việt Nam",
        }

        logger.info(f"[QR_PARSER] Parsed fields: {result}")
        return result
