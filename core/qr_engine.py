import cv2
import numpy as np
from typing import Optional, Dict, Any
from utils.logger import logger
from utils.text_utils import normalize_text, normalize_date


class QrEngine:
    """
    QR Code Detection & Decoding Engine for Vietnamese Identity Cards.
    Handles pipe-separated '|' string structure.
    """

    def __init__(self):
        self.opencv_qr_detector = cv2.QRCodeDetector()

    def decode(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Detects and decodes QR Code from image.
        Returns dictionary of extracted fields if successful, else None.
        """
        if image is None or image.size == 0:
            return None

        raw_qr_string = self._detect_qr_string(image)
        if not raw_qr_string:
            return None

        return self.parse_qr_string(raw_qr_string)

    def _detect_qr_string(self, image: np.ndarray) -> Optional[str]:
        # Method 1: OpenCV QRCodeDetector
        try:
            val, pts, _ = self.opencv_qr_detector.detectAndDecode(image)
            if val and len(val.strip()) > 0:
                return val.strip()
        except Exception:
            pass

        # Method 2: Fallback pyzbar
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            decoded_objects = pyzbar_decode(image)
            for obj in decoded_objects:
                if obj.data:
                    return obj.data.decode("utf-8", errors="ignore").strip()
        except Exception:
            pass

        # Method 3: Fallback zxing-cpp
        try:
            import zxingcpp
            results = zxingcpp.read_barcodes(image)
            for res in results:
                if res.text:
                    return res.text.strip()
        except Exception:
            pass

        return None

    def parse_qr_string(self, raw_string: str) -> Optional[Dict[str, Any]]:
        """
        Parses Vietnamese ID card QR pipe-separated string structure:
        Format: [ID_12_digits]|[CMND_9_digits]|[FullName]|[DDMMYYYY_DoB]|[Gender]|[Address]|[DDMMYYYY_DoI]
        """
        if not raw_string or "|" not in raw_string:
            return None

        parts = raw_string.split("|")
        if len(parts) < 6:
            logger.warning("QR string does not contain required pipe-separated fields.")
            return None

        identity_number = parts[0].strip() if len(parts) > 0 else None
        cmnd_9_digits = parts[1].strip() if len(parts) > 1 else None
        full_name = normalize_text(parts[2]) if len(parts) > 2 else None
        dob_raw = parts[3].strip() if len(parts) > 3 else None
        gender = parts[4].strip() if len(parts) > 4 else None
        residence = normalize_text(parts[5]) if len(parts) > 5 else None
        doi_raw = parts[6].strip() if len(parts) > 6 else None

        # Normalize dates to ISO YYYY-MM-DD
        date_of_birth = normalize_date(dob_raw)
        date_of_issue = normalize_date(doi_raw)

        return {
            "identityNumber": identity_number,
            "cmnd9Digits": cmnd_9_digits,
            "fullName": full_name,
            "dateOfBirth": date_of_birth,
            "gender": gender,
            "placeOfResidence": residence,
            "dateOfIssue": date_of_issue,
        }
