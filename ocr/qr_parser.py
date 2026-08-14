import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from ocr.normalizer import parse_date, normalize_full_name, normalize_gender, normalize_address
from utils.logger import logger


class QrParser:
    """
    High-Precision Multi-Format QR Code Parser for Vietnamese Identity Cards.
    Supports both Old CCCD chip cards (Front QR) and 2024 Căn Cước cards (Back/Front QR).
    Features multi-quadrant ROI extraction, multi-angle rotation, CLAHE/Otsu preprocessing,
    and multi-engine decoding (OpenCV, pyzbar, zxingcpp).
    """

    def __init__(self):
        self.cv_detector = cv2.QRCodeDetector()
        self.last_qr_bbox: Optional[List[float]] = None

    def decode(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Decodes and parses QR code into canonical field dictionary.
        """
        if image is None or image.size == 0:
            self.last_qr_bbox = None
            return None

        self.last_qr_bbox = None
        raw_qr_text, qr_box = self._try_detect_qr_with_box(image)
        self.last_qr_bbox = qr_box

        if not raw_qr_text:
            logger.info("[QR_PARSER] No QR code detected or unreadable.")
            return None

        logger.info(f"[QR_PARSER] Raw QR text decoded: {raw_qr_text} (bbox={qr_box})")
        return self.parse_qr_string(raw_qr_text)

    def decode_with_bbox(self, image: np.ndarray) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
        data = self.decode(image)
        return data, self.last_qr_bbox

    def _decode_variant_with_box(self, img: np.ndarray) -> Tuple[Optional[str], Optional[List[float]]]:
        if img is None or img.size == 0:
            return None, None

        # 1. OpenCV QRCodeDetector
        try:
            val, pts, _ = self.cv_detector.detectAndDecode(img)
            if val and len(val.strip()) > 0:
                box = None
                if pts is not None and len(pts) > 0:
                    pts_arr = pts[0] if len(pts.shape) == 3 else pts
                    xs = [pt[0] for pt in pts_arr]
                    ys = [pt[1] for pt in pts_arr]
                    box = [round(float(min(xs)), 1), round(float(min(ys)), 1), round(float(max(xs)), 1), round(float(max(ys)), 1)]
                return val.strip(), box
        except Exception:
            pass

        # 2. pyzbar
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode
            decoded_objects = pyzbar_decode(img)
            for obj in decoded_objects:
                text = obj.data.decode("utf-8", errors="ignore").strip()
                if text:
                    box = None
                    if hasattr(obj, 'rect') and obj.rect:
                        box = [
                            round(float(obj.rect.left), 1),
                            round(float(obj.rect.top), 1),
                            round(float(obj.rect.left + obj.rect.width), 1),
                            round(float(obj.rect.top + obj.rect.height), 1)
                        ]
                    return text, box
        except Exception:
            pass

        # 3. zxingcpp if installed
        try:
            import zxingcpp
            results = zxingcpp.read_barcodes(img)
            for res in results:
                if res.text and len(res.text.strip()) > 0:
                    box = None
                    if hasattr(res, 'position') and res.position:
                        pts_z = [res.position.top_left, res.position.top_right, res.position.bottom_right, res.position.bottom_left]
                        xs = [p.x for p in pts_z]
                        ys = [p.y for p in pts_z]
                        box = [round(float(min(xs)), 1), round(float(min(ys)), 1), round(float(max(xs)), 1), round(float(max(ys)), 1)]
                    return res.text.strip(), box
        except Exception:
            pass

        return None, None

    def _try_detect_qr_with_box(self, image: np.ndarray) -> Tuple[Optional[str], Optional[List[float]]]:
        if image is None or image.size == 0:
            return None, None

        # Tier 1: Decode directly on original full frame
        res_full, box_full = self._decode_variant_with_box(image)
        if res_full:
            return res_full, box_full

        h, w = image.shape[:2]

        # Tier 2: Multi-Quadrant Search (Old CCCD Top-Right, 2024 Card Top-Left / Bottom-Right)
        quadrants = [
            ("TOP_RIGHT", 0, int(h * 0.50), int(w * 0.45), w),
            ("TOP_LEFT", 0, int(h * 0.50), 0, int(w * 0.55)),
            ("BOTTOM_RIGHT", int(h * 0.45), h, int(w * 0.45), w),
        ]

        for q_name, y1, y2, x1, x2 in quadrants:
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # Check direct ROI
            res_roi, box_roi = self._decode_variant_with_box(roi)
            if res_roi:
                final_box = [box_roi[0] + x1, box_roi[1] + y1, box_roi[2] + x1, box_roi[3] + y1] if box_roi else [float(x1), float(y1), float(x2), float(y2)]
                return res_roi, final_box

            # Tier 3: Multi-Stage Image Enhancement on ROI
            resized = cv2.resize(roi, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized

            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, otsu_enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            variants = [
                enhanced,
                otsu,
                otsu_enhanced,
                gray,
                resized,
                cv2.rotate(enhanced, cv2.ROTATE_90_CLOCKWISE),
                cv2.rotate(enhanced, cv2.ROTATE_180),
                cv2.rotate(enhanced, cv2.ROTATE_90_COUNTERCLOCKWISE),
                cv2.rotate(otsu, cv2.ROTATE_180),
            ]

            for variant in variants:
                res_var, box_var = self._decode_variant_with_box(variant)
                if res_var:
                    if box_var:
                        final_box = [
                            round(box_var[0] / 2.0 + x1, 1),
                            round(box_var[1] / 2.0 + y1, 1),
                            round(box_var[2] / 2.0 + x1, 1),
                            round(box_var[3] / 2.0 + y1, 1)
                        ]
                    else:
                        final_box = [float(x1), float(y1), float(x2), float(y2)]
                    return res_var, final_box

        return None, None

    def parse_qr_string(self, qr_str: str) -> Optional[Dict[str, Any]]:
        """
        Parses pipe-delimited Vietnamese CCCD / Căn cước QR strings:
        Format: CCCD_12 | CMND_9 | Full Name | Date of Birth (DDMMYYYY) | Gender | Address | Date of Issue (DDMMYYYY)
        """
        if not qr_str:
            return None

        parts = [p.strip() for p in qr_str.split("|")]

        if len(parts) < 6:
            logger.warning(f"[QR_PARSER] QR string has insufficient pipe fields: {len(parts)}")
            return None

        identity_number = parts[0] if (parts[0].isdigit() and len(parts[0]) == 12) else None
        old_id_number = parts[1] if len(parts[1]) > 0 else None

        canonical_name, raw_clean_name = normalize_full_name(parts[2]) if len(parts[2]) > 0 else (None, None)
        full_name = raw_clean_name if raw_clean_name else canonical_name
        date_of_birth = parse_date(parts[3]) if len(parts[3]) > 0 else None
        gender = normalize_gender(parts[4]) if len(parts[4]) > 0 else None

        raw_address = parts[5] if len(parts[5]) > 0 else None
        place_of_residence, _ = normalize_address(raw_address) if raw_address else (None, None)

        date_of_issue = parse_date(parts[6]) if len(parts) > 6 and len(parts[6]) > 0 else None

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
