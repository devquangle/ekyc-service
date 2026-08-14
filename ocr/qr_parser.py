import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from ocr.normalizer import parse_date, normalize_full_name, normalize_gender, normalize_address
from utils.logger import logger


class QrParser:
    """
    Decodes QR Code from Vietnamese CCCD front/back card images using multi-step cropping,
    grayscale enhancement, CLAHE/Thresholding, 180-degree rotation fallbacks, and multiple QR decoding engines.
    Tracks and extracts accurate QR bounding box [x_min, y_min, x_max, y_max].
    """

    def __init__(self):
        self.cv_detector = cv2.QRCodeDetector()
        self.last_qr_bbox: Optional[List[float]] = None

    def decode(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
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

        # 3. zxing-cpp if installed
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

    def _decode_image_variants(self, img: np.ndarray) -> Optional[str]:
        text, _ = self._decode_variant_with_box(img)
        return text

    def _try_detect_qr_with_box(self, image: np.ndarray) -> Tuple[Optional[str], Optional[List[float]]]:
        if image is None or image.size == 0:
            return None, None

        # Tầng 1: Decode trực tiếp trên toàn ảnh gốc
        res_full, box_full = self._decode_variant_with_box(image)
        if res_full:
            return res_full, box_full

        # Tầng 2: Cắt vùng ROI góc trên bên phải [0:48%H, 48%W:100%W]
        h, w = image.shape[:2]
        offset_x = int(w * 0.48)
        offset_y = 0
        qr_roi = image[0:int(h * 0.48), offset_x:w]
        if qr_roi.size == 0:
            return None, None

        # Decode trực tiếp trên ROI gốc
        res_roi, box_roi = self._decode_variant_with_box(qr_roi)
        if res_roi:
            final_box = None
            if box_roi:
                final_box = [box_roi[0] + offset_x, box_roi[1] + offset_y, box_roi[2] + offset_x, box_roi[3] + offset_y]
            return res_roi, final_box

        # Tầng 3: Tiền xử lý ROI
        # + Phóng to 2x INTER_CUBIC
        resized = cv2.resize(qr_roi, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # + Chuyển sang ảnh xám (Grayscale)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
        # + Tăng tương phản CLAHE (clipLimit=3.0, tileGridSize=(8, 8))
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # + Nhị phân hóa Otsu
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Tầng 4: Chạy thử tuần tự qua các biến thể ảnh (gốc resized, CLAHE, Otsu, xoay 180 độ)
        variants = [
            enhanced,
            gray,
            otsu,
            otsu_enhanced,
            resized,
            cv2.rotate(enhanced, cv2.ROTATE_180),
            cv2.rotate(otsu, cv2.ROTATE_180),
        ]

        for variant in variants:
            res_var, box_var = self._decode_variant_with_box(variant)
            if res_var:
                final_box = None
                if box_var:
                    final_box = [
                        round(box_var[0] / 2.0 + offset_x, 1),
                        round(box_var[1] / 2.0 + offset_y, 1),
                        round(box_var[2] / 2.0 + offset_x, 1),
                        round(box_var[3] / 2.0 + offset_y, 1)
                    ]
                else:
                    # Fallback to the ROI area if exact box was not returned by engine
                    final_box = [float(offset_x), float(offset_y), float(w), float(int(h * 0.48))]
                return res_var, final_box

        return None, None

    def _try_detect_qr(self, image: np.ndarray) -> Optional[str]:
        text, _ = self._try_detect_qr_with_box(image)
        return text


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
