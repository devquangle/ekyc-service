import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from ocr.qr_parser import QrParser
from utils.logger import logger


class QrEngine:
    """
    QR Code Engine Wrapper delegating to OpenCV/PyZbar QrParser
    for extracting and parsing Vietnamese CCCD chip & non-chip QR payloads.
    """

    def __init__(self):
        self.parser = QrParser()

    def decode(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Decodes and parses QR code from card image into standardized dictionary.
        """
        if image is None or image.size == 0:
            return None
        return self.parser.decode(image)

    def decode_with_bbox(self, image: np.ndarray) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
        """
        Decodes QR code and returns (parsed_data, [x1, y1, x2, y2] bbox coordinates).
        """
        if image is None or image.size == 0:
            return None, None
        return self.parser.decode_with_bbox(image)

    @property
    def last_qr_bbox(self) -> Optional[List[float]]:
        return self.parser.last_qr_bbox

    def parse_qr_string(self, qr_str: str) -> Optional[Dict[str, Any]]:
        """
        Parses raw QR pipe-separated string into standard CCCD field dictionary.
        """
        if not qr_str:
            return None
        return self.parser.parse_qr_string(qr_str)
