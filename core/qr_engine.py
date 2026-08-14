import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from ocr.qr_parser import QrParser


class QrEngine:
    """
    QR Code Engine Wrapper delegating to ocr.qr_parser.QrParser.
    """

    def __init__(self):
        self.parser = QrParser()

    def decode(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        return self.parser.decode(image)

    def decode_with_bbox(self, image: np.ndarray) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]]]:
        return self.parser.decode_with_bbox(image)

    @property
    def last_qr_bbox(self) -> Optional[List[float]]:
        return self.parser.last_qr_bbox

    def parse_qr_string(self, qr_str: str) -> Optional[Dict[str, Any]]:
        return self.parser.parse_qr_string(qr_str)
