from typing import Optional, Dict, Any, List
from ocr.mrz_parser import MrzParser


class MrzEngine:
    """
    MRZ (Machine Readable Zone) Parser Engine for ICAO 9303 TD1 (CCCD gắn chip) back-side cards.
    """

    def __init__(self):
        self.parser = MrzParser()

    @staticmethod
    def compute_check_digit(mrz_substr: str) -> int:
        """
        Computes ICAO 9303 7-3-1 weighting check digit for MRZ field validation.
        """
        return MrzParser.compute_check_digit(mrz_substr)

    def parse(self, ocr_text_lines: List[str]) -> Optional[Dict[str, Any]]:
        """
        Parses OCR lines into structured MRZ dictionary with check-digit validation.
        """
        if not ocr_text_lines:
            return None
        return self.parser.parse_mrz_lines(ocr_text_lines)
