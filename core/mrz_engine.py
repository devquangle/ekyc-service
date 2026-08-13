from typing import Optional, Dict, Any, List
from ocr.mrz_parser import MrzParser


class MrzEngine:
    """
    MRZ Parser Wrapper delegating to ocr.mrz_parser.MrzParser.
    """

    def __init__(self):
        self.parser = MrzParser()

    @staticmethod
    def compute_check_digit(mrz_substr: str) -> int:
        return MrzParser.compute_check_digit(mrz_substr)

    def parse(self, ocr_text_lines: List[str]) -> Optional[Dict[str, Any]]:
        return self.parser.parse_mrz_lines(ocr_text_lines)
