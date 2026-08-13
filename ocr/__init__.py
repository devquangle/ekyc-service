from ocr.detector import TextDetector, OCRText
from ocr.layout_parser import LayoutParser, LayoutLine
from ocr.label_matcher import LabelMatcher, FIELD_LABELS
from ocr.normalizer import (
    normalize_unicode,
    normalize_gender,
    parse_date,
    normalize_identity_number,
    normalize_full_name,
    normalize_address,
    normalize_address_for_compare,
    normalize_text_for_compare,
)
from ocr.field_extractor import FieldExtractor, ExtractedField
from ocr.mrz_parser import MrzParser
from ocr.qr_parser import QrParser
from ocr.validators import CardTypeClassifier, CrossValidator

FIELD_KEYWORDS = FIELD_LABELS

__all__ = [
    "TextDetector",
    "OCRText",
    "LayoutParser",
    "LayoutLine",
    "LabelMatcher",
    "FIELD_LABELS",
    "FIELD_KEYWORDS",
    "normalize_unicode",
    "normalize_gender",
    "parse_date",
    "normalize_identity_number",
    "normalize_full_name",
    "normalize_address",
    "normalize_address_for_compare",
    "normalize_text_for_compare",
    "FieldExtractor",
    "ExtractedField",
    "MrzParser",
    "QrParser",
    "CardTypeClassifier",
    "CrossValidator",
]
