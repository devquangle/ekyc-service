import os
import sys
import cv2
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from core.ocr_engine import OcrEngine
from ocr.layout_parser import LayoutParser
from ocr.field_extractor import FieldExtractor

def inspect():
    img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "image")
    m_ms_path = os.path.join(img_dir, "cccd_m_ms.jpg")
    m_mt_path = os.path.join(img_dir, "cccd_m_mt.jpg")

    m_ms = cv2.imread(m_ms_path)
    m_mt = cv2.imread(m_mt_path)

    ocr_engine = OcrEngine()
    parser = LayoutParser()
    extractor = FieldExtractor()

    print("=== BACK TOKENS (cccd_m_ms.jpg) ===")
    back_tokens = ocr_engine.detect_tokens(m_ms)
    for i, t in enumerate(back_tokens):
        print(f"{i:2d}: bbox={t.bbox} text={t.text!r} conf={t.confidence:.2f}")

    print("\n=== BACK LAYOUT LINES ===")
    lines = parser.group_tokens_into_lines(back_tokens)
    for i, l in enumerate(lines):
        print(f"Line {i:2d} (y={l.center_y:.1f}, x={l.center_x:.1f}): {l.text!r}")

    print("\n=== BACK EXTRACTED FIELDS ===")
    back_fields = extractor.extract_all_fields(back_tokens)
    for k, v in back_fields.items():
        print(f"{k}: val={v.value!r} raw={v.rawText!r} keyword={v.keyword!r} label_box={v.label_box} val_box={v.value_box}")

    print("\n=== FRONT EXTRACTED FIELDS (cccd_m_mt.jpg) ===")
    front_tokens = ocr_engine.detect_tokens(m_mt)
    front_fields = extractor.extract_all_fields(front_tokens)
    for k, v in front_fields.items():
        print(f"{k}: val={v.value!r} raw={v.rawText!r} keyword={v.keyword!r} label_box={v.label_box} val_box={v.value_box}")

if __name__ == "__main__":
    inspect()
