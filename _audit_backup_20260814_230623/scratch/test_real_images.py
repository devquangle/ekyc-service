import os
import sys
import cv2
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from processors.card_processor import CardProcessor
from processors.card_validator import CardValidator

def run():
    img_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests", "image")
    c_mt_path = os.path.join(img_dir, "cccd_c_mt.jpg")
    c_ms_path = os.path.join(img_dir, "cccd_c_ms.jpg")
    m_mt_path = os.path.join(img_dir, "cccd_m_mt.jpg")
    m_ms_path = os.path.join(img_dir, "cccd_m_ms.jpg")

    ocr_engine = OcrEngine()
    qr_engine = QrEngine()
    mrz_engine = MrzEngine()
    processor = CardProcessor(ocr_engine, qr_engine, mrz_engine)
    validator = CardValidator()

    print("==================================================")
    print("TEST 1: CCCD CŨ (cccd_c_mt.jpg + cccd_c_ms.jpg)")
    print("==================================================")
    c_mt = cv2.imread(c_mt_path)
    c_ms = cv2.imread(c_ms_path)
    if c_mt is not None:
        (
            card_type,
            conf,
            extracted_data,
            qr_data,
            mrz_data,
            quality,
            meta,
            visual_regions
        ) = processor.process(c_mt, c_ms)

        valid, cross_val, errors = validator.validate(extracted_data, qr_data, mrz_data, card_type)

        print(f"Card Type: {card_type} (conf: {conf})")
        print(f"Card Verified: {valid}")
        print("Extracted Data:")
        print(json.dumps(extracted_data.model_dump(), ensure_ascii=False, indent=2))
        print("Visual Regions:")
        print(json.dumps(visual_regions.model_dump(), ensure_ascii=False, indent=2))
        print("Field Metadata (Bboxes & Text):")
        for m in meta:
            print(f" - {m.field:18}: val={m.value!r:30} labelBox={m.label_box} valBox={m.value_box}")

    print("\n==================================================")
    print("TEST 2: CĂN CƯỚC MỚI (cccd_m_mt.jpg + cccd_m_ms.jpg)")
    print("==================================================")
    m_mt = cv2.imread(m_mt_path)
    m_ms = cv2.imread(m_ms_path)
    if m_mt is not None:
        (
            card_type_m,
            conf_m,
            extracted_data_m,
            qr_data_m,
            mrz_data_m,
            quality_m,
            meta_m,
            visual_regions_m
        ) = processor.process(m_mt, m_ms)

        valid_m, cross_val_m, errors_m = validator.validate(extracted_data_m, qr_data_m, mrz_data_m, card_type_m)

        print(f"Card Type: {card_type_m} (conf: {conf_m})")
        print(f"Card Verified: {valid_m}")
        print("Extracted Data:")
        print(json.dumps(extracted_data_m.model_dump(), ensure_ascii=False, indent=2))
        print("Visual Regions:")
        print(json.dumps(visual_regions_m.model_dump(), ensure_ascii=False, indent=2))
        print("Field Metadata (Bboxes & Text):")
        for m in meta_m:
            print(f" - {m.field:18}: val={m.value!r:30} labelBox={m.label_box} valBox={m.value_box}")

if __name__ == "__main__":
    run()
