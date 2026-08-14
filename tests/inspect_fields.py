import os
import sys
import cv2
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from processors.card_processor import CardProcessor
from processors.card_validator import CardValidator



def get_image_path(filename: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "image", filename)


def inspect_card(front_filename: str, back_filename: str):
    front_path = get_image_path(front_filename)
    back_path = get_image_path(back_filename)

    if not os.path.exists(front_path) or not os.path.exists(back_path):
        print(f"[WARN] Image files not found: {front_path}, {back_path}")
        return

    front_img = cv2.imread(front_path)
    back_img = cv2.imread(back_path)

    ocr_engine = OcrEngine()
    qr_engine = QrEngine()
    mrz_engine = MrzEngine()
    processor = CardProcessor(ocr_engine, qr_engine, mrz_engine)
    validator = CardValidator()

    (
        card_type,
        conf,
        extracted_data,
        qr_data,
        mrz_data,
        quality,
        metadata,
        visual_regions
    ) = processor.process(front_img, back_img)

    valid, cross_val, errors = validator.validate(extracted_data, qr_data, mrz_data, card_type)

    print(f"\n=======================================================")
    print(f" CARD INSPECTION: {front_filename} & {back_filename}")
    print(f" Card Type: {card_type} (Confidence: {conf})")
    print(f" Card Verified: {valid}")
    print(f" Errors / Warnings: {errors}")
    print(f"=======================================================")

    print("\n--- EXTRACTED CARD DATA ---")
    print(json.dumps(extracted_data.model_dump(), indent=2, ensure_ascii=False))

    print("\n--- VISUAL REGIONS ---")
    print(json.dumps(visual_regions.model_dump() if visual_regions else {}, indent=2))

    print("\n--- FIELD METADATA & BOUNDING BOXES ---")
    for meta in metadata:
        print(
            f"Field: {meta.field:18} | Value: {str(meta.value):30} | "
            f"LabelBox: {meta.label_box} | ValueBox: {meta.value_box}"
        )

    print("\n--- CROSS VALIDATION DETAILS ---")
    for detail in cross_val.details:
        print(
            f"Field: {detail.fieldName:18} | Status: {detail.status:12} | "
            f"OCR: {str(detail.ocrValue):20} | QR: {str(detail.qrValue):20} | MRZ: {str(detail.mrzValue):20}"
        )


if __name__ == "__main__":
    print("[1] Inspecting CCCD Old...")
    inspect_card("cccd_c_mt.jpg", "cccd_c_ms.jpg")

    print("\n[2] Inspecting CCCD New (2024)...")
    inspect_card("cccd_m_mt.jpg", "cccd_m_ms.jpg")
