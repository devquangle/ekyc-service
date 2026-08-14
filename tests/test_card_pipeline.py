import os
import cv2
import pytest
from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from processors.card_processor import CardProcessor
from processors.card_validator import CardValidator


def get_image_path(filename: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "image", filename)


def test_card_pipeline_cccd_old_flow():
    front_path = get_image_path("cccd_c_mt.jpg")
    back_path = get_image_path("cccd_c_ms.jpg")

    if not os.path.exists(front_path) or not os.path.exists(back_path):
        pytest.skip("CCCD old images not present in tests/image/")

    front_img = cv2.imread(front_path)
    back_img = cv2.imread(back_path)

    processor = CardProcessor(OcrEngine(), QrEngine(), MrzEngine())
    validator = CardValidator()

    card_type, conf, extracted_data, qr_data, mrz_data, quality, meta, visual_regions = processor.process(front_img, back_img)
    valid, cross_val, errors = validator.validate(extracted_data, qr_data, mrz_data, card_type)

    assert valid is True
    assert extracted_data.identityNumber == "087204000897"
    assert "HUYNH QUANG LE" in extracted_data.fullName
    assert extracted_data.gender == "Nam"


def test_card_pipeline_cccd_new_flow():
    front_path = get_image_path("cccd_m_mt.jpg")
    back_path = get_image_path("cccd_m_ms.jpg")

    if not os.path.exists(front_path) or not os.path.exists(back_path):
        pytest.skip("CCCD new images not present in tests/image/")

    front_img = cv2.imread(front_path)
    back_img = cv2.imread(back_path)

    processor = CardProcessor(OcrEngine(), QrEngine(), MrzEngine())
    validator = CardValidator()

    card_type, conf, extracted_data, qr_data, mrz_data, quality, meta, visual_regions = processor.process(front_img, back_img)
    valid, cross_val, errors = validator.validate(extracted_data, qr_data, mrz_data, card_type)

    assert valid is True
    assert extracted_data.identityNumber == "087203001336"
    assert "NGUYEN THANH TUNG" in extracted_data.fullName
    assert extracted_data.gender == "Nam"
