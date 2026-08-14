import os
import cv2
import pytest
from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from processors.card_processor import CardProcessor
from processors.card_validator import CardValidator


@pytest.fixture(scope="module")
def real_images_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "image")


def test_real_cccd_old_extraction(real_images_dir):
    """
    Test end-to-end card extraction on real CCCD Cũ images (cccd_c_mt.jpg & cccd_c_ms.jpg).
    """
    front_path = os.path.join(real_images_dir, "cccd_c_mt.jpg")
    back_path = os.path.join(real_images_dir, "cccd_c_ms.jpg")

    if not os.path.exists(front_path) or not os.path.exists(back_path):
        pytest.skip("Real CCCD old images not present in tests/image/")

    c_mt = cv2.imread(front_path)
    c_ms = cv2.imread(back_path)
    assert c_mt is not None and c_ms is not None

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
        meta,
        visual_regions
    ) = processor.process(c_mt, c_ms)

    # 1. Verification status
    valid, cross_val, errors = validator.validate(extracted_data, qr_data, mrz_data, card_type)
    assert valid is True, f"Card validation failed: {errors}"

    # 2. Extracted Data Accuracy
    assert extracted_data.identityNumber == "087204000897"
    assert "HUYNH QUANG LE" in extracted_data.fullName
    assert extracted_data.dateOfBirth == "2004-10-04"
    assert extracted_data.gender == "Nam"
    assert extracted_data.nationality == "Việt Nam"
    assert "Tan Binh" in extracted_data.placeOfOrigin or "Châu" in extracted_data.placeOfOrigin or "Chau" in extracted_data.placeOfOrigin
    assert "Ap Tay" in extracted_data.placeOfResidence
    assert extracted_data.dateOfIssue == "2021-03-30"
    assert extracted_data.dateOfExpiry == "2029-10-04"

    # 3. MRZ parsed properly
    assert mrz_data is not None
    assert mrz_data.get("identityNumber") == "087204000897"

    # 4. Separate Bounding Boxes generated
    fields_dict = {m.field: m for m in meta}
    assert "identityNumber" in fields_dict
    assert fields_dict["identityNumber"].label_box is not None
    assert fields_dict["identityNumber"].value_box is not None


def test_real_cccd_new_extraction(real_images_dir):
    """
    Test end-to-end card extraction on real Căn Cước Mới images (cccd_m_mt.jpg & cccd_m_ms.jpg).
    """
    front_path = os.path.join(real_images_dir, "cccd_m_mt.jpg")
    back_path = os.path.join(real_images_dir, "cccd_m_ms.jpg")

    if not os.path.exists(front_path) or not os.path.exists(back_path):
        pytest.skip("Real CCCD new images not present in tests/image/")

    m_mt = cv2.imread(front_path)
    m_ms = cv2.imread(back_path)
    assert m_mt is not None and m_ms is not None

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
        meta,
        visual_regions
    ) = processor.process(m_mt, m_ms)

    valid, cross_val, errors = validator.validate(extracted_data, qr_data, mrz_data, card_type)
    assert valid is True, f"Card validation failed: {errors}"

    # Accuracy checks
    assert extracted_data.identityNumber == "087203001336"
    assert "NGUYEN THANH TUNG" in extracted_data.fullName
    assert extracted_data.dateOfBirth == "2003-12-24"
    assert extracted_data.gender == "Nam"
    assert extracted_data.nationality == "Việt Nam"
    assert "Tan Phu Trung" in extracted_data.placeOfOrigin or "Tân Phú Trung" in extracted_data.placeOfOrigin
    assert "Phu Binh" in extracted_data.placeOfResidence or "Phú Bình" in extracted_data.placeOfResidence
    assert extracted_data.dateOfIssue == "2026-02-23"
    assert extracted_data.dateOfExpiry == "2028-12-24"

    # MRZ check
    assert mrz_data is not None
    assert mrz_data.get("identityNumber") == "087203001336"
