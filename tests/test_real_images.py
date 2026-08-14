import os
import cv2
import pytest
from pathlib import Path

from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from processors.card_processor import CardProcessor
from processors.card_validator import CardValidator
from schemas.enums import CardType


@pytest.fixture(scope="module")
def real_images_dir() -> Path:
    current_dir = Path(__file__).resolve().parent
    return current_dir / "image"


def test_real_cccd_old_extraction(real_images_dir: Path):
    """
    Test end-to-end card extraction on real CCCD Cũ images (cccd_c_mt.jpg & cccd_c_ms.jpg).
    CCCD Cũ has OCR text on front, and MRZ TD1 on back.
    """
    front_path = real_images_dir / "cccd_c_mt.jpg"
    back_path = real_images_dir / "cccd_c_ms.jpg"

    if not front_path.exists() or not back_path.exists():
        pytest.skip(f"Real CCCD old images not present in {real_images_dir}")

    c_mt = cv2.imread(str(front_path))
    c_ms = cv2.imread(str(back_path))
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
    assert str(card_type) in ["CCCD_OLD", CardType.CCCD_OLD.value]

    # 2. Extracted Data Accuracy
    assert extracted_data.identityNumber == "087204000897"
    assert "HUYNH QUANG LE" in (extracted_data.fullName or "")
    assert extracted_data.dateOfBirth == "2004-10-04"
    assert extracted_data.gender == "Nam"
    assert extracted_data.nationality == "Việt Nam"
    if extracted_data.placeOfOrigin:
        assert "Tân Bình" in extracted_data.placeOfOrigin or "Đồng Tháp" in extracted_data.placeOfOrigin
    assert "Ấp Tây" in (extracted_data.placeOfResidence or "") or "Đồng Tháp" in (extracted_data.placeOfResidence or "")
    assert extracted_data.dateOfExpiry == "2029-10-04"

    # 3. MRZ parsed properly on CCCD Old
    assert mrz_data is not None
    assert mrz_data.get("identityNumber") == "087204000897"

    # 4. Separate Bounding Boxes generated
    fields_dict = {m.field: m for m in meta}
    assert "identityNumber" in fields_dict
    assert fields_dict["identityNumber"].label_box is not None or fields_dict["identityNumber"].value_box is not None


def test_real_cccd_new_extraction(real_images_dir: Path):
    """
    Test end-to-end card extraction on real Căn Cước Mới 2024 images (cccd_m_mt.jpg & cccd_m_ms.jpg).
    Thẻ Căn cước mới 2024 không có MRZ ở mặt sau, mà có mã QR Code ở mặt sau.
    """
    front_path = real_images_dir / "cccd_m_mt.jpg"
    back_path = real_images_dir / "cccd_m_ms.jpg"

    if not front_path.exists() or not back_path.exists():
        pytest.skip(f"Real CCCD new images not present in {real_images_dir}")

    m_mt = cv2.imread(str(front_path))
    m_ms = cv2.imread(str(back_path))
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
    assert str(card_type) in ["CCCD_OLD", "CCCD_NEW", CardType.CCCD_OLD.value, CardType.CCCD_NEW.value]

    # Accuracy checks
    assert extracted_data.identityNumber == "087203001336"
    assert "NGUYEN THANH TUNG" in (extracted_data.fullName or "")
    assert extracted_data.dateOfBirth == "2003-12-24"
    assert extracted_data.gender == "Nam"
    if extracted_data.placeOfOrigin:
        assert "Tân Phú Trung" in extracted_data.placeOfOrigin or "Đồng Tháp" in extracted_data.placeOfOrigin
    assert extracted_data.dateOfExpiry == "2028-12-24"

    # MRZ or QR data verification
    assert mrz_data is not None or qr_data is not None
    if mrz_data:
        assert mrz_data.get("identityNumber") == "087203001336"
    if qr_data:
        assert qr_data.get("identityNumber") == "087203001336"
