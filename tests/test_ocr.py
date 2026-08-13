import cv2
import numpy as np
from utils.image_utils import check_image_quality, decode_image_bytes, crop_image
from utils.text_utils import normalize_text, normalize_date, remove_vietnamese_accents


def test_image_quality_check():
    # Sharp image
    sharp_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.putText(sharp_img, "TEST OCR TEXT", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    is_blur, has_glare = check_image_quality(sharp_img)
    assert not is_blur

    # Blurry image
    blurry_img = cv2.GaussianBlur(sharp_img, (21, 21), 0)
    is_blur_b, _ = check_image_quality(blurry_img)
    assert is_blur_b


def test_text_utils_normalization():
    assert normalize_text("  NGUYEN  THANH   TUNG  ") == "NGUYEN THANH TUNG"
    assert remove_vietnamese_accents("HUỲNH QUANG LÊ") == "HUYNH QUANG LE"
    assert normalize_date("04/10/2004") == "2004-10-04"
    assert normalize_date("04102004") == "2004-10-04"
    assert normalize_date("041004") == "2004-10-04"


def test_decode_image_bytes(dummy_card_front_image):
    _, encoded = cv2.imencode(".jpg", dummy_card_front_image)
    bytes_data = encoded.tobytes()
    decoded = decode_image_bytes(bytes_data)
    assert decoded is not None
    assert decoded.shape == dummy_card_front_image.shape
