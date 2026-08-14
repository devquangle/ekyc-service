import pytest
import numpy as np
import base64
import cv2
from io import BytesIO
from unittest.mock import AsyncMock

from utils.image_utils import crop_image, decode_image_bytes, resize_maintain_aspect, check_image_quality
from utils.card_aligner import CardAligner, _four_point_transform
from utils.media_parser import decode_base64_media, extract_raw_bytes
from utils.text_utils import normalize_date, compare_names, remove_vietnamese_accents, normalize_text
from utils.vietnamese_administrative_restorer import VietnameseAdministrativeRestorer


def test_crop_image_safe_float_casting_and_clipping():
    """
    Ensure crop_image safely casts float bbox coordinates to integer and handles bounds.
    """
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[20:80, 30:150] = 255

    # Float coordinates with rounding
    cropped = crop_image(img, [30.2, 20.7, 149.8, 79.9])
    assert cropped is not None
    assert cropped.shape[0] == 59  # 80 - 21 = 59
    assert cropped.shape[1] == 120 # 150 - 30 = 120

    # Inverted coordinates [x2, y2, x1, y1]
    cropped_inv = crop_image(img, [150.0, 80.0, 30.0, 20.0])
    assert cropped_inv is not None
    assert cropped_inv.shape[0] == 60
    assert cropped_inv.shape[1] == 120

    # Out of bounds coordinates
    cropped_oob = crop_image(img, [-50.0, -20.0, 300.0, 200.0])
    assert cropped_oob is not None
    assert cropped_oob.shape == (100, 200, 3)

    # Empty / None cases
    assert crop_image(None, [10, 10, 50, 50]) is None
    assert crop_image(img, None) is None
    assert crop_image(img, [10, 10]) is None


def test_decode_base64_media_whitespace_newlines_and_padding():
    """
    Ensure decode_base64_media strips newlines, spaces, and fixes missing padding.
    """
    raw_data = b"Hello eKYC World! This is a test image content."
    b64_str = base64.b64encode(raw_data).decode('utf-8')

    # Add Data URI prefix, line breaks, carriage returns, whitespaces, and remove padding '='
    unpadded_b64 = b64_str.rstrip('=')
    dirty_b64 = f"data:image/jpeg;base64,\r\n  {unpadded_b64[:10]}\n\r\t {unpadded_b64[10:]}   "

    decoded = decode_base64_media(dirty_b64)
    assert decoded == raw_data


@pytest.mark.asyncio
async def test_extract_raw_bytes_upload_file_stream_reset():
    """
    Ensure extract_raw_bytes reads UploadFile and resets seek(0).
    """
    mock_upload = AsyncMock()
    mock_upload.read = AsyncMock(return_value=b"test_upload_file_bytes")
    mock_upload.seek = AsyncMock()

    content = await extract_raw_bytes(mock_upload)
    assert content == b"test_upload_file_bytes"
    # Verify seek(0) called before and after
    assert mock_upload.seek.call_count == 2
    mock_upload.seek.assert_called_with(0)


def test_normalize_date_gregorian_validity():
    """
    Ensure normalize_date parses valid dates and rejects impossible dates (e.g. 30/02, 31/04).
    """
    # Valid dates
    assert normalize_date("04/10/2004") == "2004-10-04"
    assert normalize_date("29/02/2024") == "2024-02-29" # Leap year
    assert normalize_date("24122003") == "2003-12-24"   # 8-digit
    assert normalize_date("041004") == "2004-10-04"     # 6-digit MRZ

    # Impossible dates (must return None)
    assert normalize_date("30/02/2024") is None  # February 30th does not exist
    assert normalize_date("31/04/2023") is None  # April has only 30 days
    assert normalize_date("31/06/2023") is None  # June has only 30 days
    assert normalize_date("29/02/2023") is None  # 2023 is not a leap year


def test_compare_names_order_insensitive_and_abbreviations():
    """
    Ensure compare_names handles out-of-order words, abbreviations, and exact matches.
    """
    # Exact match
    assert compare_names("HUYNH QUANG LE", "HUYNH QUANG LE") == 1.0

    # Order-insensitive match
    assert compare_names("NGUYEN VAN A", "A NGUYEN VAN") == 1.0
    assert compare_names("LE QUANG HUYNH", "HUYNH QUANG LE") == 1.0

    # Abbreviation / Initial match
    score_abbrev = compare_names("H QUANG LE", "HUYNH QUANG LE")
    assert score_abbrev >= 0.85

    score_abbrev2 = compare_names("N VAN A", "NGUYEN VAN A")
    assert score_abbrev2 >= 0.85

    # Completely different names
    score_diff = compare_names("TRAN THI UT", "NGUYEN THANH TUNG")
    assert score_diff < 0.50


def test_card_aligner_aspect_ratio_and_contour():
    """
    Ensure CardAligner validates ID-1 aspect ratio before warping.
    """
    aligner = CardAligner(model_path="weights/non_existent.onnx")

    # Create dummy synthetic image with rotated rectangle
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    # Draw high contrast ID card rectangle (ratio ~ 1.58)
    cv2.rectangle(img, (100, 100), (600, 415), (255, 255, 255), -1)

    aligned, was_aligned = aligner.align(img)
    assert aligned is not None
    assert aligned.shape[1] >= aligned.shape[0]  # Standard horizontal orientation


def test_vietnamese_administrative_restorer_short_words_and_structures():
    """
    Ensure restorer handles short words (Ba Vì, Huế, Tổ 9, Ấp 1) accurately.
    """
    # Short province names
    assert VietnameseAdministrativeRestorer.fuzzy_match("hue", {"hue": "Huế", "ha noi": "Hà Nội"})[1] == "Huế"

    # Short district / commune names
    res_bavi = VietnameseAdministrativeRestorer.restore_address_diacritics("Ba Vi, Ha Noi")
    assert "Ba Vì" in res_bavi
    assert "Hà Nội" in res_bavi

    # Group / Hamlet numbers
    res_to9 = VietnameseAdministrativeRestorer.restore_address_diacritics("To 9, Ap Phu Binh, Tan Phu Trung, Dong Thap")
    assert "Tổ 9" in res_to9
    assert "Ấp Phú Bình" in res_to9
    assert "Tân Phú Trung" in res_to9
    assert "Đồng Tháp" in res_to9
