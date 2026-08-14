import os
import cv2
import pytest
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="session")
def test_images_dir() -> str:
    """
    Returns absolute path to the test images directory.
    """
    current_dir = Path(__file__).resolve().parent
    img_dir = current_dir / "image"
    return str(img_dir)


@pytest.fixture
def dummy_card_front_image() -> np.ndarray:
    """
    Creates a synthetic BGR image representing a card front with standard ASCII text
    to avoid OpenCV font rendering artifacts.
    """
    img = np.ones((630, 1000, 3), dtype=np.uint8) * 240
    # Synthetic face portrait region
    cv2.rectangle(img, (80, 180), (280, 440), (200, 210, 220), -1)
    cv2.circle(img, (180, 280), 50, (180, 190, 200), -1)
    cv2.circle(img, (160, 260), 8, (0, 0, 0), -1)
    cv2.circle(img, (200, 260), 8, (0, 0, 0), -1)
    cv2.ellipse(img, (180, 310), (20, 10), 0, 0, 180, (0, 0, 0), 2)

    # Standard ASCII text for card fields
    cv2.putText(img, "CAN CUOC CONG DAN", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 180), 3)
    cv2.putText(img, "So / No.: 087204000897", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Ho va ten / Full name: HUYNH QUANG LE", (350, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Ngay sinh / Date of birth: 04/10/2004", (350, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Gioi tinh / Sex: Nam", (350, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Quoc tich / Nationality: Viet Nam", (350, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "Co gia tri den / Date of expiry: 04/10/2029", (100, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


@pytest.fixture
def dummy_card_back_image() -> np.ndarray:
    """
    Creates a synthetic BGR image representing a card back with TD1 MRZ lines.
    """
    img = np.ones((630, 1000, 3), dtype=np.uint8) * 240
    cv2.putText(img, "IDVNM2040008978087204000897<<9", (50, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "0410047M2910046VNM<<<<<<<<<<<6", (50, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "HUYNH<<QUANG<LE<<<<<<<<<<<<<<", (50, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


@pytest.fixture
def dummy_selfie_image() -> np.ndarray:
    """
    Creates a synthetic selfie image with a drawn face.
    """
    img = np.ones((600, 600, 3), dtype=np.uint8) * 220
    cv2.circle(img, (300, 300), 120, (180, 200, 220), -1)
    cv2.circle(img, (260, 260), 15, (0, 0, 0), -1)
    cv2.circle(img, (340, 260), 15, (0, 0, 0), -1)
    cv2.ellipse(img, (300, 340), (35, 20), 0, 0, 180, (0, 0, 0), 3)
    return img


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client
