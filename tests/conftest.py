import cv2
import pytest
import numpy as np
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def dummy_card_front_image() -> np.ndarray:
    """
    Creates a synthetic BGR image representing a card front.
    """
    img = np.ones((630, 1000, 3), dtype=np.uint8) * 240
    cv2.circle(img, (180, 300), 80, (180, 200, 220), -1)
    cv2.circle(img, (150, 280), 10, (0, 0, 0), -1)
    cv2.circle(img, (210, 280), 10, (0, 0, 0), -1)
    cv2.putText(img, "CĂN CƯỚC CÔNG DÂN", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 180), 3)
    cv2.putText(img, "So / No.: 087204000897", (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Ho va ten: HUYNH QUANG LE", (350, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Ngay sinh: 04/10/2004", (350, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Gioi tinh: Nam", (350, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Co gia tri den: 04/10/2029", (100, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return img


@pytest.fixture
def dummy_card_back_image() -> np.ndarray:
    """
    Creates a synthetic BGR image representing a card back with MRZ.
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
    return img


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client
