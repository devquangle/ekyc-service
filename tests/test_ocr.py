import cv2
import numpy as np
from core.ocr_engine import OcrLine
from processors.card_processor import CardProcessor, normalize_unicode, normalize_address
from schemas.card import ExtractedCardData, FieldMetadata


def test_1_cccd_new_front_and_back_card_type():
    processor = CardProcessor(None, None, None)
    front_lines = [
        OcrLine(text="CAN CU'O'C", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="Sadinh danh canhan", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]])
    ]
    back_lines = [
        OcrLine(text="Roi dang ky khai sinh / Pace of brth", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]])
    ]
    f_kws = processor.detect_all_keywords(front_lines)
    b_kws = processor.detect_all_keywords(back_lines)
    c_type, conf = processor._detect_card_type(front_lines, back_lines, f_kws, b_kws)
    assert c_type == "CCCD_NEW"
    assert conf >= 0.95


def test_2_cccd_old_front_and_back_card_type():
    processor = CardProcessor(None, None, None)
    front_lines = [
        OcrLine(text="CĂN CƯỚC CÔNG DÂN", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="Quê quán / Place of origin", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]]),
        OcrLine(text="Nơi thường trú / Place of residence", confidence=0.98, boundingBox=[[100, 180], [300, 180], [300, 210], [100, 210]])
    ]
    back_lines = []
    f_kws = processor.detect_all_keywords(front_lines)
    b_kws = processor.detect_all_keywords(back_lines)
    c_type, conf = processor._detect_card_type(front_lines, back_lines, f_kws, b_kws)
    assert c_type == "CCCD_OLD"
    assert conf >= 0.95


def test_3_cccd_new_with_mrz_still_classified_as_new():
    processor = CardProcessor(None, None, None)
    front_lines = [
        OcrLine(text="CĂN CƯỚC", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="Số định danh cá nhân", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]])
    ]
    back_lines = [
        OcrLine(text="IDVNM2030013364087203001336<<5", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]])
    ]
    f_kws = processor.detect_all_keywords(front_lines)
    b_kws = processor.detect_all_keywords(back_lines)
    c_type, conf = processor._detect_card_type(front_lines, back_lines, f_kws, b_kws)
    assert c_type == "CCCD_NEW"


def test_4_cccd_new_unaccented_keywords():
    processor = CardProcessor(None, None, None)
    front_lines = [
        OcrLine(text="CAN CUOC", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="So dinh danh ca nhan", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]])
    ]
    back_lines = []
    f_kws = processor.detect_all_keywords(front_lines)
    b_kws = processor.detect_all_keywords(back_lines)
    c_type, conf = processor._detect_card_type(front_lines, back_lines, f_kws, b_kws)
    assert c_type == "CCCD_NEW"


def test_5_cccd_new_distinctive_keyword_set():
    processor = CardProcessor(None, None, None)
    front_lines = [
        OcrLine(text="CĂN CƯỚC", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="Số định danh cá nhân", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]])
    ]
    back_lines = [
        OcrLine(text="Nơi cư trú", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="Nơi đăng ký khai sinh", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]])
    ]
    f_kws = processor.detect_all_keywords(front_lines)
    b_kws = processor.detect_all_keywords(back_lines)
    c_type, conf = processor._detect_card_type(front_lines, back_lines, f_kws, b_kws)
    assert c_type == "CCCD_NEW"
    assert conf >= 0.95


def test_6_cccd_old_distinctive_keyword_set():
    processor = CardProcessor(None, None, None)
    front_lines = [
        OcrLine(text="CĂN CƯỚC CÔNG DÂN", confidence=0.98, boundingBox=[[100, 100], [400, 100], [400, 130], [100, 130]]),
        OcrLine(text="Quê quán", confidence=0.98, boundingBox=[[100, 140], [300, 140], [300, 170], [100, 170]]),
        OcrLine(text="Nơi thường trú", confidence=0.98, boundingBox=[[100, 180], [300, 180], [300, 210], [100, 210]])
    ]
    back_lines = []
    f_kws = processor.detect_all_keywords(front_lines)
    b_kws = processor.detect_all_keywords(back_lines)
    c_type, conf = processor._detect_card_type(front_lines, back_lines, f_kws, b_kws)
    assert c_type == "CCCD_OLD"
    assert conf >= 0.95
