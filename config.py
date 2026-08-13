from pydantic_settings import BaseSettings
from typing import List, Dict, Any


class Settings(BaseSettings):
    APP_NAME: str = "eKYC Service"
    API_V1_STR: str = "/api/v1"
    # CORS & Security Settings
    ALLOWED_ORIGINS: List[str] = ["*"]


    # OCR Settings
    OCR_LANG: str = "vi"
    OCR_USE_ANGLE_CLS: bool = True

    DEBUG: bool = True
    FULLNAME_FUZZY_THRESHOLD: float = 0.80
    FACE_SIMILARITY_THRESHOLD: float = 0.60
    IMAGE_BLUR_THRESHOLD: float = 100.0
    LIVENESS_BLUR_THRESHOLD: float = 100.0
    LIVENESS_EYE_RATIO_THRESHOLD: float = 0.20
    MAX_IMAGE_SIZE_MB: int = 10
    FACE_DETECTION_MODEL_PATH: str = "weights/face_detection.onnx"
    FACE_RECOGNITION_MODEL_PATH: str = "weights/face_recognition.onnx"




    class Config:
        case_sensitive = True


settings = Settings()

# Dictionary of Keywords for Card Processing Field Extraction & Boundary Engine
FIELD_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "identityNumber": {
        "en": [
            "no.",
            "number",
            "identity number",
            "personal identification number",
            "personal identificationnubo",
            "personal identification",
        ],
        "vi": [
            "số",
            "số / no",
            "so",
            "số định danh cá nhân",
            "so dinh danh ca nhan",
            "sadinh danh canhan",
            "dinh danh ca nhan",
        ],
    },
    "fullName": {
        "en": [
            "full name",
            "fullname",
            "full name:",
            "fuilname",
            "fuil name",
        ],
        "vi": [
            "họ và tên",
            "ho va ten",
            "họ, chữ đệm và tên khai sinh",
            "ho, chu dem va ten khai sinh",
            "ho.chidem va ten khal sinh",
            "ten khai sinh",
            "khai sinh",
        ],
    },
    "dateOfBirth": {
        "en": [
            "date of birth",
            "date.of.birth",
            "date of bth",
            "date.of.bth",
            "birth date",
            "dob",
        ],
        "vi": [
            "ngày sinh",
            "ngay sinh",
            "ngày, tháng, năm sinh",
            "ngay, thang, nam sinh",
            "ngaythang.am sinh",
            "ngaythang am sinh",
        ],
    },
    "placeOfOrigin": {
        "en": [
            "place of origin",
            "piace of origin",
            "place.of.origin",
            "piace.of.origin",
            "place of orig",
            "place of birth registration",
            "place of birth",
            "pace of brth",
            "pace.of.brth",
            "place.of.birth",
        ],
        "vi": [
            "quê quán",
            "queguan",
            "quequan",
            "nơi đăng ký khai sinh",
            "noi dang ky khai sinh",
            "roi dang ky khai sinh",
            "dang ky khai sinh",
        ],
    },
    "placeOfResidence": {
        "en": [
            "place of residence",
            "piace of residence",
            "place.of.residence",
            "piace.of.residence",
            "residence",
        ],
        "vi": [
            "nơi thường trú",
            "noi thuong tru",
            "nơi cư trú",
            "noi cu tru",
            "c/fcnct09",
            "cư trú",
            "thuong tru",
        ],
    },
    "dateOfExpiry": {
        "en": [
            "date of expiry",
            "date.of.expiry",
            "date expiry",
            "expiry date",
            "expiry",
            "oate ofexiry",
            "oate.of.exiry",
            "oate of expiry",
        ],
        "vi": [
            "có giá trị đến",
            "co gia tri den",
            "cogiatr",
            "cogiatri",
            "có giá trị đến:",
            "ngày, tháng, năm hết hạn",
            "ngay, thang, nam het han",
            "ngy.thang.nam het han",
            "ngay het han",
        ],
    },
    "dateOfIssue": {
        "en": [
            "date of issue",
            "date.of.issue",
            "issue date",
            "date, month, year",
            "date.month.year",
            "date month year",
            "dare of ssue",
            "dare.of.ssue",
            "date of ssue",
        ],
        "vi": [
            "ngày, tháng, năm cấp",
            "ngay, thang, nam cap",
            "ngay thang nam cap",
            "hgay.than.m cap",
            "hgay than m cap",
            "ngày cấp",
            "ngay cap",
            "ngày, tháng, năm",
            "ngay, thang, nam",
            "ngaythang,nam",
        ],
    },
}
