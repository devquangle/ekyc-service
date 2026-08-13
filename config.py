from pydantic_settings import BaseSettings
from typing import List, Dict, Any


class Settings(BaseSettings):
    APP_NAME: str = "eKYC Service"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = True
    DEVICE: str = "cpu"

    # CORS & Security Settings
    ALLOWED_ORIGINS: List[str] = ["*"]

    # OCR Settings
    OCR_LANG: str = "vi"
    OCR_USE_ANGLE_CLS: bool = True
    FULLNAME_FUZZY_THRESHOLD: float = 0.80

    # Face Matching Settings
    FACE_MISMATCH_THRESHOLD: float = 0.45
    FACE_MATCH_THRESHOLD: float = 0.60
    FACE_SIMILARITY_THRESHOLD: float = 0.60
    MIN_CARD_FACE_WIDTH: int = 30
    MIN_CARD_FACE_HEIGHT: int = 30
    MIN_SELFIE_FACE_WIDTH: int = 40
    MIN_SELFIE_FACE_HEIGHT: int = 40
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    MODEL_DIR: str = "weights"
    FACE_DETECTION_MODEL_PATH: str = "weights/face_detection.onnx"
    FACE_RECOGNITION_MODEL_PATH: str = "weights/face_recognition.onnx"

    # Liveness & Image Settings
    IMAGE_BLUR_THRESHOLD: float = 100.0
    LIVENESS_BLUR_THRESHOLD: float = 100.0
    LIVENESS_EYE_RATIO_THRESHOLD: float = 0.20
    LIVENESS_PASSIVE_THRESHOLD: float = 0.50
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_VIDEO_SIZE_MB: int = 50
    MAX_VIDEO_DURATION_SEC: float = 30.0
    VIDEO_FRAME_SAMPLING_RATE: int = 10
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]

    class Config:
        case_sensitive = True


settings = Settings()

# Official Physical Card Field Labels (CCCD Mới 2023 & CCCD Cũ 2021)
FIELD_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "identityNumber": {
        "en": [
            "no.",
            "number",
            "identity number",
            "personal identification number",
        ],
        "vi": [
            "số định danh cá nhân / no:",
            "số định danh cá nhân",
            "số / no.:",
            "số / no:",
            "số",
        ],
    },
    "fullName": {
        "en": [
            "surname, given names",
            "full name",
        ],
        "vi": [
            "họ, chữ đệm và tên khai sinh / surname, given names:",
            "họ, chữ đệm và tên khai sinh",
            "họ và tên / full name:",
            "họ và tên",
        ],
    },
    "dateOfBirth": {
        "en": [
            "date of birth",
        ],
        "vi": [
            "ngày, tháng, năm sinh / date of birth:",
            "ngày, tháng, năm sinh",
            "ngày sinh / date of birth:",
            "ngày sinh",
        ],
    },
    "placeOfOrigin": {
        "en": [
            "place of birth registration",
            "place of origin",
        ],
        "vi": [
            "nơi đăng ký khai sinh / place of birth registration:",
            "nơi đăng ký khai sinh",
            "quê quán / place of origin:",
            "quê quán",
        ],
    },
    "placeOfResidence": {
        "en": [
            "place of residence",
        ],
        "vi": [
            "nơi cư trú / place of residence:",
            "nơi cư trú",
            "nơi thường trú / place of residence:",
            "nơi thường trú",
        ],
    },
    "dateOfExpiry": {
        "en": [
            "date of expiry",
        ],
        "vi": [
            "có giá trị đến / date of expiry",
            "có giá trị đến",
        ],
    },
    "dateOfIssue": {
        "en": [
            "date, month, year",
        ],
        "vi": [
            "ngày, tháng, năm cấp / date, month, year",
            "ngày, tháng, năm cấp",
            "ngày, tháng, năm / date, month, year:",
            "ngày, tháng, năm",
        ],
    },
}
