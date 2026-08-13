import os
from typing import List, Union, Dict, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Extended Canonical Field Keyword Configuration (9 Canonical Fields)
# Includes noisy OCR variants (e.g. /Piace of origin, Date.of.expiry, cogiatr, queguan)
FIELD_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "identityNumber": {
        "en": [
            "no",
            "no.",
            "number",
            "id no",
            "identity no",
            "personal identification number",
        ],
        "vi": [
            "số",
            "số định danh",
            "số định danh cá nhân",
            "số căn cước",
            "số cccd",
        ],
    },
    "fullName": {
        "en": [
            "full name",
            "name",
            "surname, given names",
        ],
        "vi": [
            "họ và tên",
            "họ tên",
            "họ và tên khai sinh",
            "họ, chữ đệm và tên khai sinh",
        ],
    },
    "dateOfBirth": {
        "en": [
            "date of birth",
            "birth date",
            "dob",
        ],
        "vi": [
            "ngày sinh",
            "ngày tháng năm sinh",
            "ngày, tháng, năm sinh",
        ],
    },
    "gender": {
        "en": [
            "sex",
            "gender",
        ],
        "vi": [
            "giới tính",
        ],
    },
    "nationality": {
        "en": [
            "nationality",
        ],
        "vi": [
            "quốc tịch",
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
        ],
        "vi": [
            "quê quán",
            "queguan",
            "quequan",
            "nơi đăng ký khai sinh",
        ],
    },
    "placeOfResidence": {
        "en": [
            "place of residence",
            "place.of.residence",
            "permanent residence",
            "permanent address",
        ],
        "vi": [
            "nơi thường trú",
            "nơi cư trú",
            "thường trú",
            "địa chỉ thường trú",
        ],
    },
    "dateOfIssue": {
        "en": [
            "date of issue",
            "issue date",
            "date month year",
            "date, month, year",
            "date,monthyear",
            "date,month,year",
            "monthyear",
        ],
        "vi": [
            "ngày cấp",
            "ngày tháng năm cấp",
            "ngày, tháng, năm cấp",
            "ngày tháng năm",
            "ngày, tháng, năm",
            "ngaythang,nam",
            "ngaythangnam",
        ],
    },
    "dateOfExpiry": {
        "en": [
            "date of expiry",
            "date.of.expiry",
            "dateofexpiry",
            "expiry date",
            "expiration date",
            "date of expiration",
        ],
        "vi": [
            "có giá trị đến",
            "co.gia.tri.den",
            "cogiatr",
            "cogiatrj",
            "cogiatrden",
            "ngày có giá trị đến",
            "ngày hết hạn",
        ],
    },
}


class Settings(BaseSettings):
    APP_NAME: str = "Python eKYC Service"
    DEBUG: bool = False
    DEVICE: str = "cpu"

    MODEL_DIR: str = "./models"
    INSIGHTFACE_MODEL_NAME: str = "buffalov"

    # Thresholds
    FACE_MATCH_THRESHOLD: float = 0.45
    LIVENESS_PASSIVE_THRESHOLD: float = 0.80
    OCR_MIN_CONFIDENCE: float = 0.85
    FULLNAME_FUZZY_THRESHOLD: float = 0.90
    IMAGE_BLUR_THRESHOLD: float = 100.0

    # Video Constraints
    MAX_VIDEO_DURATION_SEC: float = 10.0
    VIDEO_FRAME_SAMPLING_RATE: int = 10
    MAX_VIDEO_SIZE_MB: float = 20.0

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    @field_validator("ALLOWED_ORIGINS", mode="before")
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                import json
                return json.loads(v)
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
