import os
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Python eKYC Service"
    DEBUG: bool = False
    DEVICE: str = "cpu"  # "cuda" or "cpu"

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
