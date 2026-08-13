import logging
import re
import sys
from typing import Any


class MaskingFormatter(logging.Formatter):
    """
    Custom Logging Formatter to automatically mask PII in log messages.
    """

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return mask_pii(original)


def mask_pii(text: str) -> str:
    """
    Masks PII elements like identity numbers, full names, dates.
    """
    if not isinstance(text, str):
        return str(text)

    # Mask 12-digit identity number: 001095001234 -> 001095****34
    text = re.sub(r'\b(\d{6})\d{4}(\d{2})\b', r'\1****\2', text)

    # Mask ISO Date format in logs if specified: YYYY-MM-DD -> ****-**-DD
    text = re.sub(r'\b\d{4}-\d{2}-(\d{2})\b', r'****-**-\1', text)

    return text


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ekyc_service")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = MaskingFormatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
