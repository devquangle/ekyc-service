import logging
import re
import sys
from typing import Any

# Ensure Windows stdout uses UTF-8 encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass


class MaskingFormatter(logging.Formatter):
    """
    Custom Logging Formatter to automatically mask PII in log messages.
    """

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return mask_pii(original)


def mask_pii(text: str) -> str:
    """
    Masks PII elements like 12-digit CCCD, 9-digit CMND numbers, and date formats.
    """
    if not isinstance(text, str):
        return str(text)

    # Mask 12-digit identity number (CCCD): 087204000897 -> 087204****97
    text = re.sub(r'\b(\d{6})\d{4}(\d{2})\b', r'\1****\2', text)

    # Mask 9-digit identity number (CMND): 183456789 -> 183****89
    text = re.sub(r'\b(\d{3})\d{4}(\d{2})\b', r'\1****\2', text)

    # Mask ISO Date format: YYYY-MM-DD -> ****-**-DD
    text = re.sub(r'\b\d{4}-\d{2}-(\d{2})\b', r'****-**-\1', text)

    # Mask Vietnamese Date format: DD/MM/YYYY or DD-MM-YYYY -> **/**/YYYY
    text = re.sub(r'\b\d{2}[/.-]\d{2}[/.-](\d{4})\b', r'**/**/\1', text)

    return text


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ekyc_service")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        if hasattr(handler, "setStream"):
            handler.stream = sys.stdout
        formatter = MaskingFormatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()
