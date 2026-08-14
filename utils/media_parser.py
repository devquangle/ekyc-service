import re
import base64
from typing import Any, Dict, List, Optional, Union
from fastapi import Request, HTTPException, status
from config import settings
from utils.logger import logger


def decode_base64_media(data_str: str) -> Optional[bytes]:
    """
    Decodes standard Base64 or Data URI strings (e.g. data:image/jpeg;base64,...
    or data:video/mp4;base64,...), handling URL safe padding, newlines and whitespaces.
    """
    if not data_str or not isinstance(data_str, str):
        return None
    try:
        # 1. Strip Data URI header
        clean_str = re.sub(r'^data:(?:image|video)\/[a-zA-Z0-9+.-]+;base64,', '', data_str.strip())
        # 2. Remove all whitespaces, newlines, carriage returns, tabs
        clean_str = re.sub(r'[\r\n\t\s]+', '', clean_str)
        if not clean_str:
            return None

        # 3. Add missing padding if necessary
        missing_padding = len(clean_str) % 4
        if missing_padding:
            clean_str += '=' * (4 - missing_padding)

        decoded = base64.b64decode(clean_str, validate=False)
        return decoded if decoded and len(decoded) > 10 else None
    except Exception as e:
        logger.debug(f"[MEDIA_PARSER] Failed to decode base64 string: {str(e)}")
        return None


async def extract_raw_bytes(value: Any) -> Optional[bytes]:
    """
    Universally extracts raw bytes from UploadFile, bytes, or string/Base64.
    Automatically resets file pointer (seek(0)) before and after reading to prevent empty buffer issues.
    """
    if value is None:
        return None

    # 1. UploadFile or Async File-like object
    if hasattr(value, "read"):
        try:
            if hasattr(value, "seek"):
                try:
                    await value.seek(0)
                except Exception:
                    pass
            content = await value.read()
            if hasattr(value, "seek"):
                try:
                    await value.seek(0)
                except Exception:
                    pass
            return content if content and len(content) > 0 else None
        except Exception as e:
            logger.warning(f"[MEDIA_PARSER] Error reading file stream: {str(e)}")
            return None

    # 2. Raw bytes
    if isinstance(value, (bytes, bytearray)):
        return bytes(value) if len(value) > 0 else None

    # 3. String (Base64 or binary string)
    if isinstance(value, str) and len(value.strip()) > 0:
        # A. Try Base64 decoding
        decoded = decode_base64_media(value)
        if decoded:
            return decoded

        # B. Fallback: Raw binary encoded string
        for encoding in ('latin1', 'utf-8'):
            try:
                b = value.encode(encoding)
                if len(b) > 10 and (b.startswith(b'\xff\xd8') or b.startswith(b'\x89PNG') or b.startswith(b'RIFF')):
                    return b
            except Exception:
                pass

    return None


async def parse_media_payload(
    request: Request,
    alias_map: Dict[str, List[str]],
    max_size_mb: int = settings.MAX_IMAGE_SIZE_MB,
    required_fields: Optional[List[str]] = None,
    text_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Universal multi-format payload parser for eKYC endpoints.
    Seamlessly extracts media bytes and metadata from both `application/json` (Base64)
    and `multipart/form-data` / `application/x-www-form-urlencoded`.

    Args:
        request: FastAPI Request object.
        alias_map: Mapping of canonical field name to possible client parameter aliases.
                   Example: {"front_image": ["front_image", "frontImage", "card_front", "front"]}
        max_size_mb: Maximum allowed payload size in Megabytes.
        required_fields: List of canonical field names that must be present.
        text_fields: List of canonical fields that should be extracted as strings rather than bytes.

    Returns:
        Dictionary mapping canonical field names to their extracted bytes/strings.

    Raises:
        HTTPException(400): If JSON is malformed, payload exceeds size limit, or required fields are missing.
    """
    extracted: Dict[str, Any] = {}
    content_type = (request.headers.get("content-type") or "").lower()
    max_bytes = max_size_mb * 1024 * 1024
    text_field_set = set(text_fields or [])

    # 1. Parse JSON Body (Base64 payloads)
    if "application/json" in content_type:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("JSON root must be an object.")

            for canonical_name, aliases in alias_map.items():
                for alias in aliases:
                    if alias in body and body[alias] is not None:
                        val = body[alias]
                        if canonical_name in text_field_set:
                            extracted[canonical_name] = str(val).strip() if val else None
                        else:
                            extracted[canonical_name] = await extract_raw_bytes(val)
                        if extracted[canonical_name] is not None:
                            break
        except Exception as e:
            logger.warning(f"[MEDIA_PARSER] Malformed JSON request body: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {str(e)}"
            )

    # 2. Parse Multipart Form-Data / Form URL-Encoded
    else:
        try:
            form = await request.form()
            for canonical_name, aliases in alias_map.items():
                for alias in aliases:
                    if alias in form and form[alias] is not None:
                        val = form[alias]
                        if canonical_name in text_field_set:
                            extracted[canonical_name] = str(val).strip() if val else None
                        else:
                            extracted[canonical_name] = await extract_raw_bytes(val)
                        if extracted[canonical_name] is not None:
                            break
        except Exception as e:
            logger.warning(f"[MEDIA_PARSER] Error reading form data: {str(e)}")

    # 3. Validate Required Fields & Payload Sizes
    if required_fields:
        missing = [f for f in required_fields if extracted.get(f) is None]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required field(s): {', '.join(missing)}"
            )

    for field_name, val in extracted.items():
        if isinstance(val, (bytes, bytearray)) and len(val) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Payload for '{field_name}' exceeds maximum allowed size ({max_size_mb} MB)."
            )

    return extracted
