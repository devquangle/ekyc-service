import base64
import re
from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from api.dependencies import get_orchestrator
from schemas.card import CardProcessResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings
from utils.logger import logger

router = APIRouter()


def _decode_b64_image(b64_str: str) -> Optional[bytes]:
    if not b64_str:
        return None
    try:
        clean_b64 = re.sub(r'^data:image\/[a-zA-Z0-9+]+;base64,', '', str(b64_str).strip())
        return base64.b64decode(clean_b64)
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None


async def _extract_image_bytes(val: Any) -> Optional[bytes]:
    """
    Universally extracts raw image bytes from UploadFile, str (Base64 or binary string), or bytes.
    """
    if val is None:
        return None
    if hasattr(val, "read"):
        data = await val.read()
        return data if data else None
    if isinstance(val, bytes):
        return val
    if isinstance(val, str) and len(val) > 0:
        # 1. Try base64 decoding
        b64 = _decode_b64_image(val)
        if b64 and len(b64) > 10:
            return b64
        # 2. Try raw latin1 / utf-8 decoded binary strings
        for enc in ('latin1', 'utf-8'):
            try:
                b = val.encode(enc)
                if len(b) > 10 and (b.startswith(b'\xff\xd8') or b.startswith(b'\x89PNG') or b.startswith(b'RIFF')):
                    return b
            except Exception:
                pass
    return None


@router.post("/ekyc/card", response_model=CardProcessResponse, summary="Extract and Validate ID Card Data")
async def extract_card(
    request: Request,
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    front_bytes = None
    back_bytes = None
    max_size_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024

    content_type = request.headers.get("content-type", "").lower()

    # 1. Handle JSON request body with Base64 payload
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                f_val = (
                    body.get("front_image")
                    or body.get("frontImage")
                    or body.get("card_front")
                    or body.get("cardFront")
                    or body.get("front")
                    or body.get("image")
                    or body.get("file")
                )
                b_val = (
                    body.get("back_image")
                    or body.get("backImage")
                    or body.get("card_back")
                    or body.get("cardBack")
                    or body.get("back")
                )
                front_bytes = await _extract_image_bytes(f_val)
                back_bytes = await _extract_image_bytes(b_val)
        except Exception as json_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {str(json_err)}"
            )

    # 2. Handle Multipart Form-Data or URL-Encoded Form
    if front_bytes is None:
        try:
            form = await request.form()
            for key in ["front_image", "frontImage", "card_front", "cardFront", "front", "image_front", "file_front", "file", "image"]:
                if key in form:
                    front_bytes = await _extract_image_bytes(form[key])
                    if front_bytes:
                        break
            for key in ["back_image", "backImage", "card_back", "cardBack", "back", "image_back", "file_back"]:
                if key in form:
                    back_bytes = await _extract_image_bytes(form[key])
                    if back_bytes:
                        break
        except Exception:
            pass

    # 3. Fallback to raw request body if binary stream was sent directly
    if front_bytes is None:
        try:
            raw_body = await request.body()
            if raw_body and len(raw_body) > 30:
                if raw_body.startswith(b'\xff\xd8') or raw_body.startswith(b'\x89PNG') or raw_body.startswith(b'RIFF'):
                    front_bytes = raw_body
        except Exception:
            pass

    if not front_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required front card image. Please provide front_image (or frontImage) via multipart form-data or JSON base64."
        )

    if len(front_bytes) > max_size_bytes or (back_bytes and len(back_bytes) > max_size_bytes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Card image size exceeds maximum limit of {settings.MAX_IMAGE_SIZE_MB}MB."
        )

    response = orchestrator.process_card(
        front_bytes=front_bytes,
        back_bytes=back_bytes
    )
    return response
