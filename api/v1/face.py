import base64
import re
from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from api.dependencies import get_orchestrator
from schemas.face import FaceVerifyResponse
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
    if val is None:
        return None
    if hasattr(val, "read"):
        data = await val.read()
        return data if data else None
    if isinstance(val, bytes):
        return val
    if isinstance(val, str) and len(val) > 0:
        b64 = _decode_b64_image(val)
        if b64 and len(b64) > 10:
            return b64
        for enc in ('latin1', 'utf-8'):
            try:
                b = val.encode(enc)
                if len(b) > 10 and (b.startswith(b'\xff\xd8') or b.startswith(b'\x89PNG') or b.startswith(b'RIFF')):
                    return b
            except Exception:
                pass
    return None


@router.post("/ekyc/face/verify", response_model=FaceVerifyResponse, summary="Face Verification (Card Portrait vs Selfie)")
async def verify_face(
    request: Request,
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    card_bytes = None
    selfie_bytes = None
    max_size_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024

    content_type = request.headers.get("content-type", "").lower()

    # 1. JSON Base64 support
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                c_val = (
                    body.get("card_portrait")
                    or body.get("cardPortrait")
                    or body.get("card_image")
                    or body.get("cardImage")
                    or body.get("front_image")
                    or body.get("frontImage")
                    or body.get("card")
                )
                s_val = (
                    body.get("selfie_image")
                    or body.get("selfieImage")
                    or body.get("selfie")
                    or body.get("face_image")
                    or body.get("faceImage")
                    or body.get("face")
                )
                card_bytes = await _extract_image_bytes(c_val)
                selfie_bytes = await _extract_image_bytes(s_val)
        except Exception as json_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {str(json_err)}"
            )

    # 2. Multipart form upload
    if card_bytes is None or selfie_bytes is None:
        try:
            form = await request.form()
            if not card_bytes:
                for key in ["card_portrait", "cardPortrait", "card_image", "cardImage", "front_image", "frontImage", "card"]:
                    if key in form:
                        card_bytes = await _extract_image_bytes(form[key])
                        if card_bytes:
                            break
            if not selfie_bytes:
                for key in ["selfie_image", "selfieImage", "selfie", "face_image", "faceImage", "face"]:
                    if key in form:
                        selfie_bytes = await _extract_image_bytes(form[key])
                        if selfie_bytes:
                            break
        except Exception:
            pass

    if not card_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required card_portrait (or cardImage/front_image)."
        )

    if not selfie_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required selfie_image (or selfieImage/face)."
        )

    if len(card_bytes) > max_size_bytes or len(selfie_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image size exceeds maximum limit of {settings.MAX_IMAGE_SIZE_MB}MB."
        )

    response = orchestrator.verify_face(
        card_image_bytes=card_bytes,
        selfie_image_bytes=selfie_bytes
    )
    return response
