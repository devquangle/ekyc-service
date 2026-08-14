import base64
import re
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
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


@router.post("/ekyc/face/verify", response_model=FaceVerifyResponse, summary="Face Verification (Card Portrait vs Selfie)")
async def verify_face(
    request: Request,
    card_portrait: Optional[UploadFile] = File(None, description="Front side card image or cropped card portrait"),
    selfie_image: Optional[UploadFile] = File(None, description="Selfie image of user"),
    cardPortrait: Optional[UploadFile] = File(None, description="CamelCase alias for card_portrait"),
    selfieImage: Optional[UploadFile] = File(None, description="CamelCase alias for selfie_image"),
    card_image: Optional[UploadFile] = File(None, description="Alias for card_portrait"),
    front_image: Optional[UploadFile] = File(None, description="Alias for card_portrait"),
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
            c_b64 = (
                body.get("card_portrait")
                or body.get("cardPortrait")
                or body.get("card_image")
                or body.get("cardImage")
                or body.get("front_image")
                or body.get("frontImage")
            )
            s_b64 = (
                body.get("selfie_image")
                or body.get("selfieImage")
                or body.get("selfie")
                or body.get("face_image")
                or body.get("faceImage")
            )
            if c_b64:
                card_bytes = _decode_b64_image(c_b64)
            if s_b64:
                selfie_bytes = _decode_b64_image(s_b64)
        except Exception as json_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {str(json_err)}"
            )

    # 2. Multipart form upload
    if card_bytes is None or selfie_bytes is None:
        c_file = card_portrait or cardPortrait or card_image or front_image
        s_file = selfie_image or selfieImage

        try:
            form = await request.form()
            if not c_file and not card_bytes:
                for key in ["card_portrait", "cardPortrait", "card_image", "cardImage", "front_image", "frontImage", "card"]:
                    if key in form:
                        val = form[key]
                        if hasattr(val, "read"):
                            c_file = val
                            break
                        elif isinstance(val, str) and len(val) > 20:
                            card_bytes = _decode_b64_image(val)
                            break
            if not s_file and not selfie_bytes:
                for key in ["selfie_image", "selfieImage", "selfie", "face_image", "faceImage", "face"]:
                    if key in form:
                        val = form[key]
                        if hasattr(val, "read"):
                            s_file = val
                            break
                        elif isinstance(val, str) and len(val) > 20:
                            selfie_bytes = _decode_b64_image(val)
                            break
        except Exception:
            pass

        if c_file and hasattr(c_file, "read"):
            card_bytes = await c_file.read()
        if s_file and hasattr(s_file, "read"):
            selfie_bytes = await s_file.read()

    if not card_bytes or not selfie_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required images. Please provide both 'card_portrait' (or 'cardPortrait') and 'selfie_image' (or 'selfieImage')."
        )

    if len(card_bytes) > max_size_bytes or len(selfie_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image size exceeds maximum allowed limit."
        )

    response = orchestrator.verify_face(card_bytes, selfie_bytes)
    return response
