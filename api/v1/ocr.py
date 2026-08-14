import base64
import re
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
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


@router.post("/ekyc/card", response_model=CardProcessResponse, summary="Extract and Validate ID Card Data")
async def extract_card(
    request: Request,
    front_image: Optional[UploadFile] = File(None, description="Front side image of ID card (JPEG/PNG)"),
    back_image: Optional[UploadFile] = File(None, description="Back side image of ID card (JPEG/PNG)"),
    frontImage: Optional[UploadFile] = File(None, description="CamelCase alias for front_image"),
    backImage: Optional[UploadFile] = File(None, description="CamelCase alias for back_image"),
    card_front: Optional[UploadFile] = File(None, description="Alias for front_image"),
    card_back: Optional[UploadFile] = File(None, description="Alias for back_image"),
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
            f_b64 = (
                body.get("front_image")
                or body.get("frontImage")
                or body.get("card_front")
                or body.get("cardFront")
                or body.get("front")
                or body.get("image")
            )
            b_b64 = (
                body.get("back_image")
                or body.get("backImage")
                or body.get("card_back")
                or body.get("cardBack")
                or body.get("back")
            )
            if f_b64:
                front_bytes = _decode_b64_image(f_b64)
            if b_b64:
                back_bytes = _decode_b64_image(b_b64)
        except Exception as json_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {str(json_err)}"
            )

    # 2. Handle Multipart Form-Data upload
    if front_bytes is None:
        f_file = front_image or frontImage or card_front
        b_file = back_image or backImage or card_back

        # Check form dynamically if not captured in standard kwargs
        try:
            form = await request.form()
            if not f_file:
                for key in ["front_image", "frontImage", "card_front", "cardFront", "front", "image_front", "file_front", "file", "image"]:
                    if key in form:
                        val = form[key]
                        if hasattr(val, "read"):
                            f_file = val
                            break
                        elif isinstance(val, str) and len(val) > 20:
                            front_bytes = _decode_b64_image(val)
                            break
            if not b_file and not back_bytes:
                for key in ["back_image", "backImage", "card_back", "cardBack", "back", "image_back", "file_back"]:
                    if key in form:
                        val = form[key]
                        if hasattr(val, "read"):
                            b_file = val
                            break
                        elif isinstance(val, str) and len(val) > 20:
                            back_bytes = _decode_b64_image(val)
                            break
        except Exception:
            pass

        if f_file and hasattr(f_file, "read"):
            front_bytes = await f_file.read()
        if b_file and hasattr(b_file, "read"):
            back_bytes = await b_file.read()

    if not front_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required front card image. Please provide 'front_image' or 'frontImage'."
        )

    if len(front_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Front image size exceeds limit."
        )

    if back_bytes and len(back_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Back image size exceeds limit."
        )

    response = orchestrator.process_card(front_bytes, back_bytes or b"")
    return response
