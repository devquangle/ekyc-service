import base64
import re
from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from api.dependencies import get_orchestrator
from schemas.ekyc import FullEkycResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings
from utils.logger import logger

router = APIRouter()


def _decode_b64_image(b64_str: str) -> Optional[bytes]:
    if not b64_str:
        return None
    try:
        clean_b64 = re.sub(r'^data:(?:image|video)\/[a-zA-Z0-9+]+;base64,', '', str(b64_str).strip())
        return base64.b64decode(clean_b64)
    except Exception as e:
        logger.warning(f"Failed to decode base64 data: {e}")
        return None


async def _extract_bytes(val: Any) -> Optional[bytes]:
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
                if len(b) > 20:
                    return b
            except Exception:
                pass
    return None


@router.post("/ekyc/verify", response_model=FullEkycResponse, summary="Full Orchestrated eKYC Verification Pipeline")
async def process_full_ekyc(
    request: Request,
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    front_bytes = None
    back_bytes = None
    selfie_bytes = None
    video_bytes = None

    max_image_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    max_video_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024

    content_type = request.headers.get("content-type", "").lower()

    # 1. JSON Base64 support
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                f_val = body.get("front_image") or body.get("frontImage") or body.get("card_front") or body.get("cardFront") or body.get("front") or body.get("image")
                b_val = body.get("back_image") or body.get("backImage") or body.get("card_back") or body.get("cardBack") or body.get("back")
                s_val = body.get("selfie_image") or body.get("selfieImage") or body.get("selfie") or body.get("face_image") or body.get("face")
                v_val = body.get("video_file") or body.get("videoFile") or body.get("video")
                front_bytes = await _extract_bytes(f_val)
                back_bytes = await _extract_bytes(b_val)
                selfie_bytes = await _extract_bytes(s_val)
                video_bytes = await _extract_bytes(v_val)
        except Exception as json_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {str(json_err)}"
            )

    # 2. Multipart form upload
    if front_bytes is None:
        try:
            form = await request.form()
            for key in ["front_image", "frontImage", "card_front", "cardFront", "front", "image_front", "file_front", "image", "file"]:
                if key in form:
                    front_bytes = await _extract_bytes(form[key])
                    if front_bytes:
                        break
            for key in ["back_image", "backImage", "card_back", "cardBack", "back", "image_back", "file_back"]:
                if key in form:
                    back_bytes = await _extract_bytes(form[key])
                    if back_bytes:
                        break
            for key in ["selfie_image", "selfieImage", "selfie", "face_image", "faceImage", "face"]:
                if key in form:
                    selfie_bytes = await _extract_bytes(form[key])
                    if selfie_bytes:
                        break
            for key in ["video_file", "videoFile", "video"]:
                if key in form:
                    video_bytes = await _extract_bytes(form[key])
                    if video_bytes:
                        break
        except Exception:
            pass

    if not front_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required front card image."
        )

    if not selfie_bytes and not video_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either selfie_image (or selfieImage) or video_file (or videoFile) must be provided for eKYC verification."
        )

    if len(front_bytes) > max_image_bytes or (back_bytes and len(back_bytes) > max_image_bytes):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID Card image size exceeds maximum allowed limit."
        )

    if selfie_bytes and len(selfie_bytes) > max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selfie image size exceeds maximum allowed limit."
        )

    if video_bytes and len(video_bytes) > max_video_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video size exceeds maximum allowed limit."
        )

    response = orchestrator.process_full_ekyc(
        front_bytes=front_bytes,
        back_bytes=back_bytes or b"",
        selfie_bytes=selfie_bytes,
        video_bytes=video_bytes
    )
    return response
