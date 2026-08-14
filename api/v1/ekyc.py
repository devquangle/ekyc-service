import base64
import re
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
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


@router.post("/ekyc/verify", response_model=FullEkycResponse, summary="Full Orchestrated eKYC Verification Pipeline")
async def process_full_ekyc(
    request: Request,
    front_image: Optional[UploadFile] = File(None, description="Front side image of ID card"),
    back_image: Optional[UploadFile] = File(None, description="Back side image of ID card"),
    frontImage: Optional[UploadFile] = File(None, description="CamelCase alias for front_image"),
    backImage: Optional[UploadFile] = File(None, description="CamelCase alias for back_image"),
    selfie_image: Optional[UploadFile] = File(None, description="Selfie photo for face verification"),
    selfieImage: Optional[UploadFile] = File(None, description="CamelCase alias for selfie_image"),
    video_file: Optional[UploadFile] = File(None, description="Video file for liveness verification"),
    videoFile: Optional[UploadFile] = File(None, description="CamelCase alias for video_file"),
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
            f_b64 = body.get("front_image") or body.get("frontImage") or body.get("card_front") or body.get("cardFront")
            b_b64 = body.get("back_image") or body.get("backImage") or body.get("card_back") or body.get("cardBack")
            s_b64 = body.get("selfie_image") or body.get("selfieImage") or body.get("selfie")
            v_b64 = body.get("video_file") or body.get("videoFile") or body.get("video")
            if f_b64:
                front_bytes = _decode_b64_image(f_b64)
            if b_b64:
                back_bytes = _decode_b64_image(b_b64)
            if s_b64:
                selfie_bytes = _decode_b64_image(s_b64)
            if v_b64:
                video_bytes = _decode_b64_image(v_b64)
        except Exception as json_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {str(json_err)}"
            )

    # 2. Multipart form upload
    if front_bytes is None:
        f_file = front_image or frontImage
        b_file = back_image or backImage
        s_file = selfie_image or selfieImage
        v_file = video_file or videoFile

        try:
            form = await request.form()
            if not f_file:
                for key in ["front_image", "frontImage", "card_front", "cardFront", "front"]:
                    if key in form and hasattr(form[key], "read"):
                        f_file = form[key]
                        break
            if not b_file:
                for key in ["back_image", "backImage", "card_back", "cardBack", "back"]:
                    if key in form and hasattr(form[key], "read"):
                        b_file = form[key]
                        break
            if not s_file:
                for key in ["selfie_image", "selfieImage", "selfie", "face"]:
                    if key in form and hasattr(form[key], "read"):
                        s_file = form[key]
                        break
            if not v_file:
                for key in ["video_file", "videoFile", "video"]:
                    if key in form and hasattr(form[key], "read"):
                        v_file = form[key]
                        break
        except Exception:
            pass

        if f_file and hasattr(f_file, "read"):
            front_bytes = await f_file.read()
        if b_file and hasattr(b_file, "read"):
            back_bytes = await b_file.read()
        if s_file and hasattr(s_file, "read"):
            selfie_bytes = await s_file.read()
        if v_file and hasattr(v_file, "read"):
            video_bytes = await v_file.read()

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
