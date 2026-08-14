from typing import Optional, Union
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from starlette.concurrency import run_in_threadpool
from api.dependencies import get_orchestrator
from schemas.ekyc import FullEkycResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings
from utils.media_parser import parse_media_payload

router = APIRouter()

EKYC_FIELD_ALIASES = {
    "front_image": [
        "front_image", "frontImage", "card_front", "cardFront",
        "front", "image_front", "file_front", "image", "file"
    ],
    "back_image": [
        "back_image", "backImage", "card_back", "cardBack",
        "back", "image_back", "file_back"
    ],
    "selfie_image": [
        "selfie_image", "selfieImage", "selfie", "face_image",
        "faceImage", "face", "selfie_file"
    ],
    "video_file": [
        "video_file", "videoFile", "video", "video_data"
    ]
}


@router.post("/ekyc/verify", response_model=FullEkycResponse, summary="Full Orchestrated eKYC Verification Pipeline")
@router.post("/ekyc/full", response_model=FullEkycResponse, summary="Alias for Full Orchestrated eKYC Verification Pipeline")
async def process_full_ekyc(

    request: Request,
    front_image: Optional[Union[UploadFile, str]] = File(None, description="Front side ID card image (Upload file or Base64 string)"),
    back_image: Optional[Union[UploadFile, str]] = File(None, description="Back side ID card image (Upload file or Base64 string)"),
    selfie_image: Optional[Union[UploadFile, str]] = File(None, description="Selfie photo for face verification (Upload file or Base64 string)"),
    video_file: Optional[Union[UploadFile, str]] = File(None, description="Video file for liveness verification (Upload file or Base64 string)"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
) -> FullEkycResponse:
    """
    Executes the end-to-end full eKYC verification pipeline:
    Card OCR & QR/MRZ Validation -> Face Crop -> Video Liveness / Anti-Spoofing -> 1-1 Face Matching.
    Accepts both multipart/form-data and application/json (Base64 payloads).
    Executes full pipeline in an asynchronous threadpool to ensure non-blocking server performance.
    """
    payload = await parse_media_payload(
        request=request,
        alias_map=EKYC_FIELD_ALIASES,
        max_size_mb=settings.MAX_VIDEO_SIZE_MB,  # allows large video size for full pipeline
        required_fields=["front_image"]
    )

    front_bytes = payload.get("front_image")
    back_bytes = payload.get("back_image") or b""
    selfie_bytes = payload.get("selfie_image")
    video_bytes = payload.get("video_file")

    if not selfie_bytes and not video_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either selfie_image or video_file must be provided for eKYC verification."
        )

    # Validate image size bounds specifically
    max_image_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if front_bytes and len(front_bytes) > max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Front card image size exceeds limit of {settings.MAX_IMAGE_SIZE_MB}MB."
        )
    if back_bytes and len(back_bytes) > max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Back card image size exceeds limit of {settings.MAX_IMAGE_SIZE_MB}MB."
        )
    if selfie_bytes and len(selfie_bytes) > max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Selfie image size exceeds limit of {settings.MAX_IMAGE_SIZE_MB}MB."
        )

    response = await run_in_threadpool(
        orchestrator.process_full_ekyc,
        front_bytes=front_bytes,
        back_bytes=back_bytes,
        selfie_bytes=selfie_bytes,
        video_bytes=video_bytes
    )
    return response
