from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from api.dependencies import get_orchestrator
from schemas.ekyc import FullEkycResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings

router = APIRouter()


@router.post("/ekyc/verify", response_model=FullEkycResponse, summary="Full Orchestrated eKYC Verification Pipeline")
async def process_full_ekyc(
    front_image: UploadFile = File(..., description="Front side image of ID card"),
    back_image: UploadFile = File(..., description="Back side image of ID card"),
    selfie_image: Optional[UploadFile] = File(None, description="Selfie photo for face verification"),
    video_file: Optional[UploadFile] = File(None, description="Video file for liveness verification"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    if not (front_image.content_type or "").startswith("image/") or not (back_image.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Front and back images must be JPEG or PNG."
        )

    if not selfie_image and not video_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either selfie_image or video_file must be provided for eKYC verification."
        )

    max_image_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    max_video_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024

    front_bytes = await front_image.read()
    back_bytes = await back_image.read()

    if len(front_bytes) > max_image_bytes or len(back_bytes) > max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="ID Card image size exceeds maximum allowed limit."
        )

    selfie_bytes = None
    if selfie_image:
        if not (selfie_image.content_type or "").startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid selfie image format. Must be JPEG or PNG."
            )
        selfie_bytes = await selfie_image.read()
        if len(selfie_bytes) > max_image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selfie image size exceeds maximum allowed limit."
            )

    video_bytes = None
    if video_file:
        if not (video_file.content_type or "").startswith("video/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid video format. Must upload MP4 or WebM video."
            )
        video_bytes = await video_file.read()
        if len(video_bytes) > max_video_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video file size exceeds maximum limit."
            )

    response = orchestrator.process_full_ekyc(
        front_bytes=front_bytes,
        back_bytes=back_bytes,
        selfie_bytes=selfie_bytes,
        video_bytes=video_bytes
    )
    return response
