from typing import Optional
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from api.dependencies import get_orchestrator
from schemas.ekyc import FullEkycResponse
from services.ekyc_orchestrator import EkycOrchestrator

router = APIRouter()


@router.post("/ekyc/verify", response_model=FullEkycResponse, summary="Full Orchestrated eKYC Verification Pipeline")
async def process_full_ekyc(
    front_image: UploadFile = File(..., description="Front side image of ID card"),
    back_image: UploadFile = File(..., description="Back side image of ID card"),
    selfie_image: Optional[UploadFile] = File(None, description="Selfie photo for face verification"),
    video_file: Optional[UploadFile] = File(None, description="Video file for liveness verification"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    if not front_image.content_type.startswith("image/") or not back_image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Front and back images must be JPEG or PNG."
        )

    front_bytes = await front_image.read()
    back_bytes = await back_image.read()

    selfie_bytes = await selfie_image.read() if selfie_image else None
    video_bytes = await video_file.read() if video_file else None

    if not selfie_bytes and not video_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must provide either selfie_image or video_file for verification."
        )

    response = orchestrator.process_full_ekyc(
        front_bytes=front_bytes,
        back_bytes=back_bytes,
        selfie_bytes=selfie_bytes,
        video_bytes=video_bytes
    )
    return response
