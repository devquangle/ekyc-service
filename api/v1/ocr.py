from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from api.dependencies import get_orchestrator
from schemas.card import CardProcessResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings

router = APIRouter()


@router.post("/ekyc/card", response_model=CardProcessResponse, summary="Extract and Validate ID Card Data")
async def extract_card(
    front_image: UploadFile = File(..., description="Front side image of ID card (JPEG/PNG)"),
    back_image: UploadFile = File(..., description="Back side image of ID card (JPEG/PNG)"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    if not (front_image.content_type or "").startswith("image/") or not (back_image.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Front and back images must be JPEG or PNG."
        )

    front_bytes = await front_image.read()
    back_bytes = await back_image.read()

    max_size_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024

    if len(front_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Front image size exceeds limit."
        )

    if len(back_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Back image size exceeds limit."
        )

    response = orchestrator.process_card(front_bytes, back_bytes)
    return response
