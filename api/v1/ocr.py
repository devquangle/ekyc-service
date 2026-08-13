from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from api.dependencies import get_orchestrator
from schemas.card import CardProcessResponse
from services.ekyc_orchestrator import EkycOrchestrator

router = APIRouter()


@router.post("/ekyc/card", response_model=CardProcessResponse, summary="Extract and Validate ID Card Data")
async def extract_card(
    front_image: UploadFile = File(..., description="Front side image of ID card (JPEG/PNG)"),
    back_image: UploadFile = File(..., description="Back side image of ID card (JPEG/PNG)"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    if not front_image.content_type.startswith("image/") or not back_image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Must upload JPEG or PNG images."
        )

    front_bytes = await front_image.read()
    back_bytes = await back_image.read()

    response = orchestrator.process_card(front_bytes, back_bytes)
    return response
