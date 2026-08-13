from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from api.dependencies import get_orchestrator
from schemas.face import FaceVerifyResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings

router = APIRouter()


@router.post("/ekyc/face/verify", response_model=FaceVerifyResponse, summary="Face Verification (Card Portrait vs Selfie)")
async def verify_face(
    card_portrait: UploadFile = File(..., description="Front side card image or cropped card portrait"),
    selfie_image: UploadFile = File(..., description="Selfie image of user"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    if not (card_portrait.content_type or "").startswith("image/") or not (selfie_image.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Must upload JPEG or PNG images."
        )

    card_bytes = await card_portrait.read()
    selfie_bytes = await selfie_image.read()

    max_size_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(card_bytes) > max_size_bytes or len(selfie_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image size exceeds maximum allowed limit."
        )

    response = orchestrator.verify_face(card_bytes, selfie_bytes)
    return response
