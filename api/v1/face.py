from typing import Optional, Union
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from starlette.concurrency import run_in_threadpool
from api.dependencies import get_orchestrator
from schemas.face import FaceVerifyResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings
from utils.media_parser import parse_media_payload

router = APIRouter()

FACE_FIELD_ALIASES = {
    "card_portrait": [
        "card_portrait", "cardPortrait", "card_image", "cardImage",
        "front_image", "frontImage", "card", "card_file"
    ],
    "selfie_image": [
        "selfie_image", "selfieImage", "selfie", "face_image",
        "faceImage", "face", "selfie_file"
    ]
}


@router.post("/ekyc/face/verify", response_model=FaceVerifyResponse, summary="Face Verification (Card Portrait vs Selfie)")
async def verify_face(
    request: Request,
    card_portrait: Optional[Union[UploadFile, str]] = File(None, description="Card image or cropped card portrait (Upload file or Base64 string)"),
    selfie_image: Optional[Union[UploadFile, str]] = File(None, description="Selfie photo of user (Upload file or Base64 string)"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
) -> FaceVerifyResponse:
    """
    Asynchronously verifies face 1-1 between ID card portrait and live selfie.
    Accepts both multipart/form-data (files/strings) and application/json (Base64 payloads).
    Executes InsightFace ArcFace deep learning extraction in an asynchronous threadpool.
    """
    payload = await parse_media_payload(
        request=request,
        alias_map=FACE_FIELD_ALIASES,
        max_size_mb=settings.MAX_IMAGE_SIZE_MB,
        required_fields=["card_portrait", "selfie_image"]
    )

    card_bytes = payload.get("card_portrait")
    selfie_bytes = payload.get("selfie_image")

    response = await run_in_threadpool(
        orchestrator.verify_face,
        card_image_bytes=card_bytes,
        selfie_image_bytes=selfie_bytes
    )
    return response
