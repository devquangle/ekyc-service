from typing import Optional, Union
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Request
from starlette.concurrency import run_in_threadpool
from api.dependencies import get_orchestrator
from schemas.card import CardProcessResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings
from utils.media_parser import parse_media_payload

router = APIRouter()

CARD_FIELD_ALIASES = {
    "front_image": [
        "front_image", "frontImage", "card_front", "cardFront",
        "front", "image_front", "file_front", "image", "file"
    ],
    "back_image": [
        "back_image", "backImage", "card_back", "cardBack",
        "back", "image_back", "file_back"
    ]
}


@router.post("/ekyc/card", response_model=CardProcessResponse, summary="Extract and Validate ID Card Data (OCR, QR, MRZ)")
async def extract_card(
    request: Request,
    front_image: Optional[Union[UploadFile, str]] = File(None, description="Front side ID card image (Upload file or Base64 string)"),
    back_image: Optional[Union[UploadFile, str]] = File(None, description="Back side ID card image (Upload file or Base64 string)"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
) -> CardProcessResponse:
    """
    Asynchronously extracts, cross-validates and restores diacritics for Vietnamese ID Cards.
    Accepts both multipart/form-data (files/strings) and application/json (Base64 payloads).
    Executes heavy AI inference in an asynchronous threadpool to keep the main event loop non-blocking.
    """
    payload = await parse_media_payload(
        request=request,
        alias_map=CARD_FIELD_ALIASES,
        max_size_mb=settings.MAX_IMAGE_SIZE_MB,
        required_fields=["front_image"]
    )

    front_bytes = payload.get("front_image")
    back_bytes = payload.get("back_image")

    response = await run_in_threadpool(
        orchestrator.process_card,
        front_bytes=front_bytes,
        back_bytes=back_bytes
    )
    return response
