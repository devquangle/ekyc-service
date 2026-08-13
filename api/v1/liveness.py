from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from api.dependencies import get_orchestrator
from schemas.liveness import LivenessResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings

router = APIRouter()


@router.post("/ekyc/face/liveness", response_model=LivenessResponse, summary="Video Liveness & Anti-Spoofing Detection")
async def detect_liveness(
    video_file: UploadFile = File(..., description="Recorded video file (MP4/WebM)"),
    expected_gestures: Optional[str] = Form(None, description="Comma-separated expected active gestures (e.g. BLINK,TURN_LEFT)"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    if not (video_file.content_type or "").startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid video format. Must upload MP4 or WebM video."
        )

    video_bytes = await video_file.read()

    max_size_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
    if len(video_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Video file size exceeds maximum limit."
        )

    gestures_list: Optional[List[str]] = None
    if expected_gestures:
        gestures_list = [g.strip().upper() for g in expected_gestures.split(",") if g.strip()]

    response = orchestrator.detect_liveness(video_bytes, gestures_list)
    return response
