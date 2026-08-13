from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from api.dependencies import get_orchestrator
from schemas.liveness import LivenessResponse
from services.ekyc_orchestrator import EkycOrchestrator

router = APIRouter()


@router.post("/ekyc/face/liveness", response_model=LivenessResponse, summary="Video Liveness & Anti-Spoofing Detection")
async def detect_liveness(
    video_file: UploadFile = File(..., description="Recorded video file (MP4/WebM)"),
    expected_gestures: Optional[str] = Form(None, description="Comma-separated expected active gestures (e.g. BLINK,TURN_LEFT)"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    video_bytes = await video_file.read()

    gestures_list: Optional[List[str]] = None
    if expected_gestures:
        gestures_list = [g.strip().upper() for g in expected_gestures.split(",") if g.strip()]

    response = orchestrator.detect_liveness(video_bytes, gestures_list)
    return response
