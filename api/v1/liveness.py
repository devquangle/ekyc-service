from typing import Optional, List, Union
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status, Request
from starlette.concurrency import run_in_threadpool
from api.dependencies import get_orchestrator
from schemas.liveness import LivenessResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings
from utils.media_parser import parse_media_payload

router = APIRouter()

LIVENESS_FIELD_ALIASES = {
    "video_file": [
        "video_file", "videoFile", "video", "file", "video_data"
    ],
    "expected_gestures": [
        "expected_gestures", "expectedGestures", "gestures", "actions"
    ]
}


@router.post("/ekyc/face/liveness", response_model=LivenessResponse, summary="Video Liveness & Anti-Spoofing Detection")
async def detect_liveness(
    request: Request,
    video_file: Optional[Union[UploadFile, str]] = File(None, description="Recorded video file (Upload MP4/WebM or Base64 string)"),
    expected_gestures: Optional[str] = Form(None, description="Comma-separated expected active gestures (e.g. BLINK,TURN_LEFT)"),
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
) -> LivenessResponse:
    """
    Asynchronously analyzes video stream for passive MiniFASNet anti-spoofing and active challenge gestures.
    Accepts both multipart/form-data (video upload/string) and application/json (Base64 video string).
    Executes OpenCV VideoCapture frame analysis in an asynchronous threadpool.
    """
    payload = await parse_media_payload(
        request=request,
        alias_map=LIVENESS_FIELD_ALIASES,
        max_size_mb=settings.MAX_VIDEO_SIZE_MB,
        required_fields=["video_file"],
        text_fields=["expected_gestures"]
    )

    video_bytes = payload.get("video_file")
    raw_gestures = payload.get("expected_gestures")

    gestures_list: Optional[List[str]] = None
    if raw_gestures:
        if isinstance(raw_gestures, list):
            gestures_list = [str(g).strip().upper() for g in raw_gestures if str(g).strip()]
        elif isinstance(raw_gestures, str):
            gestures_list = [g.strip().upper() for g in raw_gestures.split(",") if g.strip()]

    response = await run_in_threadpool(
        orchestrator.detect_liveness,
        video_bytes=video_bytes,
        expected_gestures=gestures_list
    )
    return response
