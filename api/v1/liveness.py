import base64
import re
from typing import Optional, List, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from api.dependencies import get_orchestrator
from schemas.liveness import LivenessResponse
from services.ekyc_orchestrator import EkycOrchestrator
from config import settings
from utils.logger import logger

router = APIRouter()


def _decode_b64_video(b64_str: str) -> Optional[bytes]:
    if not b64_str:
        return None
    try:
        clean_b64 = re.sub(r'^data:video\/[a-zA-Z0-9+]+;base64,', '', str(b64_str).strip())
        return base64.b64decode(clean_b64)
    except Exception as e:
        logger.warning(f"Failed to decode base64 video: {e}")
        return None


async def _extract_video_bytes(val: Any) -> Optional[bytes]:
    if val is None:
        return None
    if hasattr(val, "read"):
        data = await val.read()
        return data if data else None
    if isinstance(val, bytes):
        return val
    if isinstance(val, str) and len(val) > 0:
        b64 = _decode_b64_video(val)
        if b64 and len(b64) > 10:
            return b64
        for enc in ('latin1', 'utf-8'):
            try:
                b = val.encode(enc)
                if len(b) > 50:
                    return b
            except Exception:
                pass
    return None


@router.post("/ekyc/face/liveness", response_model=LivenessResponse, summary="Video Liveness & Anti-Spoofing Detection")
async def detect_liveness(
    request: Request,
    orchestrator: EkycOrchestrator = Depends(get_orchestrator)
):
    video_bytes = None
    expected_gestures = None
    max_size_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024

    content_type = request.headers.get("content-type", "").lower()

    # 1. JSON Base64 support
    if "application/json" in content_type:
        try:
            body = await request.json()
            if isinstance(body, dict):
                v_val = body.get("video_file") or body.get("videoFile") or body.get("video") or body.get("file")
                video_bytes = await _extract_video_bytes(v_val)
                expected_gestures = body.get("expected_gestures") or body.get("expectedGestures") or body.get("gestures")
        except Exception as json_err:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON payload: {str(json_err)}"
            )

    # 2. Multipart form upload
    if video_bytes is None:
        try:
            form = await request.form()
            for key in ["video_file", "videoFile", "video", "file"]:
                if key in form:
                    video_bytes = await _extract_video_bytes(form[key])
                    if video_bytes:
                        break
            if not expected_gestures:
                for key in ["expected_gestures", "expectedGestures", "gestures"]:
                    if key in form and isinstance(form[key], str):
                        expected_gestures = form[key]
                        break
        except Exception:
            pass

    if not video_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required video_file (or videoFile)."
        )

    if len(video_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video file size exceeds maximum limit of {settings.MAX_VIDEO_SIZE_MB}MB."
        )

    gestures_list: Optional[List[str]] = None
    if expected_gestures:
        if isinstance(expected_gestures, list):
            gestures_list = [str(g).strip().upper() for g in expected_gestures if str(g).strip()]
        elif isinstance(expected_gestures, str):
            gestures_list = [g.strip().upper() for g in expected_gestures.split(",") if g.strip()]

    response = orchestrator.detect_liveness(video_bytes, gestures_list)
    return response
