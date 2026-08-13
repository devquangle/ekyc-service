from fastapi import APIRouter
from api.v1.ocr import router as ocr_router
from api.v1.face import router as face_router
from api.v1.liveness import router as liveness_router
from api.v1.ekyc import router as ekyc_router

api_v1_router = APIRouter(prefix="/v1")

api_v1_router.include_router(ocr_router, tags=["Card OCR & Processing"])
api_v1_router.include_router(face_router, tags=["Face Verification"])
api_v1_router.include_router(liveness_router, tags=["Video Liveness"])
api_v1_router.include_router(ekyc_router, tags=["Full eKYC Pipeline"])
