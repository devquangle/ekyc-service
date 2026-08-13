from fastapi import Request
from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from core.face_engine import FaceEngine
from core.liveness_engine import LivenessEngine
from processors.card_processor import CardProcessor
from processors.card_validator import CardValidator
from services.ekyc_orchestrator import EkycOrchestrator


def get_ocr_engine(request: Request) -> OcrEngine:
    return request.app.state.ocr_engine


def get_qr_engine(request: Request) -> QrEngine:
    return request.app.state.qr_engine


def get_mrz_engine(request: Request) -> MrzEngine:
    return request.app.state.mrz_engine


def get_face_engine(request: Request) -> FaceEngine:
    return request.app.state.face_engine


def get_liveness_engine(request: Request) -> LivenessEngine:
    return request.app.state.liveness_engine


def get_card_processor(request: Request) -> CardProcessor:
    return request.app.state.card_processor


def get_card_validator(request: Request) -> CardValidator:
    return request.app.state.card_validator


def get_orchestrator(request: Request) -> EkycOrchestrator:
    return request.app.state.orchestrator
