from typing import Any
from fastapi import Request, HTTPException, status
from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from core.face_engine import FaceEngine
from core.liveness_engine import LivenessEngine
from processors.card_processor import CardProcessor
from processors.card_validator import CardValidator
from services.ekyc_orchestrator import EkycOrchestrator


def _get_state_attr(request: Request, attr_name: str, service_name: str) -> Any:
    instance = getattr(request.app.state, attr_name, None)
    if instance is None:
        # Lazy fallback initialization if lifespan was bypassed in testing
        if attr_name == "ocr_engine":
            instance = OcrEngine()
        elif attr_name == "qr_engine":
            instance = QrEngine()
        elif attr_name == "mrz_engine":
            instance = MrzEngine()
        elif attr_name == "face_engine":
            instance = FaceEngine()
        elif attr_name == "liveness_engine":
            instance = LivenessEngine()
        elif attr_name == "card_processor":
            instance = CardProcessor(
                get_ocr_engine(request),
                get_qr_engine(request),
                get_mrz_engine(request)
            )
        elif attr_name == "card_validator":
            instance = CardValidator()
        elif attr_name == "orchestrator":
            instance = EkycOrchestrator(
                get_card_processor(request),
                get_card_validator(request),
                get_face_engine(request),
                get_liveness_engine(request)
            )
        if instance is not None:
            setattr(request.app.state, attr_name, instance)
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"{service_name} is not initialized or unavailable."
            )
    return instance


def get_ocr_engine(request: Request) -> OcrEngine:
    return _get_state_attr(request, "ocr_engine", "OCR Engine")


def get_qr_engine(request: Request) -> QrEngine:
    return _get_state_attr(request, "qr_engine", "QR Engine")


def get_mrz_engine(request: Request) -> MrzEngine:
    return _get_state_attr(request, "mrz_engine", "MRZ Engine")


def get_face_engine(request: Request) -> FaceEngine:
    return _get_state_attr(request, "face_engine", "Face Engine")


def get_liveness_engine(request: Request) -> LivenessEngine:
    return _get_state_attr(request, "liveness_engine", "Liveness Engine")


def get_card_processor(request: Request) -> CardProcessor:
    return _get_state_attr(request, "card_processor", "Card Processor")


def get_card_validator(request: Request) -> CardValidator:
    return _get_state_attr(request, "card_validator", "Card Validator")


def get_orchestrator(request: Request) -> EkycOrchestrator:
    return _get_state_attr(request, "orchestrator", "eKYC Orchestrator")
