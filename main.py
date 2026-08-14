from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from api.v1.router import api_v1_router
from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from core.face_engine import FaceEngine
from core.liveness_engine import LivenessEngine
from processors.card_processor import CardProcessor
from processors.card_validator import CardValidator
from services.ekyc_orchestrator import EkycOrchestrator
from utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application Lifespan Manager: Pre-warms AI Models/Engines on startup
    and releases resources on shutdown.
    """
    if getattr(app.state, "orchestrator", None) is None:
        logger.info("Initializing Python eKYC Service AI Engines...")

        ocr_engine = getattr(app.state, "ocr_engine", None) or OcrEngine()
        qr_engine = getattr(app.state, "qr_engine", None) or QrEngine()
        mrz_engine = getattr(app.state, "mrz_engine", None) or MrzEngine()
        face_engine = getattr(app.state, "face_engine", None) or FaceEngine()
        liveness_engine = getattr(app.state, "liveness_engine", None) or LivenessEngine()

        card_processor = getattr(app.state, "card_processor", None) or CardProcessor(ocr_engine, qr_engine, mrz_engine)
        card_validator = getattr(app.state, "card_validator", None) or CardValidator()
        orchestrator = EkycOrchestrator(card_processor, card_validator, face_engine, liveness_engine)

        # Store in app.state for FastAPI Dependency Injection
        app.state.ocr_engine = ocr_engine
        app.state.qr_engine = qr_engine
        app.state.mrz_engine = mrz_engine
        app.state.face_engine = face_engine
        app.state.liveness_engine = liveness_engine
        app.state.card_processor = card_processor
        app.state.card_validator = card_validator
        app.state.orchestrator = orchestrator

        logger.info("All eKYC AI Engines initialized and pre-warmed successfully.")
    yield
    logger.info("Shutting down eKYC Service...")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Production-Ready Computer Vision & Biometrics Engine for Vietnamese eKYC",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 Router
app.include_router(api_v1_router, prefix="/api")


@app.get("/health", tags=["System Health"])
async def health_check():
    return {
        "status": "UP",
        "service": settings.APP_NAME,
        "device": settings.DEVICE
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
