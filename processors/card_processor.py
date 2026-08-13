import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from ocr import (
    FieldExtractor,
    CardTypeClassifier,
    CrossValidator,
    OCRText,
    normalize_unicode,
)
from schemas.card import ExtractedCardData, QualityChecks, FieldMetadata, CrossValidationResult
from utils.image_utils import check_image_quality
from utils.logger import logger


class CardProcessor:
    def __init__(self, ocr_engine: OcrEngine, qr_engine: QrEngine, mrz_engine: MrzEngine):
        self.ocr_engine = ocr_engine
        self.qr_engine = qr_engine
        self.mrz_engine = mrz_engine
        self.field_extractor = FieldExtractor()
        self.card_classifier = CardTypeClassifier()
        self.cross_validator = CrossValidator()

    def process(
        self, front_image: np.ndarray, back_image: Optional[np.ndarray] = None
    ) -> Tuple[
        bool,
        str,
        float,
        ExtractedCardData,
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        CrossValidationResult,
        QualityChecks,
        List[FieldMetadata],
        List[str]
    ]:
        # 1. Quality Checks for both front and back images
        is_blur_f, has_glare_f, is_cropped_f = check_image_quality(front_image)
        if back_image is not None and back_image.size > 0:
            is_blur_b, has_glare_b, is_cropped_b = check_image_quality(back_image)
        else:
            is_blur_b, has_glare_b, is_cropped_b = False, False, False

        quality_checks = QualityChecks(
            isBlur=is_blur_f or is_blur_b,
            hasGlare=has_glare_f or has_glare_b,
            isCropped=is_cropped_f or is_cropped_b
        )

        # 2. OCR Token Detection
        front_tokens: List[OCRText] = self.ocr_engine.detect_tokens(front_image) if self.ocr_engine else []
        back_tokens: List[OCRText] = self.ocr_engine.detect_tokens(back_image) if (self.ocr_engine and back_image is not None and back_image.size > 0) else []

        logger.info(f"[OCR_FRONT_TOKENS] count={len(front_tokens)}")
        logger.info(f"[OCR_BACK_TOKENS] count={len(back_tokens)}")

        # 3. Spatial Field Extraction
        front_fields = self.field_extractor.extract_all_fields(front_tokens)
        back_fields = self.field_extractor.extract_all_fields(back_tokens)

        all_ocr_fields = {**back_fields, **front_fields}

        # 4. QR Parser
        qr_data = self.qr_engine.decode(front_image) if self.qr_engine else None
        if not qr_data and self.qr_engine and back_image is not None and back_image.size > 0:
            qr_data = self.qr_engine.decode(back_image)

        # 5. MRZ Parser
        back_text_lines = [t.text for t in back_tokens]
        mrz_data = self.mrz_engine.parse(back_text_lines) if (self.mrz_engine and back_text_lines) else None

        # 6. Card Type Classifier
        card_type, card_type_confidence = self.card_classifier.classify(
            front_tokens, back_tokens, all_ocr_fields
        )

        logger.info(f"[CARD_PROCESSOR] Classified Card Type: {card_type} (conf={card_type_confidence})")

        # 7. Merge Field Sources & Cross Validate
        card_verified, final_data, cross_val_result, field_metadata, errors = self.cross_validator.merge_and_validate(
            all_ocr_fields, qr_data, mrz_data, card_type
        )

        return (
            card_verified,
            card_type,
            card_type_confidence,
            final_data,
            qr_data,
            mrz_data,
            cross_val_result,
            quality_checks,
            field_metadata,
            errors
        )
