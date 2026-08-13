import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from core.ocr_engine import OcrEngine
from core.qr_engine import QrEngine
from core.mrz_engine import MrzEngine
from ocr import (
    FieldExtractor,
    CardTypeClassifier,
    OCRText,
)
from schemas.card import ExtractedCardData, QualityChecks, FieldMetadata
from utils.image_utils import check_image_quality
from utils.text_utils import remove_vietnamese_accents
from utils.logger import logger


class CardProcessor:
    """
    Primary Data Extraction Orchestrator for Card Processing.
    Collects raw data across OCR, QR, and MRZ engines, performs image quality checks,
    classifies card type, and returns structured extracted data with MRZ/QR fallbacks for validation.
    """

    def __init__(self, ocr_engine: OcrEngine, qr_engine: QrEngine, mrz_engine: MrzEngine):
        self.ocr_engine = ocr_engine
        self.qr_engine = qr_engine
        self.mrz_engine = mrz_engine
        self.field_extractor = FieldExtractor()
        self.card_classifier = CardTypeClassifier()

    def _get_field_val_with_fallback(
        self,
        all_ocr_fields: Dict[str, Any],
        qr_data: Optional[Dict[str, Any]],
        mrz_data: Optional[Dict[str, Any]],
        field_name: str
    ) -> Optional[str]:
        """
        Fallback Priority for ExtractedCardData with sticky word normalization for fullName:
        1. OCR extracted value
        2. QR decoded value
        3. MRZ parsed value
        """
        ocr_ext = all_ocr_fields.get(field_name)
        ocr_val = ocr_ext.value if ocr_ext and ocr_ext.value else None
        qr_val = qr_data.get(field_name) if qr_data else None
        mrz_val = mrz_data.get(field_name) if mrz_data else None

        # Sticky name resolution for fullName
        if field_name == "fullName":
            better_name = qr_val or mrz_val
            if ocr_val and better_name:
                k_ocr = remove_vietnamese_accents(ocr_val).replace(" ", "").upper()
                k_better = remove_vietnamese_accents(better_name).replace(" ", "").upper()
                if k_ocr == k_better and ocr_val.count(" ") < better_name.count(" "):
                    logger.info(f"[CARD_PROCESSOR] Replacing sticky OCR name '{ocr_val}' with properly spaced name '{better_name}'")
                    return better_name

        if ocr_val:
            return ocr_val
        if qr_val:
            return qr_val
        if mrz_val:
            return mrz_val

        return None

    def process(
        self, front_image: np.ndarray, back_image: Optional[np.ndarray] = None
    ) -> Tuple[
        str,
        float,
        ExtractedCardData,
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        QualityChecks,
        List[FieldMetadata]
    ]:
        empty_data = ExtractedCardData()
        empty_quality = QualityChecks(isBlur=False, hasGlare=False, isCropped=False)

        if front_image is None or front_image.size == 0:
            return (
                "UNKNOWN",
                0.0,
                empty_data,
                None,
                None,
                empty_quality,
                []
            )

        try:
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

            # 7. Construct ExtractedCardData with Fallback Priority (OCR -> QR -> MRZ)
            extracted_data = ExtractedCardData(
                identityNumber=self._get_field_val_with_fallback(all_ocr_fields, qr_data, mrz_data, "identityNumber"),
                fullName=self._get_field_val_with_fallback(all_ocr_fields, qr_data, mrz_data, "fullName"),
                dateOfBirth=self._get_field_val_with_fallback(all_ocr_fields, qr_data, mrz_data, "dateOfBirth"),
                gender=self._get_field_val_with_fallback(all_ocr_fields, qr_data, mrz_data, "gender"),
                nationality=self._get_field_val_with_fallback(all_ocr_fields, qr_data, mrz_data, "nationality"),
                placeOfOrigin=self._get_field_val_with_fallback(all_ocr_fields, qr_data, mrz_data, "placeOfOrigin"),
                placeOfResidence=self._get_field_val_with_fallback(all_ocr_fields, qr_data, mrz_data, "placeOfResidence"),
                dateOfIssue=self._get_field_val_with_fallback(all_ocr_fields, qr_data, mrz_data, "dateOfIssue"),
                dateOfExpiry=self._get_field_val_with_fallback(all_ocr_fields, qr_data, mrz_data, "dateOfExpiry"),
            )

            field_metadata = [
                FieldMetadata(
                    field=fname,
                    value=fext.value,
                    source="OCR",
                    keyword=fext.keyword,
                    language=fext.language,
                    confidence=fext.confidence,
                    rawText=fext.rawText,
                    ocrValue=fext.value,
                    ocrKeyword=fext.keyword,
                    ocrLanguage=fext.language
                )
                for fname, fext in all_ocr_fields.items()
            ]

            return (
                card_type,
                card_type_confidence,
                extracted_data,
                qr_data,
                mrz_data,
                quality_checks,
                field_metadata
            )
        except Exception as e:
            logger.error(f"[CARD_PROCESSOR] Unexpected error during card processing: {str(e)}", exc_info=True)
            return (
                "UNKNOWN",
                0.0,
                empty_data,
                None,
                None,
                empty_quality,
                []
            )
