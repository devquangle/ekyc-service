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
from schemas.card import ExtractedCardData, QualityChecks, FieldMetadata, VisualRegions
from core.face_verification.card_face_extractor import CardFaceExtractor
from utils.card_aligner import CardAligner
from utils.image_utils import check_image_quality
from utils.text_utils import remove_vietnamese_accents
from utils.text_normalizer import VietnameseTextCorrector
from utils.vietnamese_administrative_restorer import VietnameseAdministrativeRestorer
from utils.logger import logger
from config import settings


class CardProcessor:
    """
    Primary Data Extraction Orchestrator for Card Processing.
    Collects raw data across OCR, QR, and MRZ engines, performs image quality checks,
    classifies card type, and returns structured extracted data with MRZ/QR fallbacks for validation.
    Extracts distinct label_box, value_box, and visual regions (portrait, qrCode, mrzBlock).
    """

    def __init__(self, ocr_engine: OcrEngine, qr_engine: QrEngine, mrz_engine: MrzEngine):
        self.ocr_engine = ocr_engine
        self.qr_engine = qr_engine
        self.mrz_engine = mrz_engine
        self.field_extractor = FieldExtractor()
        self.card_classifier = CardTypeClassifier()
        self.card_face_extractor = CardFaceExtractor()
        # Card Alignment: YOLOv8-seg ONNX with OpenCV contour fallback
        self.card_aligner = CardAligner(model_path=settings.CARD_ALIGNER_MODEL_PATH) if settings.CARD_ALIGNER_ENABLED else None

    def _get_field_val_with_fallback(
        self,
        all_ocr_fields: Dict[str, Any],
        qr_data: Optional[Dict[str, Any]],
        mrz_data: Optional[Dict[str, Any]],
        field_name: str
    ) -> Optional[str]:
        """
        Resolves field value using prioritized sources:
        1. OCR extracted field
        2. QR Code fallback
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

        # Address fields: ensure full diacritic restoration & cross-validation with QR
        if field_name in ("placeOfResidence", "placeOfOrigin"):
            if qr_val and not ocr_val:
                return VietnameseAdministrativeRestorer.restore_address_diacritics(qr_val) or qr_val
            if ocr_val:
                restored_ocr = VietnameseAdministrativeRestorer.restore_address_diacritics(ocr_val) or ocr_val
                if qr_val:
                    restored_qr = VietnameseAdministrativeRestorer.restore_address_diacritics(qr_val) or qr_val
                    # If QR has more complete accents or content, prefer QR
                    if len(restored_qr) > len(restored_ocr):
                        return restored_qr
                return restored_ocr

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
        List[FieldMetadata],
        VisualRegions
    ]:
        empty_data = ExtractedCardData()
        empty_quality = QualityChecks(isBlur=False, hasGlare=False, isCropped=False)
        empty_visual = VisualRegions()

        if front_image is None or front_image.size == 0:
            return (
                "UNKNOWN",
                0.0,
                empty_data,
                None,
                None,
                empty_quality,
                [],
                empty_visual
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

            # 2. Card Alignment — straighten tilted/skewed card before OCR
            ocr_front = front_image
            ocr_back = back_image
            if self.card_aligner:
                aligned_front, front_was_aligned = self.card_aligner.align(front_image)
                if front_was_aligned:
                    ocr_front = aligned_front
                if back_image is not None and back_image.size > 0:
                    aligned_back, back_was_aligned = self.card_aligner.align(back_image)
                    if back_was_aligned:
                        ocr_back = aligned_back

            # 3. Portrait Face Extraction & Bounding Box
            portrait_box: Optional[List[float]] = None
            try:
                _, _, bbox_info, _ = self.card_face_extractor.extract_face(ocr_front)
                if bbox_info and bbox_info.detected:
                    portrait_box = [
                        round(float(bbox_info.x), 1),
                        round(float(bbox_info.y), 1),
                        round(float(bbox_info.x + bbox_info.w), 1),
                        round(float(bbox_info.y + bbox_info.h), 1)
                    ]
            except Exception as fe_err:
                logger.debug(f"[CARD_PROCESSOR] Portrait face detection skipped: {fe_err}")

            # 4. OCR Token Detection (on aligned images)
            front_tokens: List[OCRText] = self.ocr_engine.detect_tokens(ocr_front) if self.ocr_engine else []
            back_tokens: List[OCRText] = self.ocr_engine.detect_tokens(ocr_back) if (self.ocr_engine and ocr_back is not None and ocr_back.size > 0) else []

            logger.info(f"[OCR_FRONT_TOKENS] count={len(front_tokens)}")
            logger.info(f"[OCR_BACK_TOKENS] count={len(back_tokens)}")

            # 5. Spatial Field Extraction
            front_fields = self.field_extractor.extract_all_fields(front_tokens)
            back_fields = self.field_extractor.extract_all_fields(back_tokens)

            # Smart merge: front_fields takes priority; fallback to back_fields if front value is None
            all_ocr_fields: Dict[str, Any] = {**back_fields}
            for field_name, front_ext in front_fields.items():
                if front_ext and front_ext.value:
                    all_ocr_fields[field_name] = front_ext
                elif field_name not in all_ocr_fields or not (all_ocr_fields[field_name] and all_ocr_fields[field_name].value):
                    all_ocr_fields[field_name] = front_ext

            # Preserve dateOfIssue from back side if not found on front
            if "dateOfIssue" in back_fields and back_fields["dateOfIssue"] and back_fields["dateOfIssue"].value:
                if "dateOfIssue" not in front_fields or not front_fields["dateOfIssue"] or not front_fields["dateOfIssue"].value:
                    all_ocr_fields["dateOfIssue"] = back_fields["dateOfIssue"]

            logger.info(f"[CARD_PROCESSOR] Merged OCR fields: front={list(front_fields.keys())} back={list(back_fields.keys())} merged={list(all_ocr_fields.keys())}")

            # 6. QR Parser (Scan front_image first, fallback to back_image)
            qr_data = self.qr_engine.decode(front_image) if self.qr_engine else None
            qr_box = getattr(self.qr_engine, 'last_qr_bbox', None)

            if not qr_data and self.qr_engine and back_image is not None and back_image.size > 0:
                logger.info("[CARD_PROCESSOR] Front image QR null or unreadable. Executing back image QR scan fallback...")
                qr_data = self.qr_engine.decode(back_image)
                if not qr_box:
                    qr_box = getattr(self.qr_engine, 'last_qr_bbox', None)

            # 7. MRZ Parser & Bounding Box
            back_text_lines = [t.text for t in back_tokens]
            mrz_data = self.mrz_engine.parse(back_text_lines) if (self.mrz_engine and back_text_lines) else None

            mrz_box: Optional[List[float]] = None
            if back_tokens:
                back_h = max([pt[1] for t in back_tokens if t.bbox for pt in t.bbox] + [1000.0])
                mrz_tokens = []
                for t in back_tokens:
                    t_txt = t.text.strip()
                    if ("<" in t_txt and len(t_txt) >= 15) or len(t_txt) >= 28 or (t.center_y > 0.60 * back_h and len(t_txt) >= 15):
                        mrz_tokens.append(t)
                if mrz_tokens:
                    mrz_box = self.field_extractor._compute_bbox_4(mrz_tokens)

            visual_regions = VisualRegions(
                portrait=portrait_box,
                qrCode=qr_box,
                mrzBlock=mrz_box
            )

            # 8. Card Type Classifier
            card_type, card_type_confidence = self.card_classifier.classify(
                front_tokens, back_tokens, all_ocr_fields
            )

            logger.info(f"[CARD_PROCESSOR] Classified Card Type: {card_type} (conf={card_type_confidence})")

            # 9. Construct ExtractedCardData with Fallback Priority (OCR -> QR -> MRZ)
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

            field_metadata = []
            for fname, fext in all_ocr_fields.items():
                raw_t = fext.rawText if fext.rawText is not None else fext.value
                raw_txt, ocr_norm, corr_val, corr_conf = VietnameseTextCorrector.normalize_pipeline(raw_t)
                field_metadata.append(
                    FieldMetadata(
                        field=fname,
                        value=fext.value or ocr_norm,
                        label=fext.keyword,
                        label_box=fext.label_box,
                        value_box=fext.value_box,
                        labelBox=fext.label_box,
                        valueBox=fext.value_box,
                        source="OCR",
                        keyword=fext.keyword,
                        language=fext.language,
                        confidence=fext.confidence,
                        rawText=fext.rawText or raw_txt,
                        ocrValue=fext.value or ocr_norm,
                        ocrKeyword=fext.keyword,
                        ocrLanguage=fext.language,
                        correctedValue=corr_val,
                        correctionConfidence=corr_conf
                    )
                )

            return (
                card_type,
                card_type_confidence,
                extracted_data,
                qr_data,
                mrz_data,
                quality_checks,
                field_metadata,
                visual_regions
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
                [],
                empty_visual
            )
