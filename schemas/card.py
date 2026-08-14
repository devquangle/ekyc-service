from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, AliasChoices, field_validator, model_validator
from schemas.enums import CardType, CrossValidationStatus


def _validate_box_coords(box: Optional[List[Union[int, float]]]) -> Optional[List[float]]:
    if box is None:
        return None
    if len(box) != 4:
        raise ValueError(f"Bounding box must contain exactly 4 coordinates [x1, y1, x2, y2], got {len(box)}")
    x1, y1, x2, y2 = [float(v) for v in box]
    if x1 > x2 or y1 > y2:
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


class ExtractedCardData(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    identityNumber: Optional[str] = Field(default=None, description="12-digit Citizen Identity Number")
    fullName: Optional[str] = Field(default=None, description="Full Name of cardholder")
    dateOfBirth: Optional[str] = Field(default=None, description="Date of Birth YYYY-MM-DD")
    gender: Optional[str] = Field(default=None, description="Gender: Nam / Nữ")
    nationality: Optional[str] = Field(default=None, description="Nationality: Việt Nam")
    placeOfOrigin: Optional[str] = Field(default=None, description="Place of Origin / Nơi đăng ký khai sinh")
    placeOfResidence: Optional[str] = Field(default=None, description="Place of Residence / Nơi cư trú")
    dateOfIssue: Optional[str] = Field(default=None, description="Date of Issue YYYY-MM-DD")
    dateOfExpiry: Optional[str] = Field(default=None, description="Date of Expiry YYYY-MM-DD")


class VisualRegions(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    portrait: Optional[List[float]] = Field(default=None, description="[x_min, y_min, x_max, y_max] of portrait face")
    qrCode: Optional[List[float]] = Field(default=None, description="[x_min, y_min, x_max, y_max] of QR code")
    mrzBlock: Optional[List[float]] = Field(default=None, description="[x_min, y_min, x_max, y_max] of MRZ block")

    @field_validator("portrait", "qrCode", "mrzBlock", mode="before")
    @classmethod
    def validate_region_box(cls, v: Any) -> Optional[List[float]]:
        return _validate_box_coords(v)


class FieldMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    field: str = Field(description="Name of extracted field")
    value: Optional[str] = Field(default=None, description="Final selected field value")
    label: Optional[str] = Field(default=None, description="Clean label text")
    label_box: Optional[List[float]] = Field(
        default=None,
        validation_alias=AliasChoices("label_box", "labelBox"),
        description="Bounding box of label [x_min, y_min, x_max, y_max]"
    )
    value_box: Optional[List[float]] = Field(
        default=None,
        validation_alias=AliasChoices("value_box", "valueBox"),
        description="Bounding box of value [x_min, y_min, x_max, y_max]"
    )
    labelBox: Optional[List[float]] = Field(
        default=None,
        validation_alias=AliasChoices("labelBox", "label_box"),
        description="CamelCase alias for label_box"
    )
    valueBox: Optional[List[float]] = Field(
        default=None,
        validation_alias=AliasChoices("valueBox", "value_box"),
        description="CamelCase alias for value_box"
    )
    source: Optional[str] = Field(default=None, description="Selected data source: OCR, MRZ, or QR")
    keyword: Optional[str] = Field(default=None, description="Matched label keyword for selected source")
    language: Optional[str] = Field(default=None, description="Language of matched keyword (EN/VI)")
    confidence: float = Field(default=1.0, description="Confidence score 0.0 - 1.0")
    rawText: Optional[str] = Field(default=None, description="Exact uncorrected raw text from OCR/MRZ")
    ocrValue: Optional[str] = Field(default=None, description="Value extracted from OCR source")
    mrzValue: Optional[str] = Field(default=None, description="Value extracted from MRZ source")
    qrValue: Optional[str] = Field(default=None, description="Value extracted from QR source")
    ocrKeyword: Optional[str] = Field(default=None, description="Matched OCR keyword if detected")
    ocrLanguage: Optional[str] = Field(default=None, description="Language of matched OCR keyword")
    correctedValue: Optional[str] = Field(default=None, description="Corrected value if confidence threshold met")
    correctionConfidence: float = Field(default=0.0, description="Confidence of the correction (0.0 if not corrected)")

    @field_validator("label_box", "value_box", "labelBox", "valueBox", mode="before")
    @classmethod
    def validate_metadata_boxes(cls, v: Any) -> Optional[List[float]]:
        return _validate_box_coords(v)

    @model_validator(mode="before")
    @classmethod
    def sync_box_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            l_box = data.get("label_box") or data.get("labelBox")
            v_box = data.get("value_box") or data.get("valueBox")
            if l_box is not None:
                data["label_box"] = l_box
                data["labelBox"] = l_box
            if v_box is not None:
                data["value_box"] = v_box
                data["valueBox"] = v_box
            kw = data.get("keyword")
            lbl = data.get("label")
            if kw and not lbl:
                data["label"] = kw
        return data


class CrossValidationDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    fieldName: str
    ocrValue: Optional[str] = None
    qrValue: Optional[str] = None
    mrzValue: Optional[str] = None
    status: Union[CrossValidationStatus, str] = CrossValidationStatus.NOT_AVAILABLE


FieldValidationDetail = CrossValidationDetail


class CrossValidationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ocrMatchQr: Optional[bool] = None
    ocrMatchMrz: Optional[bool] = None
    mrzCheckDigitValid: Optional[bool] = None
    isExpired: Optional[bool] = None
    details: List[CrossValidationDetail] = Field(default_factory=list)


class QualityChecks(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    isBlur: bool = False
    hasGlare: bool = False
    isCropped: bool = False


class CardProcessResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    cardVerified: bool = True
    cardType: Union[CardType, str] = Field(default=CardType.CCCD_OLD, description="Detected Card Type: CCCD_OLD, CCCD_NEW, UNKNOWN")
    cardTypeConfidence: float = Field(default=1.0, description="Card type detection confidence 0.0 - 1.0")
    extractedData: ExtractedCardData = Field(default_factory=ExtractedCardData)
    crossValidation: CrossValidationResult = Field(default_factory=CrossValidationResult)
    qualityChecks: QualityChecks = Field(default_factory=QualityChecks)
    visualRegions: Optional[VisualRegions] = Field(
        default=None,
        validation_alias=AliasChoices("visualRegions", "visual_regions"),
        description="Bounding boxes for portrait, qrCode, mrzBlock"
    )
    visual_regions: Optional[VisualRegions] = Field(
        default=None,
        validation_alias=AliasChoices("visual_regions", "visualRegions"),
        description="Snake_case alias for visualRegions"
    )
    fieldMetadata: List[FieldMetadata] = Field(
        default_factory=list,
        validation_alias=AliasChoices("fieldMetadata", "field_metadata")
    )
    errors: List[str] = Field(default_factory=list, description="List of card validation/processing errors or warnings")

    @field_validator("fieldMetadata", mode="before")
    @classmethod
    def validate_field_metadata(cls, v: Any) -> List[Any]:
        if isinstance(v, dict):
            return list(v.values())
        if isinstance(v, list):
            return v
        return []


    @model_validator(mode="before")
    @classmethod
    def sync_visual_regions_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            vr = data.get("visualRegions") or data.get("visual_regions")
            if vr is not None:
                data["visualRegions"] = vr
                data["visual_regions"] = vr
        return data
