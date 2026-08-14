from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ExtractedCardData(BaseModel):
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
    portrait: Optional[List[float]] = Field(default=None, description="[x_min, y_min, x_max, y_max] of portrait face")
    qrCode: Optional[List[float]] = Field(default=None, description="[x_min, y_min, x_max, y_max] of QR code")
    mrzBlock: Optional[List[float]] = Field(default=None, description="[x_min, y_min, x_max, y_max] of MRZ block")


class FieldMetadata(BaseModel):
    field: str = Field(description="Name of extracted field")
    value: Optional[str] = Field(default=None, description="Final selected field value")
    label: Optional[str] = Field(default=None, description="Clean label text")
    label_box: Optional[List[float]] = Field(default=None, description="Bounding box of label [x_min, y_min, x_max, y_max]")
    value_box: Optional[List[float]] = Field(default=None, description="Bounding box of value [x_min, y_min, x_max, y_max]")
    labelBox: Optional[List[float]] = Field(default=None, description="CamelCase alias for label_box")
    valueBox: Optional[List[float]] = Field(default=None, description="CamelCase alias for value_box")
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

    def model_post_init(self, __context):
        if self.label_box and not self.labelBox:
            self.labelBox = self.label_box
        elif self.labelBox and not self.label_box:
            self.label_box = self.labelBox
        if self.value_box and not self.valueBox:
            self.valueBox = self.value_box
        elif self.valueBox and not self.value_box:
            self.value_box = self.valueBox
        if self.keyword and not self.label:
            self.label = self.keyword


class CrossValidationDetail(BaseModel):
    fieldName: str
    ocrValue: Optional[str] = None
    qrValue: Optional[str] = None
    mrzValue: Optional[str] = None
    status: str = "NOT_AVAILABLE"  # MATCH, MISMATCH, NOT_AVAILABLE


FieldValidationDetail = CrossValidationDetail


class CrossValidationResult(BaseModel):
    ocrMatchQr: Optional[bool] = None
    ocrMatchMrz: Optional[bool] = None
    mrzCheckDigitValid: Optional[bool] = None
    isExpired: Optional[bool] = None
    details: List[CrossValidationDetail] = []


class QualityChecks(BaseModel):
    isBlur: bool = False
    hasGlare: bool = False
    isCropped: bool = False


class CardProcessResponse(BaseModel):
    cardVerified: bool = True
    cardType: str = Field(default="CCCD_OLD", description="Detected Card Type: CCCD_OLD, CCCD_NEW, UNKNOWN")
    cardTypeConfidence: float = Field(default=1.0, description="Card type detection confidence 0.0 - 1.0")
    extractedData: ExtractedCardData = Field(default_factory=ExtractedCardData)
    crossValidation: CrossValidationResult = Field(default_factory=CrossValidationResult)
    qualityChecks: QualityChecks = Field(default_factory=QualityChecks)
    visualRegions: Optional[VisualRegions] = Field(default=None, description="Bounding boxes for portrait, qrCode, mrzBlock")
    visual_regions: Optional[VisualRegions] = Field(default=None, description="Alias for visualRegions")
    fieldMetadata: List[FieldMetadata] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list, description="List of card validation/processing errors or warnings")

    def model_post_init(self, __context):
        if self.visualRegions and not self.visual_regions:
            self.visual_regions = self.visualRegions
        elif self.visual_regions and not self.visualRegions:
            self.visualRegions = self.visual_regions
