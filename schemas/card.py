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


class FieldMetadata(BaseModel):
    field: str = Field(description="Name of extracted field")
    value: Optional[str] = Field(default=None, description="Final selected field value")
    source: str = Field(default="OCR", description="Selected data source: OCR, MRZ, or QR")
    keyword: Optional[str] = Field(default=None, description="Matched label keyword for selected source")
    language: Optional[str] = Field(default=None, description="Language of matched keyword (EN/VI)")
    confidence: float = Field(default=1.0, description="Confidence score 0.0 - 1.0")
    rawText: Optional[str] = Field(default=None, description="Exact uncorrected raw text from OCR/MRZ")
    ocrValue: Optional[str] = Field(default=None, description="Value extracted from OCR source")
    mrzValue: Optional[str] = Field(default=None, description="Value extracted from MRZ source")
    qrValue: Optional[str] = Field(default=None, description="Value extracted from QR source")
    ocrKeyword: Optional[str] = Field(default=None, description="Matched OCR keyword if detected")
    ocrLanguage: Optional[str] = Field(default=None, description="Language of matched OCR keyword")


class FieldValidationDetail(BaseModel):
    fieldName: str
    ocrValue: Optional[str] = None
    qrValue: Optional[str] = None
    mrzValue: Optional[str] = None
    status: str = "NOT_AVAILABLE"  # MATCH, MISMATCH, NOT_AVAILABLE


class CrossValidationResult(BaseModel):
    ocrMatchQr: Optional[bool] = None
    ocrMatchMrz: Optional[bool] = None
    mrzCheckDigitValid: Optional[bool] = None
    isExpired: Optional[bool] = None
    details: List[FieldValidationDetail] = []


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
    fieldMetadata: List[FieldMetadata] = Field(default_factory=list)
