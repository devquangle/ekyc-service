from typing import Optional, List
from pydantic import BaseModel, Field


class ExtractedCardData(BaseModel):
    identityNumber: Optional[str] = Field(default=None, description="12-digit Personal Identification Number")
    fullName: Optional[str] = Field(default=None, description="Full name of card holder")
    dateOfBirth: Optional[str] = Field(default=None, description="Date of birth in ISO format YYYY-MM-DD")
    gender: Optional[str] = Field(default=None, description="Gender (Nam / Nữ)")
    nationality: Optional[str] = Field(default="Việt Nam", description="Nationality")
    placeOfBirth: Optional[str] = Field(default=None, description="Place of birth (Place of birth on CCCD_NEW)")
    placeOfOrigin: Optional[str] = Field(default=None, description="Place of origin (Place of origin on CCCD_OLD)")
    placeOfResidence: Optional[str] = Field(default=None, description="Place of residence")
    dateOfIssue: Optional[str] = Field(default=None, description="Date of issue in ISO format YYYY-MM-DD")
    dateOfExpiry: Optional[str] = Field(default=None, description="Date of expiry in ISO format YYYY-MM-DD")


class CrossValidationDetail(BaseModel):
    fieldName: str
    ocrValue: Optional[str] = None
    qrValue: Optional[str] = None
    mrzValue: Optional[str] = None
    status: str  # MATCH, MISMATCH, NOT_AVAILABLE


class CrossValidationResult(BaseModel):
    ocrMatchQr: Optional[bool] = Field(default=None, description="True if match, False if mismatch, None if QR not available")
    ocrMatchMrz: Optional[bool] = Field(default=None, description="True if match, False if mismatch, None if MRZ not available")
    mrzCheckDigitValid: bool = True
    isExpired: bool = False
    details: List[CrossValidationDetail] = []


class QualityChecks(BaseModel):
    isBlur: bool = False
    hasGlare: bool = False
    isCropped: bool = False


class CardProcessResponse(BaseModel):
    cardVerified: bool = False
    cardType: str = "UNKNOWN"  # CCCD_NEW, CCCD_OLD, UNKNOWN
    cardTypeConfidence: float = Field(default=0.0, description="Confidence score for card type classification (0.0 to 1.0)")
    extractedData: ExtractedCardData = Field(default_factory=ExtractedCardData)
    crossValidation: CrossValidationResult = Field(default_factory=CrossValidationResult)
    qualityChecks: QualityChecks = Field(default_factory=QualityChecks)
    errors: List[str] = []
