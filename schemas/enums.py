from enum import Enum


class CardType(str, Enum):
    CCCD_OLD = "CCCD_OLD"
    CCCD_NEW = "CCCD_NEW"
    UNKNOWN = "UNKNOWN"


class VerificationDecision(str, Enum):
    MATCH = "MATCH"
    BORDERLINE = "BORDERLINE"
    MISMATCH = "MISMATCH"


class CrossValidationStatus(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    CONFLICT = "CONFLICT"
    OCR_ONLY = "OCR_ONLY"
    QR_ONLY = "QR_ONLY"
    MRZ_ONLY = "MRZ_ONLY"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class EkycOutcome(str, Enum):
    EKYC_VERIFIED = "EKYC_VERIFIED"
    EKYC_NOT_VERIFIED = "EKYC_NOT_VERIFIED"


class EkycExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
