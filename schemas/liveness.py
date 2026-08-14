from typing import List
from pydantic import BaseModel, Field, ConfigDict


class LivenessResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    livenessVerified: bool = Field(default=False, description="True if liveness tests pass")
    livenessScore: float = Field(default=0.0, description="Overall liveness confidence score")
    threshold: float = Field(default=0.80, description="Decision threshold for liveness pass")
    checksPassed: List[str] = Field(default_factory=list, description="List of passed action/texture checks")
    errors: List[str] = Field(default_factory=list, description="List of liveness failure reason codes")
