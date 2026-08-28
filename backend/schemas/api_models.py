"""
backend/schemas/api_models.py
Pydantic API request and response data models.
Owner: Member 1
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

InputType = Literal["single", "optical_sar", "bitemporal"]

class QueryRequest(BaseModel):
    """Incoming query request payload."""
    query: str = Field(..., description="User natural language question or prompt")
    input_type: InputType = Field(..., description="Input image configuration: single, optical_sar, or bitemporal")
    image_primary: str = Field(..., description="Primary image path, identifier, or data URL")
    image_secondary: Optional[str] = Field(None, description="Secondary image for pairs (optical+SAR or bitemporal)")

class EvidencePayload(BaseModel):
    """Visual evidence supporting the model output."""
    type: str = Field(..., description="Evidence format: heatmap, bbox, mask, or overlay")
    data_url: Optional[str] = Field(None, description="Base64 or URL representation of the visual evidence")
    description: Optional[str] = Field(None, description="Human-readable explanation of the visual evidence")

class TraceStep(BaseModel):
    """Individual execution step for the auditable trace."""
    step_name: str
    status: str
    detail: Optional[str] = None

class ExecutionSummary(BaseModel):
    """Auditable execution trace for ISRO compliance."""
    steps: List[TraceStep] = Field(default_factory=list)
    total_duration_ms: Optional[int] = None

class QueryResponse(BaseModel):
    """Outgoing response payload matching api_contract.md."""
    answer: str
    confidence: float
    task: str
    specialists: List[str]
    evidence: Optional[EvidencePayload] = None
    execution_summary: ExecutionSummary
