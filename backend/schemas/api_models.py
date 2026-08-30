"""
backend/schemas/api_models.py
Pydantic API request and response data models matching frontend/src/types/index.ts.
Owner: Member 1
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field

# Supported visual evidence types
EvidenceType = Literal["overlay", "heatmap", "change_mask", "split", "image", "diff"]

class VisualEvidence(BaseModel):
    """Visual evidence supporting the model output."""
    type: Optional[EvidenceType] = Field(None, description="Visual overlay format")
    title: Optional[str] = Field(None, description="Title of the visual evidence")
    url: Optional[str] = Field(None, description="HTTP or public URL of the raster/overlay")
    base64: Optional[str] = Field(None, description="Base64 encoded raster/image data")
    description: Optional[str] = Field(None, description="Human-readable explanation of the visual evidence")
    metrics: Optional[Dict[str, Union[str, int, float]]] = Field(
        default=None, description="Extracted numerical geospatial metrics"
    )

class ExecutionSummary(BaseModel):
    """Auditable execution trace for ISRO hackathon compliance."""
    selected_task: str = Field(..., description="Authoritative task selected by backend agent")
    task_hint_provided: Optional[str] = Field("none", description="UI-provided task hint")
    models_used: List[str] = Field(default_factory=list, description="List of neural models/specialists deployed")
    num_images_provided: int = Field(..., description="Count of input images successfully ingested")
    compatibility_notes: str = Field(..., description="Diagnostic notes on sensor alignment and compatibility")
    trace_steps: List[str] = Field(default_factory=list, description="Step-by-step agent decision logs")
    implementation_status: Optional[str] = Field(
        None, description="Implementation status of the specialist (e.g. real_feature_based_change, real_feature_based_fusion, vqa_placeholder, caption_placeholder, mock_demo)"
    )

class QueryResponse(BaseModel):
    """
    Locked API response payload matching frontend expectations.
    Returned by POST /api/analyze.
    """
    success: bool = Field(True, description="Whether the analysis succeeded")
    task: str = Field(..., description="Executed task identifier (vqa, caption, change, multimodal_fusion)")
    answer: str = Field(..., description="Natural language geospatial synthesis")
    confidence: Optional[float] = Field(None, description="Confidence score between 0.0 and 1.0")
    processing_time: float = Field(..., description="Total pipeline latency in seconds")
    execution_summary: ExecutionSummary = Field(..., description="Auditable agent execution trace")
    visualization: Optional[Union[VisualEvidence, str]] = Field(
        None, description="Visual evidence card or overlay URL"
    )
    error: Optional[str] = Field(None, description="Error message if success is false")
