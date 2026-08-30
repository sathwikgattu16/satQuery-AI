"""
backend/main.py
FastAPI application entry point for SatQuery AI.
Owner: Member 1
"""

from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.schemas.api_models import QueryResponse
from backend.agent.compatibility import CompatibilityChecker
from backend.agent.data_adapter import RasterDataAdapter
from backend.agent.task_classifier import TaskClassifier
from backend.agent.registry import ModelRegistry
from backend.agent.model_manager import ModelLifecycleManager
from backend.agent.controller import SatQueryController

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal Remote Sensing Vision-Language Assistant Backend"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Initialize Registry and Lifecycle Manager (Singleton Model Cache)
registry = ModelRegistry()
lifecycle_manager = ModelLifecycleManager(registry=registry)
lifecycle_manager.initialize_models()

# 2. Initialize Agent Controller and Sub-modules
compatibility_checker = CompatibilityChecker()
data_adapter = RasterDataAdapter()
task_classifier = TaskClassifier()

controller = SatQueryController(
    registry=registry,
    compatibility_checker=compatibility_checker,
    data_adapter=data_adapter,
    task_classifier=task_classifier
)

@app.get("/")
def read_root():
    """Health check & model lifecycle diagnostics endpoint."""
    status_report = lifecycle_manager.get_status_report()
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "lifecycle": status_report
    }

@app.post("/api/analyze", response_model=QueryResponse)
async def analyze_satellite_data(
    task_hint: Optional[str] = Form(None),
    question: Optional[str] = Form(None),
    image: UploadFile = File(..., description="Primary satellite image (required)"),
    image_t2: Optional[UploadFile] = File(None, description="Time 2 observation image (for bi-temporal)"),
    sar: Optional[UploadFile] = File(None, description="SAR radar image (for optical + SAR)"),
) -> QueryResponse:
    """
    Primary multimodal inference endpoint matching frontend contract.
    Transport: multipart/form-data
    """
    return await controller.process_analysis_request(
        image=image,
        image_t2=image_t2,
        sar=sar,
        question=question,
        task_hint=task_hint
    )
