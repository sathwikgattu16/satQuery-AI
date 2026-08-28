"""
backend/main.py
FastAPI application entry point for SatQuery AI.
Owner: Member 1
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import settings
from backend.schemas.api_models import QueryRequest, QueryResponse
from backend.agent.registry import ModelRegistry
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize registry and controller
registry = ModelRegistry()
controller = SatQueryController(registry=registry)

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {"status": "online", "service": settings.PROJECT_NAME, "version": settings.VERSION}

@app.post("/api/query", response_model=QueryResponse)
def handle_query(request: QueryRequest) -> QueryResponse:
    """Primary inference endpoint for multimodal queries."""
    return controller.process_query(request)
