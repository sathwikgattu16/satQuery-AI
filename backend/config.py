"""
backend/config.py
Configuration settings for SatQuery AI backend service.
Owner: Member 1
"""

import os
from typing import List, Optional

class Settings:
    PROJECT_NAME: str = "SatQuery AI"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    
    # Allowed remote sensing, benchmark, and synthetic demo image formats
    ALLOWED_EXTENSIONS: List[str] = [
        ".tif", ".tiff", ".geotiff",
        ".png", ".jpg", ".jpeg",
        ".svg", ".webp", ".bmp"
    ]
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]
    
    # Directories
    UPLOAD_DIR: str = os.path.join(os.path.dirname(__file__), "uploads")
    
    # Model Lifecycle & ML Settings
    PRITHVI_CHECKPOINT: Optional[str] = os.getenv("PRITHVI_CHECKPOINT", None)
    LORA_ADAPTER_PATH: Optional[str] = os.getenv("LORA_ADAPTER_PATH", None)
    MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "auto")
    USE_REAL_MODELS: bool = os.getenv("USE_REAL_MODELS", "false").lower() in ("true", "1", "yes")
    
settings = Settings()
