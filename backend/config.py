"""
backend/config.py
Configuration settings for SatQuery AI backend service.
Owner: Member 1
"""

import os
from typing import List

class Settings:
    PROJECT_NAME: str = "SatQuery AI"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api"
    
    # Allowed remote sensing and benchmark image formats
    ALLOWED_EXTENSIONS: List[str] = [".tif", ".tiff", ".geotiff", ".png", ".jpg", ".jpeg"]
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    
    # Directories
    UPLOAD_DIR: str = os.path.join(os.path.dirname(__file__), "uploads")
    
settings = Settings()
