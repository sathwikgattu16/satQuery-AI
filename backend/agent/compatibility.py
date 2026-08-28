"""
backend/agent/compatibility.py
Input compatibility checker for validating image formats, counts, and modality alignment.
Owner: Member 1
"""

from typing import Tuple, Optional
from backend.schemas.api_models import QueryRequest

class CompatibilityChecker:
    """Validates user query inputs against modal requirements."""

    def validate(self, request: QueryRequest) -> Tuple[bool, Optional[str]]:
        """
        Validate input configuration.
        Returns:
            (is_valid: bool, error_message: Optional[str])
        """
        if request.input_type == "single":
            if not request.image_primary:
                return False, "Primary image is required for single-image mode."
        elif request.input_type in ("optical_sar", "bitemporal"):
            if not request.image_primary or not request.image_secondary:
                return False, f"Both primary and secondary images are required for {request.input_type} mode."
        else:
            return False, f"Unsupported input_type: {request.input_type}"

        return True, None
