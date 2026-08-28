"""
backend/agent/task_classifier.py
Task classifier for routing natural language queries to the appropriate specialist model.
Owner: Member 1
"""

from typing import Dict, Any
from backend.schemas.api_models import QueryRequest

class TaskClassifier:
    """Classifies user intent and image configuration into specific remote sensing tasks."""

    def classify(self, request: QueryRequest) -> Dict[str, Any]:
        """
        Determine the analytical task and target specialist.
        Returns:
            Dict containing 'task_name' and 'specialist_key'.
        """
        input_type = request.input_type
        query_lower = request.query.lower()

        if input_type == "bitemporal":
            return {"task": "bitemporal_change", "specialist_key": "change"}
        elif input_type == "optical_sar":
            return {"task": "optical_sar_fusion", "specialist_key": "fusion"}
        else:
            # Single image routing: determine captioning vs VQA
            if any(k in query_lower for k in ["caption", "describe", "summary", "overview"]):
                return {"task": "single_image_captioning", "specialist_key": "captioning"}
            return {"task": "single_image_vqa", "specialist_key": "vqa"}
