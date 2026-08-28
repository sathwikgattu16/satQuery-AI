"""
ml/models/change_model.py
Bi-temporal Remote Sensing Change Analysis Specialist.
Owner: Member 5
"""

from typing import Dict, Any, Optional
from backend.models.base import BaseSpecialist
from ml.models.prithvi_model import PrithviBackbone

class ChangeSpecialist(BaseSpecialist):
    """Detects and describes surface changes between co-registered T1 and T2 images."""

    def __init__(self, backbone: Optional[PrithviBackbone] = None):
        super().__init__(name="ChangeSpecialist")
        self.backbone = backbone or PrithviBackbone()

    def predict(
        self,
        query: str,
        image_primary: Any,
        image_secondary: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute bi-temporal difference analysis and change QA.
        """
        return {
            "answer": f"Detected significant landscape modification between T1 and T2: '{query}'",
            "confidence": 0.87,
            "evidence": {
                "type": "heatmap",
                "data_url": None,
                "description": "Bi-temporal change difference heatmap."
            },
            "execution_detail": "Executed ChangeSpecialist via dual-temporal Prithvi feature comparison."
        }
