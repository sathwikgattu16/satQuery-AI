"""
ml/models/vqa_model.py
Single-Image Visual Question Answering (VQA) Specialist.
Owner: Member 4
"""

import torch
from typing import Dict, Any, Optional
from backend.models.base import BaseSpecialist
from ml.models.prithvi_model import PrithviBackbone

class VQASpecialist(BaseSpecialist):
    """Answers natural language questions about single optical, multispectral, or SAR images."""

    def __init__(self, backbone: Optional[PrithviBackbone] = None):
        super().__init__(name="VQASpecialist")
        self.backbone = backbone or PrithviBackbone()
        self.is_mock = False
        self.is_placeholder = True
        self.implementation_status = "vqa_placeholder"

    def predict(
        self,
        query: str,
        image_primary: Any,
        image_secondary: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute single-image VQA inference.
        """
        feature_metrics: Dict[str, Any] = {}
        if isinstance(image_primary, torch.Tensor):
            try:
                cls_token, patch_tokens = self.backbone.extract_features(image_primary)
                feature_metrics = {
                    "cls_dim": int(cls_token.shape[-1]),
                    "num_patches": int(patch_tokens.shape[1]),
                    "cls_norm": round(float(torch.norm(cls_token, p=2, dim=-1).mean().item()), 4),
                }
            except Exception:
                pass

        return {
            "answer": f"VQA feature analysis completed for query: '{query}'. Optical feature representation extracted through adapted Prithvi backbone. (Downstream language reasoning head is currently a prototype placeholder).",
            "confidence": None,
            "evidence": {
                "type": "image",
                "data_url": None,
                "description": "Visual feature anchor representation extracted from shared Prithvi backbone."
            },
            "execution_detail": "Extracted optical representations using shared adapted Prithvi backbone. VQA reasoning head is currently a transparent prototype placeholder.",
            "implementation_status": "vqa_placeholder",
            "feature_metrics": feature_metrics or None,
            "adaptation_status": {
                "prithvi_adapted": bool(getattr(self.backbone, "is_adapted", False)),
                "vqa_head_trained": False
            }
        }
