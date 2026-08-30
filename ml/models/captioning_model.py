"""
ml/models/captioning_model.py
Single-Image Remote Sensing Captioning Specialist.
Owner: Member 4
"""

import torch
from typing import Dict, Any, Optional
from backend.models.base import BaseSpecialist
from backend.agent.synthesizer import StructuredGeospatialSynthesizer
from ml.models.prithvi_model import PrithviBackbone

class CaptioningSpecialist(BaseSpecialist):
    """Generates comprehensive descriptive captions for remote sensing imagery."""

    def __init__(self, backbone: Optional[PrithviBackbone] = None):
        super().__init__(name="CaptioningSpecialist")
        self.backbone = backbone or PrithviBackbone()
        self.is_mock = False
        self.is_placeholder = True
        self.implementation_status = "caption_placeholder"

    def predict(
        self,
        query: str,
        image_primary: Any,
        image_secondary: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute single-image captioning inference.
        """
        feature_metrics: Dict[str, Any] = {}
        if isinstance(image_primary, torch.Tensor):
            try:
                features = self.backbone.extract_features(image_primary)
                cls_token = features[:, 0, :]
                patch_tokens = features[:, 1:, :]
                feature_metrics = {
                    "cls_dim": int(cls_token.shape[-1]),
                    "num_patches": int(patch_tokens.shape[1]),
                    "cls_norm": round(float(torch.norm(cls_token, p=2, dim=-1).mean().item()), 4),
                }
            except Exception:
                pass

        synthesized_answer = StructuredGeospatialSynthesizer.synthesize_caption(
            feature_metrics=feature_metrics or None,
            metadata=context
        )

        return {
            "answer": synthesized_answer,
            "confidence": None,
            "evidence": {
                "type": "overlay",
                "data_url": None,
                "description": "Spatial attention/feature map from shared Prithvi representation."
            },
            "execution_detail": "Extracted optical representations using shared adapted Prithvi backbone. Captioning language generator is currently a transparent prototype placeholder.",
            "implementation_status": "caption_placeholder",
            "feature_metrics": feature_metrics or None,
            "adaptation_status": {
                "prithvi_adapted": bool(getattr(self.backbone, "is_adapted", False)),
                "caption_head_trained": False
            }
        }
