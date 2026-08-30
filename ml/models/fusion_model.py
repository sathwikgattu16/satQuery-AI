"""
ml/models/fusion_model.py

Optical + SAR Joint Analysis Specialist.
Owner: Member 6

Feature-level optical/SAR fusion using:
    - shared adapted Prithvi-EO-2.0 optical representation
    - deterministic Sentinel-1 VH/VV SAR statistics

No separately trained fusion model is assumed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from backend.models.base import BaseSpecialist
from ml.models.prithvi_model import PrithviBackbone


class FusionSpecialist(BaseSpecialist):
    """
    Optical + SAR feature-level fusion.

    Optical:
        [B, 6, 1, 224, 224]
        -> adapted Prithvi
        -> [B, 768]

    SAR:
        [B, 2, 1, H, W]
        -> deterministic VH/VV statistics
        -> [B, 256]

    Fusion:
        [B, 768] + [B, 256]
        -> [B, 1024]
    """

    def __init__(
        self,
        backbone: Optional[PrithviBackbone] = None,
    ) -> None:

        super().__init__(
            name="FusionSpecialist"
        )

        # IMPORTANT:
        # The backend should inject the shared adapted
        # PrithviBackbone here.
        self.backbone = (
            backbone
            or PrithviBackbone(
                checkpoint_path="checkpoints/prithvi_lora_best"
            )
        )

        self.device = self.backbone.device

    @staticmethod
    def _validate_optical(
        optical: torch.Tensor,
    ) -> None:

        if optical.ndim != 5:
            raise ValueError(
                "Optical input must have shape "
                "[B, 6, 1, H, W]. "
                f"Got {tuple(optical.shape)}"
            )

        if optical.shape[1] != 6:
            raise ValueError(
                "Optical input must contain 6 channels "
                "(B02-B07)."
            )

        if optical.shape[2] != 1:
            raise ValueError(
                "Optical input must contain one temporal frame."
            )

    @staticmethod
    def _validate_sar(
        sar: torch.Tensor,
    ) -> None:

        if sar.ndim != 5:
            raise ValueError(
                "SAR input must have shape "
                "[B, 2, 1, H, W]. "
                f"Got {tuple(sar.shape)}"
            )

        if sar.shape[1] != 2:
            raise ValueError(
                "SAR input must contain two channels "
                "(VH and VV)."
            )

        if sar.shape[2] != 1:
            raise ValueError(
                "SAR input must contain one temporal frame."
            )

    @staticmethod
    def _sar_features(
        sar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert VH/VV into a deterministic 256-dimensional
        representation.

        We deliberately do not use randomly initialized
        trainable layers here.

        Features per channel:
            mean
            std
            min
            max
            median
            q25
            q75

        plus VH/VV interaction statistics.

        The final vector is deterministically expanded to 256
        dimensions by repeating the base statistics.
        """

        # Remove singleton temporal dimension:
        # [B,2,1,H,W] -> [B,2,H,W]
        x = sar.squeeze(2)

        vh = x[:, 0]
        vv = x[:, 1]

        def stats(a: torch.Tensor) -> torch.Tensor:
            flat = a.flatten(1)

            mean = flat.mean(dim=1)
            std = flat.std(dim=1)
            min_v = flat.min(dim=1).values
            max_v = flat.max(dim=1).values

            q25 = torch.quantile(
                flat,
                0.25,
                dim=1,
            )

            median = torch.quantile(
                flat,
                0.50,
                dim=1,
            )

            q75 = torch.quantile(
                flat,
                0.75,
                dim=1,
            )

            return torch.stack(
                [
                    mean,
                    std,
                    min_v,
                    max_v,
                    q25,
                    median,
                    q75,
                ],
                dim=1,
            )

        vh_stats = stats(vh)
        vv_stats = stats(vv)

        # Interaction information.
        diff = stats(vh - vv)
        ratio = stats(
            vh / (vv.abs() + 1e-6)
        )

        base = torch.cat(
            [
                vh_stats,
                vv_stats,
                diff,
                ratio,
            ],
            dim=1,
        )

        # 28 deterministic statistics -> 252 would be awkward.
        # Pad to exactly 256.
        padding = torch.zeros(
            base.shape[0],
            256 - base.shape[1],
            device=base.device,
            dtype=base.dtype,
        )

        return torch.cat(
            [base, padding],
            dim=1,
        )

    def extract_fused_features(
        self,
        optical_tensor: torch.Tensor,
        sar_tensor: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        self._validate_optical(
            optical_tensor
        )

        self._validate_sar(
            sar_tensor
        )

        optical_tensor = optical_tensor.to(
            self.device,
            dtype=torch.float32,
        )

        sar_tensor = sar_tensor.to(
            self.device,
            dtype=torch.float32,
        )

        # Shared adapted Prithvi -> CLS representation.
        optical_features = (
            self.backbone.extract_cls_features(
                optical_tensor
            )
        )

        # Deterministic SAR representation.
        with torch.no_grad():
            sar_features = self._sar_features(
                sar_tensor
            )

        # Feature-level multimodal fusion.
        fused_features = torch.cat(
            [
                optical_features,
                sar_features,
            ],
            dim=-1,
        )

        return {
            "optical": optical_features,
            "sar": sar_features,
            "fused": fused_features,
        }

    def predict(
        self,
        query: str,
        image_primary: Any,
        image_secondary: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        if not isinstance(
            image_primary,
            torch.Tensor,
        ):
            raise TypeError(
                "FusionSpecialist expects "
                "preprocessed optical tensor."
            )

        if image_secondary is None:
            raise ValueError(
                "Fusion requires both optical "
                "and SAR inputs."
            )

        if not isinstance(
            image_secondary,
            torch.Tensor,
        ):
            raise TypeError(
                "FusionSpecialist expects "
                "preprocessed SAR tensor."
            )

        representations = (
            self.extract_fused_features(
                optical_tensor=image_primary,
                sar_tensor=image_secondary,
            )
        )

        optical = representations["optical"]
        sar = representations["sar"]
        fused = representations["fused"]

        optical_norm = (
            torch.linalg.vector_norm(
                optical,
                dim=-1,
            )
            .mean()
            .item()
        )

        sar_norm = (
            torch.linalg.vector_norm(
                sar,
                dim=-1,
            )
            .mean()
            .item()
        )

        fused_norm = (
            torch.linalg.vector_norm(
                fused,
                dim=-1,
            )
            .mean()
            .item()
        )

        return {
            "answer": (
                f"Optical-SAR feature fusion completed "
                f"for query: '{query}'."
            ),

            # Do NOT pretend this is calibrated model confidence.
            "confidence": None,

            "evidence": {
                "type": "feature_fusion",
                "data_url": None,
                "description": (
                    "Combined adapted Prithvi optical "
                    "representation with Sentinel-1 "
                    "VH/VV SAR representation."
                ),
            },

            "execution_detail": (
                "Used the shared adapted Prithvi backbone "
                "for Sentinel-2 optical features and a "
                "deterministic Sentinel-1 VH/VV feature "
                "representation. No separately trained "
                "fusion head is assumed."
            ),

            "fusion_features": {
                "optical_dim": int(
                    optical.shape[-1]
                ),
                "sar_dim": int(
                    sar.shape[-1]
                ),
                "fused_dim": int(
                    fused.shape[-1]
                ),
                "optical_norm": optical_norm,
                "sar_norm": sar_norm,
                "fused_norm": fused_norm,
            },

            "adaptation_status": {
                "prithvi_adapted": bool(
                    self.backbone.is_adapted
                ),
                "fusion_head_trained": False,
            },
        }