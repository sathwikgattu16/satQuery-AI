"""
ml/models/change_model.py

Bi-temporal Remote Sensing Change Analysis Specialist.
Owner: Member 5

Compares two co-registered Sentinel-2 observations using the shared
adapted Prithvi backbone.

No separate change-detection foundation model is used.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from backend.models.base import BaseSpecialist
from ml.models.prithvi_model import PrithviBackbone


class ChangeSpecialist(BaseSpecialist):
    """
    Detects representation-level change between T1 and T2.

    Inputs:
        T1: [B, 6, 1, 224, 224]
        T2: [B, 6, 1, 224, 224]

    Prithvi returns:
        [B, 197, 768]

    Token 0:
        CLS token

    Tokens 1:197:
        196 spatial patch tokens = 14 x 14
    """

    def __init__(
        self,
        backbone: Optional[PrithviBackbone] = None,
    ) -> None:

        super().__init__(
            name="ChangeSpecialist"
        )
        self.is_mock = False
        self.is_placeholder = False
        self.implementation_status = "real_feature_based_change"

        # IMPORTANT:
        # Reuse the shared adapted Prithvi backbone.
        self.backbone = (
            backbone
            or PrithviBackbone(
                checkpoint_path="checkpoints/prithvi_lora_best"
            )
        )

        self.device = self.backbone.device

    @staticmethod
    def _validate_image(
        image: torch.Tensor,
        name: str,
    ) -> None:

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor."
            )

        if image.ndim != 5:
            raise ValueError(
                f"{name} must have shape "
                "[B, 6, 1, H, W]. "
                f"Got {tuple(image.shape)}"
            )

        if image.shape[1] != 6:
            raise ValueError(
                f"{name} must contain 6 channels "
                "(B02-B07)."
            )

        if image.shape[2] != 1:
            raise ValueError(
                f"{name} must contain one temporal frame."
            )

        if image.shape[-2:] != (224, 224):
            raise ValueError(
                f"{name} must be 224x224. "
                f"Got {tuple(image.shape[-2:])}"
            )

    def _extract_features(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return final Prithvi feature tensor.

        Shape:
            [B, 197, 768]
        """

        return self.backbone.extract_features(
            image
        )

    @staticmethod
    def _representation_change(
        t1_features: torch.Tensor,
        t2_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CLS and patch-token change.

        Returns:
            cls_distance:
                [B]

            patch_change:
                [B, 196]

            normalized_patch_change:
                [B, 196]
        """

        # Normalize feature vectors before comparison.
        t1_norm = F.normalize(
            t1_features,
            dim=-1,
        )

        t2_norm = F.normalize(
            t2_features,
            dim=-1,
        )

        # --------------------------------------------------
        # CLS-level change
        # --------------------------------------------------

        cls_t1 = t1_norm[:, 0, :]
        cls_t2 = t2_norm[:, 0, :]

        cls_similarity = (
            F.cosine_similarity(
                cls_t1,
                cls_t2,
                dim=-1,
            )
        )

        # Convert similarity to distance.
        cls_distance = 1.0 - cls_similarity

        # --------------------------------------------------
        # Spatial patch-level change
        # --------------------------------------------------

        patches_t1 = t1_norm[:, 1:, :]
        patches_t2 = t2_norm[:, 1:, :]

        patch_similarity = (
            F.cosine_similarity(
                patches_t1,
                patches_t2,
                dim=-1,
            )
        )

        patch_change = 1.0 - patch_similarity

        # Normalize each image's patch scores to [0,1].
        min_value = patch_change.min(
            dim=1,
            keepdim=True,
        ).values

        max_value = patch_change.max(
            dim=1,
            keepdim=True,
        ).values

        normalized_patch_change = (
            patch_change - min_value
        ) / (
            max_value - min_value + 1e-8
        )

        return {
            "cls_distance": cls_distance,
            "patch_change": patch_change,
            "normalized_patch_change": normalized_patch_change,
        }

    @staticmethod
    def _heuristic_summary(
        cls_distance: float,
        mean_patch_change: float,
        max_patch_change: float,
    ) -> str:
        """
        Human-readable heuristic summary.

        IMPORTANT:
        These thresholds are representation-level heuristics,
        not calibrated change-detection probabilities.
        """

        if cls_distance < 0.03:
            level = "low"
        elif cls_distance < 0.10:
            level = "moderate"
        else:
            level = "high"

        return (
            f"{level.capitalize()} representation-level change "
            f"was detected between T1 and T2. "
            f"CLS distance={cls_distance:.4f}, "
            f"mean spatial change={mean_patch_change:.4f}, "
            f"maximum spatial change={max_patch_change:.4f}."
        )

    def analyze_change(
        self,
        image_t1: torch.Tensor,
        image_t2: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Run bi-temporal Prithvi feature comparison.

        Returns:
            Structured change representation.
        """

        self._validate_image(
            image_t1,
            "image_t1",
        )

        self._validate_image(
            image_t2,
            "image_t2",
        )

        image_t1 = image_t1.to(
            self.device,
            dtype=torch.float32,
        )

        image_t2 = image_t2.to(
            self.device,
            dtype=torch.float32,
        )

        with torch.no_grad():

            features_t1 = self._extract_features(
                image_t1
            )

            features_t2 = self._extract_features(
                image_t2
            )

            change = self._representation_change(
                features_t1,
                features_t2,
            )

        cls_distance = (
            change["cls_distance"]
        )

        patch_change = (
            change["patch_change"]
        )

        normalized_patch_change = (
            change["normalized_patch_change"]
        )

        # One scalar per batch item.
        mean_patch_change = (
            patch_change.mean(dim=1)
        )

        max_patch_change = (
            patch_change.max(dim=1).values
        )

        # Reshape 196 patch tokens -> 14x14 map.
        heatmap = (
            normalized_patch_change
            .reshape(
                normalized_patch_change.shape[0],
                14,
                14,
            )
        )

        return {
            "features_t1": features_t1,
            "features_t2": features_t2,
            "cls_distance": cls_distance,
            "patch_change": patch_change,
            "normalized_heatmap": heatmap,
            "mean_patch_change": mean_patch_change,
            "max_patch_change": max_patch_change,
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
                "ChangeSpecialist expects image_primary "
                "to be a preprocessed T1 tensor."
            )

        if image_secondary is None:
            raise ValueError(
                "Change analysis requires T1 and T2 images."
            )

        if not isinstance(
            image_secondary,
            torch.Tensor,
        ):
            raise TypeError(
                "ChangeSpecialist expects image_secondary "
                "to be a preprocessed T2 tensor."
            )

        result = self.analyze_change(
            image_t1=image_primary,
            image_t2=image_secondary,
        )

        cls_distance = (
            result["cls_distance"].mean().item()
        )

        mean_patch_change = (
            result["mean_patch_change"].mean().item()
        )

        max_patch_change = (
            result["max_patch_change"].mean().item()
        )

        answer = self._heuristic_summary(
            cls_distance,
            mean_patch_change,
            max_patch_change,
        )

        return {
            "answer": (
                f"{answer} Query: '{query}'"
            ),

            # This is NOT a calibrated confidence score.
            "confidence": None,

            "evidence": {
                "type": "heatmap",
                "data_url": None,
                "description": (
                    "14x14 spatial change map generated "
                    "from differences between T1 and T2 "
                    "Prithvi patch-token representations."
                ),
            },

            "execution_detail": (
                "Compared T1 and T2 independently through "
                "the shared adapted Prithvi backbone using "
                "CLS and spatial patch-token cosine distance."
            ),

            "change_metrics": {
                "cls_distance": cls_distance,
                "mean_patch_change": mean_patch_change,
                "max_patch_change": max_patch_change,
                "heatmap_height": 14,
                "heatmap_width": 14,
            },

            "adaptation_status": {
                "prithvi_adapted": bool(
                    self.backbone.is_adapted
                ),
                "change_head_trained": False,
            },
        }