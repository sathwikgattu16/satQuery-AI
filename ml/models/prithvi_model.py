"""
ml/models/prithvi_model.py

Shared Prithvi-EO-2.0 foundational backbone representation.
Owner: Member 3

One shared adapted Prithvi backbone is used by the VQA, Captioning,
Change, and Fusion specialists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
from peft import PeftModel
from terratorch.registry import BACKBONE_REGISTRY


MODEL_NAME = "prithvi_eo_v2_100_tl"


class PrithviBackbone:
    """Shared Prithvi-EO-2.0 backbone with optional LoRA adapter."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        print(f"Loading Prithvi: {MODEL_NAME}")
        print(f"Device: {self.device}")

        # Load the pretrained Prithvi-EO-2.0 backbone.
        self.model = BACKBONE_REGISTRY.build(
            MODEL_NAME,
            pretrained=True,
        )

        self.is_adapted = False

        # Load LoRA adapter when supplied.
        if self.checkpoint_path is not None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Prithvi adapter not found: "
                    f"{self.checkpoint_path}"
                )

            print(
                f"Loading LoRA adapter: "
                f"{self.checkpoint_path}"
            )

            self.model = PeftModel.from_pretrained(
                self.model,
                str(self.checkpoint_path),
            )

            self.is_adapted = True

        self.model = self.model.to(self.device)
        self.model.eval()

        print(
            "Prithvi ready "
            f"(adapted={self.is_adapted})"
        )

    def extract_features(
        self,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract final-layer Prithvi features.

        Expected input:
            [B, 6, 1, 224, 224]

        Returns:
            [B, 197, 768]

        The returned representation contains the CLS token followed
        by the spatial patch tokens.
        """

        if not isinstance(image_tensor, torch.Tensor):
            raise TypeError(
                "image_tensor must be a torch.Tensor"
            )

        if image_tensor.ndim != 5:
            raise ValueError(
                "Expected 5-D tensor [B, 6, 1, H, W], "
                f"got shape {tuple(image_tensor.shape)}"
            )

        if image_tensor.shape[1] != 6:
            raise ValueError(
                "Prithvi expects 6 input channels "
                f"(B02-B07); got {image_tensor.shape[1]}"
            )

        if image_tensor.shape[2] != 1:
            raise ValueError(
                "This Prithvi configuration expects "
                f"1 temporal frame; got {image_tensor.shape[2]}"
            )

        image_tensor = image_tensor.to(
            self.device,
            dtype=torch.float32,
        )

        # PeftModel exposes the underlying TerraTorch model
        # through base_model.model.
        if isinstance(self.model, PeftModel):
            backbone = self.model.base_model.model
        else:
            backbone = self.model

        with torch.no_grad():
            features = backbone.forward_features(
                image_tensor
            )

        if not features:
            raise RuntimeError(
                "Prithvi returned no feature tensors."
            )

        final_features = features[-1]

        if final_features.ndim != 3:
            raise RuntimeError(
                "Unexpected Prithvi feature shape: "
                f"{tuple(final_features.shape)}"
            )

        return final_features

    def extract_cls_features(
        self,
        image_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract only the CLS representation.

        Input:
            [B, 6, 1, 224, 224]

        Returns:
            [B, 768]
        """
        features = self.extract_features(image_tensor)
        return features[:, 0, :]

    def get_model(self):
        """Return the underlying Prithvi model."""
        return self.model

    def to(self, device: str):
        """Move the shared backbone to another device."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self