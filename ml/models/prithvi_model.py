"""
ml/models/prithvi_model.py
Shared Prithvi-EO-2.0 foundational backbone representation.
Owner: Member 3

NOTE:
Prithvi-EO-2.0 is the single shared adapted remote-sensing backbone across the project.
Individual specialists (VQA, Captioning, Change, Fusion) hook into features extracted
by this shared model rather than maintaining separate training pipelines.
"""

from typing import Any, Optional

class PrithviBackbone:
    """Shared remote sensing foundation model wrapper."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.is_adapted = checkpoint_path is not None

    def extract_features(self, image_tensor: Any) -> Any:
        """
        Extract dense spatial and contextual feature embeddings from multi-band imagery.
        """
        # Placeholder: forward pass through Prithvi-EO-2.0 ViT
        return None
