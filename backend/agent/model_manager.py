"""
backend/agent/model_manager.py
Model Lifecycle & Integration Manager for SatQuery AI.
Handles device resolution (CUDA with CPU fallback), lazy/singleton instantiation,
configurable checkpoint loading, and seamless Real vs. Mock specialist switching.

Owner: Member 1
"""

import os
import logging
from typing import Dict, Any, Optional
import torch
from backend.config import settings
from backend.models.base import BaseSpecialist
from backend.agent.registry import ModelRegistry
from backend.models.mock_specialists import (
    MockVQASpecialist,
    MockCaptionSpecialist,
    MockChangeSpecialist,
    MockOpticalSARSpecialist
)

logger = logging.getLogger("satquery.lifecycle")

def resolve_device(device_setting: Optional[str] = None) -> str:
    """
    Resolves compute device with automatic CUDA detection and CPU fallback.
    """
    target = (device_setting or settings.MODEL_DEVICE).strip().lower()
    
    if target == "auto":
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"
    
    if target.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(f"Requested device '{target}' unavailable. Falling back to 'cpu'.")
            return "cpu"
        return target
        
    return "cpu"

class ModelLifecycleManager:
    """
    Singleton lifecycle manager for remote sensing models.
    Loads models once at startup/first request and caches instances across HTTP queries.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        use_real_models: Optional[bool] = None,
        device_setting: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        adapter_path: Optional[str] = None
    ):
        self.registry = registry or ModelRegistry()
        self.use_real_models = use_real_models if use_real_models is not None else settings.USE_REAL_MODELS
        self.device = resolve_device(device_setting)
        self.checkpoint_path = checkpoint_path or settings.PRITHVI_CHECKPOINT
        self.adapter_path = adapter_path or settings.LORA_ADAPTER_PATH
        
        self._is_initialized = False
        self._shared_backbone = None
        self._execution_mode_note = "MOCK MODE"

    def initialize_models(self) -> None:
        """
        Initializes and registers specialists in the model registry.
        Reuses cached instances across all subsequent queries.
        """
        if self._is_initialized:
            return

        if self.use_real_models:
            self._load_real_specialists()
        else:
            self._load_mock_specialists()

        self._is_initialized = True

    def _load_mock_specialists(self) -> None:
        """Loads and registers clean, domain-accurate mock specialists."""
        self.registry.register("vqa", MockVQASpecialist())
        self.registry.register("caption", MockCaptionSpecialist())
        self.registry.register("change", MockChangeSpecialist())
        self.registry.register("fusion", MockOpticalSARSpecialist())
        self._execution_mode_note = "MOCK MODE"
        logger.info(f"ModelLifecycleManager: Initialized 4 Mock Specialists on device '{self.device}'.")

    def _load_real_specialists(self) -> None:
        """
        Attempts to load Member 3's real Prithvi backbone and specialist heads from ml/.
        If weights are unavailable, notes clear diagnostic and falls back without false claims.
        """
        try:
            from ml.models.prithvi_model import PrithviBackbone
            from ml.models.vqa_model import VQASpecialist
            from ml.models.captioning_model import CaptioningSpecialist
            from ml.models.change_model import ChangeSpecialist
            from ml.models.fusion_model import FusionSpecialist

            # Check if real checkpoint exists
            if not self.checkpoint_path or not os.path.exists(self.checkpoint_path):
                self._execution_mode_note = "REAL MODEL REQUESTED — MOCK FALLBACK (Checkpoint not found)"
                logger.warning(
                    f"Configured Prithvi checkpoint not found at '{self.checkpoint_path}'. "
                    "Falling back to baseline mock specialists."
                )
                self._load_mock_specialists()
                return

            # Instantiate shared Prithvi backbone
            self._shared_backbone = PrithviBackbone(checkpoint_path=self.checkpoint_path)

            # Register Member 3 specialists hooked into the shared backbone
            vqa_inst = VQASpecialist(backbone=self._shared_backbone)
            caption_inst = CaptioningSpecialist(backbone=self._shared_backbone)
            change_inst = ChangeSpecialist(backbone=self._shared_backbone)
            fusion_inst = FusionSpecialist(backbone=self._shared_backbone)

            self.registry.register("vqa", vqa_inst)
            self.registry.register("caption", caption_inst)
            self.registry.register("change", change_inst)
            self.registry.register("fusion", fusion_inst)
            
            self._execution_mode_note = "REAL MODEL AVAILABLE"
            logger.info("ModelLifecycleManager: Initialized real Member 3 specialists with shared Prithvi backbone.")

        except Exception as err:
            self._execution_mode_note = f"REAL MODEL REQUESTED — MOCK FALLBACK (Error: {str(err)})"
            logger.error(
                f"Failed to load Member 3 real models ({str(err)}). Falling back to mock specialists."
            )
            self._load_mock_specialists()

    def get_specialist(self, key: str) -> Optional[BaseSpecialist]:
        """Retrieves a cached specialist instance by task key."""
        if not self._is_initialized:
            self.initialize_models()
        return self.registry.get(key)

    def get_status_report(self) -> Dict[str, Any]:
        """Returns diagnostic status of the model lifecycle and device."""
        if not self._is_initialized:
            self.initialize_models()

        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "use_real_models": self.use_real_models,
            "execution_mode": self._execution_mode_note,
            "checkpoint_path": self.checkpoint_path or "None (Default / Demo)",
            "adapter_path": self.adapter_path or "None (Zero-shot / Baseline)",
            "specialists_registered": self.registry.list_specialists(),
            "is_initialized": self._is_initialized
        }
