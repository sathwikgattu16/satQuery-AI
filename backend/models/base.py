"""
backend/models/base.py
Abstract base class defining the uniform specialist model interface.
Owners: Member 1, Member 4, Member 5, Member 6
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseSpecialist(ABC):
    """
    Abstract base specialist for all SatQuery domain models.
    All specialists (VQA, Captioning, Change, Fusion) inherit from this class.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(
        self,
        query: str,
        image_primary: Any,
        image_secondary: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute prediction for the specialist task.

        Returns:
            Dict containing:
            - answer (str): Natural language text result.
            - confidence (float): Confidence score (0.0 - 1.0).
            - evidence (Optional[Dict]): Visual evidence metadata and data_url.
            - execution_detail (Optional[str]): Diagnostic message for auditable trace.
        """
        raise NotImplementedError("Specialists must implement the predict method.")
