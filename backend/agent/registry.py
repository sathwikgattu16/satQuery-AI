"""
backend/agent/registry.py
Specialist model registry for registering and retrieving task specialists.
Owner: Member 1
"""

from typing import Dict, Optional
from backend.models.base import BaseSpecialist

class ModelRegistry:
    """Central registry holding active specialist models."""

    def __init__(self):
        self._specialists: Dict[str, BaseSpecialist] = {}

    def register(self, key: str, specialist: BaseSpecialist) -> None:
        """Register a specialist instance under a unique key."""
        self._specialists[key] = specialist

    def get(self, key: str) -> Optional[BaseSpecialist]:
        """Retrieve a specialist instance by key."""
        return self._specialists.get(key)

    def list_specialists(self) -> Dict[str, str]:
        """List all registered specialists and their names."""
        return {key: spec.name for key, spec in self._specialists.items()}
