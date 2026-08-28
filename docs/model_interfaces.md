# SATQUERY AI — Model & Specialist Interfaces

> **Source of Truth for Specialist Implementations**  
> All specialist models (`vqa_model.py`, `captioning_model.py`, `change_model.py`, `fusion_model.py`) must implement `BaseSpecialist`.

---

## 🐍 Python Specialist Interface Definition

Located in `backend/models/base.py`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseSpecialist(ABC):
    """Abstract base class for all SatQuery domain specialist models."""

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

        Returns a dictionary containing:
        - answer (str): Natural language response.
        - confidence (float): Numerical confidence (0.0 to 1.0).
        - evidence (Optional[Dict[str, Any]]): Visual evidence dict (type, data_url, description).
        - execution_detail (Optional[str]): Diagnostic message for the auditable trace.
        """
        pass
```
