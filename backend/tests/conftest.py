"""
backend/tests/conftest.py
Pytest configuration for SatQuery AI backend tests.
Ensures unit test suite runs deterministically against mock specialists.
"""
import os

# Set test environment defaults before backend modules are imported
os.environ["USE_REAL_MODELS"] = "false"
os.environ["MODEL_DEVICE"] = "cpu"
