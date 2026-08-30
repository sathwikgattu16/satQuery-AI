"""
backend/tests/test_lifecycle.py
Automated test suite for Model Lifecycle, Singleton Caching, Device Resolution, and Real/Mock Switching.
Owner: Member 1
"""

import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.agent.model_manager import ModelLifecycleManager, resolve_device
from backend.agent.registry import ModelRegistry
from backend.models.base import BaseSpecialist

client = TestClient(app)

def test_device_resolution_cpu_and_auto():
    """Verify device resolution detects CPU and handles auto setting."""
    assert resolve_device("cpu") == "cpu"
    auto_dev = resolve_device("auto")
    assert auto_dev in ("cpu", "cuda:0")

def test_device_resolution_cuda_fallback():
    """Verify requesting unavailable CUDA device gracefully falls back to CPU."""
    dev = resolve_device("cuda:9999")
    assert dev == "cpu"

def test_mock_specialist_lifecycle_and_registry_lookup():
    """Verify lifecycle manager registers all 4 baseline specialists."""
    registry = ModelRegistry()
    manager = ModelLifecycleManager(registry=registry, use_real_models=False)
    manager.initialize_models()

    assert manager.get_specialist("vqa") is not None
    assert manager.get_specialist("caption") is not None
    assert manager.get_specialist("change") is not None
    assert manager.get_specialist("fusion") is not None

    specs = registry.list_specialists()
    assert "vqa" in specs
    assert "caption" in specs
    assert "change" in specs
    assert "fusion" in specs

def test_singleton_model_reuse_across_requests():
    """Verify specialist instances are singletons reused across multiple calls."""
    registry = ModelRegistry()
    manager = ModelLifecycleManager(registry=registry, use_real_models=False)

    spec_call_1 = manager.get_specialist("vqa")
    spec_call_2 = manager.get_specialist("vqa")

    assert spec_call_1 is spec_call_2
    assert id(spec_call_1) == id(spec_call_2)

def test_missing_real_weights_fallback():
    """Verify configuring a non-existent checkpoint falls back gracefully to mocks."""
    registry = ModelRegistry()
    manager = ModelLifecycleManager(
        registry=registry,
        use_real_models=True,
        checkpoint_path="non_existent_weights.pt"
    )
    manager.initialize_models()

    # Must still provide working specialist via mock fallback
    vqa_spec = manager.get_specialist("vqa")
    assert vqa_spec is not None
    assert isinstance(vqa_spec, BaseSpecialist)

def test_specialist_interface_compliance():
    """Verify all registered specialists implement BaseSpecialist with valid predict return signature."""
    registry = ModelRegistry()
    manager = ModelLifecycleManager(registry=registry, use_real_models=False)
    manager.initialize_models()

    for key in ("vqa", "caption", "change", "fusion"):
        spec = manager.get_specialist(key)
        assert isinstance(spec, BaseSpecialist)
        res = spec.predict(query="Test Query", image_primary=b"fake_bytes")
        assert "answer" in res
        assert "confidence" in res
        assert "evidence" in res
        assert isinstance(res["confidence"], float)

def test_health_check_reports_model_lifecycle_status():
    """Verify root health check endpoint reports active device and model lifecycle metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "online"
    assert "lifecycle" in data
    lifecycle = data["lifecycle"]
    assert "device" in lifecycle
    assert "use_real_models" in lifecycle
    assert "specialists_registered" in lifecycle
    assert len(lifecycle["specialists_registered"]) == 4
