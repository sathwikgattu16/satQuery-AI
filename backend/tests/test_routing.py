"""
backend/tests/test_routing.py
Automated tests for Task Classification, Routing, Controller Orchestration, and Execution Trace.
Owner: Member 1
"""

import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def create_mock_file(filename: str = "scene.png", content: bytes = b"valid_image_bytes"):
    return (filename, io.BytesIO(content), "image/png")

def test_routing_single_image_vqa():
    """Verify single image without caption hint routes to VQA specialist."""
    files = {"image": create_mock_file("cartosat.png")}
    data = {"question": "What is the dominant land cover class?", "task_hint": "vqa"}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["task"] == "vqa"
    assert "Mock-VQASpecialist-v1" in res["execution_summary"]["models_used"]
    assert "[MOCK DEMO]" in res["answer"]
    assert res["confidence"] >= 0.85
    assert len(res["execution_summary"]["trace_steps"]) >= 6

def test_routing_single_image_caption():
    """Verify single image with caption hint or question routes to Caption specialist."""
    files = {"image": create_mock_file("coastal_harbor.png")}
    data = {"task_hint": "caption"}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["task"] == "caption"
    assert "Mock-CaptionSpecialist-v1" in res["execution_summary"]["models_used"]
    assert "coastal harbor" in res["answer"].lower()

def test_routing_bitemporal_change():
    """Verify T1 + T2 input pair routes to Change Detection specialist."""
    files = {
        "image": create_mock_file("t1_2022.tif"),
        "image_t2": create_mock_file("t2_2024.tif")
    }
    data = {"question": "What urban expansion occurred between T1 and T2?"}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["task"] == "change"
    assert "Mock-ChangeSpecialist-v1" in res["execution_summary"]["models_used"]
    assert res["visualization"]["type"] == "change_mask"
    assert "Changed Area" in res["visualization"]["metrics"]

def test_routing_optical_sar_fusion():
    """Verify Optical + SAR pair routes to Optical-SAR Fusion specialist."""
    files = {
        "image": create_mock_file("optical_rgb.png"),
        "sar": create_mock_file("risat1a_sar.tif")
    }
    data = {"question": "Identify maritime vessels under clouds."}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    assert res["success"] is True
    assert res["task"] == "multimodal_fusion"
    assert "Mock-OpticalSARSpecialist-v1" in res["execution_summary"]["models_used"]
    assert res["visualization"]["type"] == "heatmap"
    assert "Cloud Penetration" in res["visualization"]["metrics"]

def test_authoritative_override_bitemporal_with_wrong_hint():
    """
    Verify agent overrides conflicting user task hint ('vqa') when dual temporal images are provided.
    """
    files = {
        "image": create_mock_file("t1.png"),
        "image_t2": create_mock_file("t2.png")
    }
    data = {"task_hint": "vqa", "question": "Analyze change"}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    # Agent must authoritatively choose 'change'
    assert res["task"] == "change"
    assert res["execution_summary"]["task_hint_provided"] == "vqa"
    # Verify trace mentions override
    traces = " ".join(res["execution_summary"]["trace_steps"])
    assert "Override" in traces or "Overrode" in traces

def test_authoritative_override_optical_sar_with_wrong_hint():
    """
    Verify agent overrides conflicting user task hint ('caption') when SAR radar file is provided.
    """
    files = {
        "image": create_mock_file("optical.png"),
        "sar": create_mock_file("sar.tif")
    }
    data = {"task_hint": "caption"}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    res = response.json()

    # Agent must authoritatively choose 'multimodal_fusion'
    assert res["task"] == "multimodal_fusion"
    assert res["execution_summary"]["task_hint_provided"] == "caption"

def test_auditable_execution_trace_stages():
    """Verify all 7 decision and execution stages are represented in the trace."""
    files = {"image": create_mock_file("scene.png")}
    response = client.post("/api/analyze", files=files, data={"question": "Count buildings"})
    assert response.status_code == 200
    res = response.json()

    traces = res["execution_summary"]["trace_steps"]
    trace_text = "\n".join(traces)

    assert "Stage 1 [Validation]" in trace_text
    assert "Stage 2 [Sensor Compatibility]" in trace_text
    assert "Stage 3 [Data Adapter]" in trace_text
    assert "Stage 4 [Task Classification]" in trace_text
    assert "Stage 5 [Specialist Selection]" in trace_text
    assert "Stage 6 [Execution]" in trace_text
    assert "Stage 7 [Synthesis]" in trace_text
