"""
backend/tests/test_contract.py
Automated test suite verifying the locked API contract of POST /api/analyze.
Owner: Member 1
"""

import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def create_mock_file(filename: str = "test.png", content: bytes = b"dummy_satellite_data"):
    return (filename, io.BytesIO(content), "image/png")

def test_health_check():
    """Verify health check endpoint returns 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_single_image_vqa_contract():
    """Verify single-image VQA multipart request contract."""
    files = {"image": create_mock_file("sentinel2_l2a.png")}
    data = {"question": "What is the dominant land cover?", "task_hint": "vqa"}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["task"] == "vqa"
    assert "answer" in json_data
    assert isinstance(json_data["confidence"], float)
    assert isinstance(json_data["processing_time"], float)
    
    summary = json_data["execution_summary"]
    assert summary["selected_task"] == "vqa"
    assert summary["num_images_provided"] == 1
    assert len(summary["trace_steps"]) > 0

def test_single_image_caption_contract():
    """Verify single-image captioning multipart request contract."""
    files = {"image": create_mock_file("optical_scene.tif")}
    data = {"task_hint": "caption"}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["task"] == "caption"
    assert json_data["execution_summary"]["task_hint_provided"] == "caption"

def test_bitemporal_contract():
    """Verify bi-temporal (T1 + T2) multipart request contract."""
    files = {
        "image": create_mock_file("t1_baseline.png"),
        "image_t2": create_mock_file("t2_target.png"),
    }
    data = {"question": "What changed between T1 and T2?"}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["task"] == "change"
    assert json_data["execution_summary"]["num_images_provided"] == 2

def test_optical_sar_contract():
    """Verify optical + SAR radar multipart request contract."""
    files = {
        "image": create_mock_file("optical.png"),
        "sar": create_mock_file("sar_radar.tif"),
    }
    data = {"question": "Detect vessels beneath cloud cover using SAR."}

    response = client.post("/api/analyze", files=files, data=data)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["task"] == "multimodal_fusion"
    assert json_data["execution_summary"]["num_images_provided"] == 2

def test_missing_primary_image_fails_with_400():
    """Verify missing primary image returns HTTP 400 Bad Request."""
    data = {"question": "Describe the area"}
    response = client.post("/api/analyze", data=data)
    assert response.status_code == 422 or response.status_code == 400

def test_unsupported_file_extension_fails_with_400():
    """Verify unsupported file extensions are rejected with HTTP 400."""
    files = {"image": ("malicious.exe", io.BytesIO(b"content"), "application/octet-stream")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "unsupported" in response.json()["detail"].lower()
