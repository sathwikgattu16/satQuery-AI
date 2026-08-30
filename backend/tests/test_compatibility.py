"""
backend/tests/test_compatibility.py
Comprehensive test suite for the Input Compatibility Layer.
Owner: Member 1
"""

import io
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def create_mock_file(filename: str = "scene.tif", content: bytes = b"valid_geospatial_raster_bytes_header"):
    return (filename, io.BytesIO(content), "image/tiff" if filename.endswith(".tif") else "image/png")

# --- POSITIVE VALIDATION TESTS ---

def test_valid_single_image_geotiff():
    """Verify single GeoTIFF file is accepted and validated."""
    files = {"image": create_mock_file("cartosat_scene.tif")}
    response = client.post("/api/analyze", files=files, data={"question": "Identify features"})
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["execution_summary"]["num_images_provided"] == 1
    assert "Single scene" in res["execution_summary"]["compatibility_notes"]

def test_valid_single_image_svg_demo():
    """Verify frontend synthetic SVG demo file is accepted."""
    svg_content = b"<svg xmlns='http://www.w3.org/2000/svg'><rect width='100' height='100'/></svg>"
    files = {"image": ("isro_demo.svg", io.BytesIO(svg_content), "image/svg+xml")}
    response = client.post("/api/analyze", files=files, data={"task_hint": "vqa"})
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["execution_summary"]["num_images_provided"] == 1

def test_valid_bitemporal_pair():
    """Verify bi-temporal pair (T1 + T2) is properly co-registered and noted."""
    files = {
        "image": create_mock_file("t1_baseline_2022.tif"),
        "image_t2": create_mock_file("t2_target_2024.tif"),
    }
    response = client.post("/api/analyze", files=files, data={"question": "What changed?"})
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["task"] == "change"
    assert res["execution_summary"]["num_images_provided"] == 2
    assert "T1 Baseline" in res["execution_summary"]["compatibility_notes"]
    assert "T2 Observation" in res["execution_summary"]["compatibility_notes"]

def test_valid_optical_sar_pair():
    """Verify Optical + SAR pair is co-registered and noted."""
    files = {
        "image": create_mock_file("sentinel2_optical.png"),
        "sar": create_mock_file("risat1a_sar_polarimetric.tif"),
    }
    response = client.post("/api/analyze", files=files, data={"question": "Find ship beneath clouds"})
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["task"] == "multimodal_fusion"
    assert res["execution_summary"]["num_images_provided"] == 2
    assert "Optical imagery" in res["execution_summary"]["compatibility_notes"]
    assert "SAR radar" in res["execution_summary"]["compatibility_notes"]

# --- NEGATIVE VALIDATION TESTS ---

def test_empty_primary_file_rejected_with_400():
    """Verify uploading an empty file (0 bytes) raises HTTP 400."""
    files = {"image": ("empty_corrupted.png", io.BytesIO(b""), "image/png")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_empty_secondary_t2_file_rejected_with_400():
    """Verify uploading an empty T2 file raises HTTP 400."""
    files = {
        "image": create_mock_file("t1.tif"),
        "image_t2": ("t2_empty.tif", io.BytesIO(b""), "image/tiff"),
    }
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()

def test_unsupported_image_extension_rejected_with_400():
    """Verify unallowed file extensions (.zip, .pdf) raise HTTP 400."""
    files = {"image": ("archive.zip", io.BytesIO(b"PK\x03\x04dummy"), "application/zip")}
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "unsupported format" in response.json()["detail"].lower()

def test_unsupported_sar_extension_rejected_with_400():
    """Verify unallowed SAR file extension raises HTTP 400."""
    files = {
        "image": create_mock_file("optical.png"),
        "sar": ("sar_raw.bin", io.BytesIO(b"binary"), "application/octet-stream"),
    }
    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400
    assert "unsupported format" in response.json()["detail"].lower()
