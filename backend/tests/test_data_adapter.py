"""
backend/tests/test_data_adapter.py
Automated tests for Backend -> ML Data Adapter.
Owner: Member 1
"""

import io
import pytest
import numpy as np
import tifffile
from PIL import Image
import torch
from fastapi import UploadFile
from backend.agent.data_adapter import RasterDataAdapter, AdaptedDataPayload

def create_mock_upload_file(filename: str, content: bytes, content_type: str = "image/tiff") -> UploadFile:
    """Helper to instantiate a FastAPI UploadFile in-memory."""
    return UploadFile(
        filename=filename,
        file=io.BytesIO(content),
        headers={"content-type": content_type}
    )

def create_multiband_tiff_bytes(channels: int = 6, height: int = 120, width: int = 120) -> bytes:
    """Generates an in-memory multi-band GeoTIFF array."""
    # Shape: [C, H, W]
    arr = (np.random.rand(channels, height, width) * 10000).astype(np.float32)
    bio = io.BytesIO()
    tifffile.imwrite(bio, arr)
    return bio.getvalue()

def create_rgb_png_bytes() -> bytes:
    """Generates an in-memory 3-channel RGB PNG."""
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()

@pytest.mark.anyio
async def test_multispectral_6band_tiff_processing():
    """
    Test 1, 2, 3: Valid multispectral GeoTIFF creates [1, 6, 1, 224, 224] tensor
    with B02-B07 band mapping without error.
    """
    adapter = RasterDataAdapter()
    tiff_bytes = create_multiband_tiff_bytes(channels=6, height=120, width=120)
    upload_file = create_mock_upload_file("sentinel2_l2a_6band.tif", tiff_bytes)

    payload: AdaptedDataPayload = await adapter.adapt_inputs(image=upload_file)

    assert payload.is_multispectral is True
    assert payload.is_demo is False
    assert payload.primary_tensor is not None

    # Verify exact Prithvi shape: [B=1, C=6, T=1, H=224, W=224]
    assert payload.primary_tensor.shape == torch.Size([1, 6, 1, 224, 224])
    assert payload.primary_tensor.dtype == torch.float32

    # Verify documented band mapping
    meta = payload.metadata["primary"]
    assert meta["band_mapping"] == ["B02", "B03", "B04", "B05", "B06", "B07"]
    assert meta["channels_detected"] == 6

@pytest.mark.anyio
async def test_demo_svg_safe_handling_no_fabrication():
    """
    Test 4 & 5: SVG demo files are safely handled on the Demo path without
    fabricating multispectral bands.
    """
    adapter = RasterDataAdapter()
    svg_bytes = b"<svg xmlns='http://www.w3.org/2000/svg'><rect width='100' height='100'/></svg>"
    upload_file = create_mock_upload_file("demo_scene.svg", svg_bytes, "image/svg+xml")

    payload: AdaptedDataPayload = await adapter.adapt_inputs(image=upload_file)

    assert payload.is_multispectral is False
    assert payload.is_demo is True
    assert payload.primary_tensor is None
    assert isinstance(payload.primary_raw, str)
    assert "SVG" in payload.metadata["primary"]["format"]

@pytest.mark.anyio
async def test_demo_rgb_png_safe_handling():
    """
    Test 5: Standard 3-channel RGB PNG is safely handled on Demo path without fabrication.
    """
    adapter = RasterDataAdapter()
    png_bytes = create_rgb_png_bytes()
    upload_file = create_mock_upload_file("cartosat_rgb.png", png_bytes, "image/png")

    payload: AdaptedDataPayload = await adapter.adapt_inputs(image=upload_file)

    assert payload.is_multispectral is False
    assert payload.is_demo is True
    assert payload.primary_tensor is None
    assert isinstance(payload.primary_raw, Image.Image)

@pytest.mark.anyio
async def test_bitemporal_t1_t2_pairing():
    """
    Test 6: Bi-temporal mode pairs T1 and T2 multi-band tensors.
    """
    adapter = RasterDataAdapter()
    t1_bytes = create_multiband_tiff_bytes(channels=6, height=120, width=120)
    t2_bytes = create_multiband_tiff_bytes(channels=6, height=120, width=120)

    t1_file = create_mock_upload_file("t1_baseline.tif", t1_bytes)
    t2_file = create_mock_upload_file("t2_target.tif", t2_bytes)

    payload: AdaptedDataPayload = await adapter.adapt_inputs(
        image=t1_file,
        image_t2=t2_file,
        task_hint="change"
    )

    assert payload.mode == "bitemporal"
    assert payload.is_multispectral is True
    assert payload.primary_tensor.shape == torch.Size([1, 6, 1, 224, 224])
    assert payload.secondary_tensor.shape == torch.Size([1, 6, 1, 224, 224])

@pytest.mark.anyio
async def test_optical_and_sar_pairing():
    """
    Test 7: Optical 6-band and SAR 2-band rasters are properly paired.
    """
    adapter = RasterDataAdapter()
    optical_bytes = create_multiband_tiff_bytes(channels=6, height=120, width=120)
    sar_bytes = create_multiband_tiff_bytes(channels=2, height=120, width=120)

    opt_file = create_mock_upload_file("optical_s2.tif", optical_bytes)
    sar_file = create_mock_upload_file("radar_s1_vv_vh.tif", sar_bytes)

    payload: AdaptedDataPayload = await adapter.adapt_inputs(
        image=opt_file,
        sar=sar_file,
        task_hint="multimodal"
    )

    assert payload.mode == "optical_sar"
    assert payload.primary_tensor.shape == torch.Size([1, 6, 1, 224, 224])
    assert payload.secondary_tensor.shape == torch.Size([1, 2, 1, 224, 224])
    assert payload.metadata["secondary"]["band_mapping"] == ["VV", "VH"]
