"""
backend/agent/data_adapter.py
Backend to ML Data Adapter.
Safely ingests FastAPI UploadFiles, extracts multi-band raster data,
and prepares [B, 6, 1, 224, 224] multispectral representations for Prithvi-EO-2.0,
while maintaining a non-fabricating safe path for synthetic demo data.

Owner: Member 1
"""

import io
import os
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from PIL import Image
import torch
import tifffile
from fastapi import UploadFile
from ml.data.preprocess import PRITHVI_MEAN, PRITHVI_STD

class AdaptedDataPayload:
    """
    Standardized payload passed from the backend adapter to downstream specialists.
    """
    def __init__(
        self,
        mode: str,
        is_multispectral: bool,
        is_demo: bool,
        primary_tensor: Optional[torch.Tensor] = None,
        secondary_tensor: Optional[torch.Tensor] = None,
        primary_raw: Optional[Union[Image.Image, bytes, str]] = None,
        secondary_raw: Optional[Union[Image.Image, bytes, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.mode = mode
        self.is_multispectral = is_multispectral
        self.is_demo = is_demo
        self.primary_tensor = primary_tensor
        self.secondary_tensor = secondary_tensor
        self.primary_raw = primary_raw
        self.secondary_raw = secondary_raw
        self.metadata = metadata or {}

class RasterDataAdapter:
    """
    Data boundary adapter between FastAPI multipart file streams and the ML layer.
    """

    TARGET_SIZE: Tuple[int, int] = (224, 224)
    OPTICAL_BAND_NAMES: List[str] = ["B02", "B03", "B04", "B05", "B06", "B07"]
    SAR_BAND_NAMES: List[str] = ["VV", "VH"]

    def _resize_tensor_spatial(self, tensor: torch.Tensor, target_h: int = 224, target_w: int = 224) -> torch.Tensor:
        """
        Bilinearly resample [C, H, W] tensor to [C, target_h, target_w].
        """
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # [1, H, W]
        
        # PyTorch interpolate expects [B, C, H, W]
        unsqueezed = tensor.unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            unsqueezed,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        )
        return resized.squeeze(0)  # [C, target_h, target_w]

    async def _decode_file(self, file: UploadFile) -> Tuple[Optional[torch.Tensor], Any, Dict[str, Any]]:
        """
        Reads and inspects an uploaded file.
        Returns:
            (tensor_representation, raw_representation, metadata_dict)
        """
        filename = file.filename or "unknown"
        ext = os.path.splitext(filename)[1].lower()
        content = await file.read()
        await file.seek(0)

        meta: Dict[str, Any] = {
            "filename": filename,
            "extension": ext,
            "size_bytes": len(content),
        }

        # 1. Handle Synthetic / Demo SVG files (Strictly Demo Path, No Band Fabrication)
        if ext == ".svg" or content.startswith(b"<svg") or b"<svg" in content[:200]:
            meta["is_multispectral"] = False
            meta["is_demo"] = True
            meta["format"] = "SVG"
            meta["note"] = "Synthetic SVG demo asset. Preserved without multispectral fabrication."
            return None, content.decode("utf-8", errors="ignore"), meta

        # 2. Handle Multi-band GeoTIFF / TIFF files (Path A: Real Multi-spectral Raster)
        if ext in (".tif", ".tiff", ".geotiff"):
            try:
                with io.BytesIO(content) as bio:
                    arr = tifffile.imread(bio)
                
                # Check array shape
                if arr.ndim == 2:
                    channels = 1
                    tensor = torch.from_numpy(arr).unsqueeze(0)
                elif arr.ndim == 3:
                    if arr.shape[0] in (2, 3, 4, 6, 12, 13):
                        # Shape is [C, H, W]
                        channels = arr.shape[0]
                        tensor = torch.from_numpy(arr)
                    elif arr.shape[2] in (2, 3, 4, 6, 12, 13):
                        # Shape is [H, W, C] -> permute to [C, H, W]
                        channels = arr.shape[2]
                        tensor = torch.from_numpy(arr).permute(2, 0, 1)
                    else:
                        channels = arr.shape[0]
                        tensor = torch.from_numpy(arr)
                else:
                    raise ValueError(f"Unsupported TIFF array dimension: {arr.ndim}")

                meta["original_shape"] = list(tensor.shape)
                meta["channels_detected"] = channels

                # If exactly 6 channels (Sentinel-2 B02-B07 Prithvi representation)
                if channels == 6:
                    meta["band_mapping"] = self.OPTICAL_BAND_NAMES
                    meta["is_multispectral"] = True
                    meta["is_demo"] = False
                    
                    # Resize to [6, 224, 224]
                    resized = self._resize_tensor_spatial(tensor, 224, 224)

                    # Standardize with Prithvi-EO-2.0 optical constants if raw DN
                    # Avoid double normalization if caller already passed standardized data
                    if resized.abs().mean() > 10.0 or resized.max() > 50.0:
                        mean = PRITHVI_MEAN.view(6, 1, 1).to(resized.device)
                        std = PRITHVI_STD.view(6, 1, 1).to(resized.device)
                        resized = (resized - mean) / std

                    # Reshape to Prithvi contract: [B=1, C=6, T=1, H=224, W=224]
                    prithvi_tensor = resized.unsqueeze(0).unsqueeze(2).float()
                    meta["formatted_prithvi_shape"] = list(prithvi_tensor.shape)
                    return prithvi_tensor, arr, meta

                # If 2 channels (SAR VV/VH representation)
                elif channels == 2:
                    meta["band_mapping"] = self.SAR_BAND_NAMES
                    meta["is_multispectral"] = False
                    meta["is_sar"] = True
                    meta["is_demo"] = False
                    
                    # Resize to [2, 224, 224] and format as [B=1, C=2, T=1, H=224, W=224]
                    resized = self._resize_tensor_spatial(tensor, 224, 224)
                    sar_tensor = resized.unsqueeze(0).unsqueeze(2).float()
                    meta["formatted_sar_shape"] = list(sar_tensor.shape)
                    return sar_tensor, arr, meta

                else:
                    meta["is_multispectral"] = False
                    meta["is_demo"] = True
                    meta["note"] = f"TIFF with {channels} channel(s). Preserved on demo path without multispectral fabrication."
                    return None, arr, meta

            except Exception:
                # If tifffile read fails on arbitrary mock bytes, fallback gracefully to raw demo bytes
                meta["is_multispectral"] = False
                meta["is_demo"] = True
                meta["note"] = "Raw binary raster content preserved on demo path."
                return None, content, meta

        # 3. Handle Standard RGB / PNG / JPEG Images (Path B: Demo Path)
        try:
            with io.BytesIO(content) as bio:
                pil_img = Image.open(bio).convert("RGB")
            
            meta["is_multispectral"] = False
            meta["is_demo"] = True
            meta["format"] = pil_img.format or ext.upper().lstrip(".")
            meta["original_size"] = pil_img.size
            meta["note"] = "Standard 3-channel RGB image. Preserved on Demo Path without fabricating 6 bands."
            
            return None, pil_img, meta

        except Exception:
            # Safe demo fallback for test dummy bytes
            meta["is_multispectral"] = False
            meta["is_demo"] = True
            meta["note"] = "Raw image payload preserved on demo path."
            return None, content, meta

    async def adapt_inputs(
        self,
        image: UploadFile,
        image_t2: Optional[UploadFile] = None,
        sar: Optional[UploadFile] = None,
        task_hint: Optional[str] = None
    ) -> AdaptedDataPayload:
        """
        Main entrypoint: parses uploaded files into a validated, ML-ready AdaptedDataPayload.
        """
        # Primary image decoding
        p_tensor, p_raw, p_meta = await self._decode_file(image)
        
        has_t2 = bool(image_t2 and image_t2.filename)
        has_sar = bool(sar and sar.filename)

        s_tensor, s_raw, s_meta = None, None, {}
        if has_t2 and image_t2:
            s_tensor, s_raw, s_meta = await self._decode_file(image_t2)
            mode = "bitemporal"
        elif has_sar and sar:
            s_tensor, s_raw, s_meta = await self._decode_file(sar)
            mode = "optical_sar"
        else:
            mode = "single"

        # Determine overall payload characteristics
        is_p_multi = bool(p_meta.get("is_multispectral", False))
        is_s_valid = (s_tensor is None) or bool(s_meta.get("is_multispectral", False) or s_meta.get("is_sar", False))
        is_multi = is_p_multi and is_s_valid
        
        is_demo = bool(p_meta.get("is_demo", False) or s_meta.get("is_demo", False))

        combined_meta = {
            "mode": mode,
            "primary": p_meta,
            "secondary": s_meta if (has_t2 or has_sar) else None,
            "task_hint": task_hint,
        }

        return AdaptedDataPayload(
            mode=mode,
            is_multispectral=is_multi,
            is_demo=is_demo,
            primary_tensor=p_tensor,
            secondary_tensor=s_tensor,
            primary_raw=p_raw,
            secondary_raw=s_raw,
            metadata=combined_meta
        )
