from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import rasterio
import torch
import torch.nn.functional as F


PRITHVI_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07"]

PRITHVI_MEAN = torch.tensor(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
    dtype=torch.float32,
)

PRITHVI_STD = torch.tensor(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
    dtype=torch.float32,
)


def read_band(path: Path) -> np.ndarray:
    """Read the first raster band as float32."""
    with rasterio.open(path) as src:
        data = src.read(1)

    return data.astype(np.float32)


def find_band_file(
    s2_dir: Path,
    patch_id: str,
    band: str,
) -> Path:
    path = s2_dir / f"{patch_id}_{band}.tif"

    if not path.exists():
        raise FileNotFoundError(f"Missing band: {path}")

    return path


def preprocess_s2_patch(
    s2_dir: str,
    patch_id: str,
) -> torch.Tensor:
    """
    Load BigEarthNet Sentinel-2 B02-B07.

    Returns:
        [1, 6, 1, 224, 224]
    """

    s2_path = Path(s2_dir)

    bands: List[np.ndarray] = []

    for band in PRITHVI_BANDS:
        path = find_band_file(
            s2_path,
            patch_id,
            band,
        )

        bands.append(read_band(path))

    # B02-B04 are 120x120.
    # B05-B07 are 60x60.
    target_h = 120
    target_w = 120

    tensors = []

    for image in bands:

        tensor = torch.from_numpy(image)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        if tensor.shape[-2:] != (
            target_h,
            target_w,
        ):
            tensor = F.interpolate(
                tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        tensors.append(
            tensor.squeeze(0)
        )

    # [6, 120, 120]
    stacked = torch.cat(
        tensors,
        dim=0,
    )

    # [6, 224, 224]
    stacked = F.interpolate(
        stacked.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Prithvi normalization.
    mean = PRITHVI_MEAN[:, None, None]
    std = PRITHVI_STD[:, None, None]

    stacked = (stacked - mean) / std

    # [1, 6, 1, 224, 224]
    return stacked.unsqueeze(0).unsqueeze(2)


def preprocess_s1_patch(
    s1_dir: str,
    patch_id: str,
) -> torch.Tensor:
    """
    Load BigEarthNet Sentinel-1 VH/VV.

    Returns:
        [1, 2, 1, 224, 224]

    Channels:
        0 = VH
        1 = VV
    """

    s1_path = Path(s1_dir)

    vh_path = s1_path / f"{patch_id}_VH.tif"
    vv_path = s1_path / f"{patch_id}_VV.tif"

    if not vh_path.exists():
        raise FileNotFoundError(
            f"Missing VH band: {vh_path}"
        )

    if not vv_path.exists():
        raise FileNotFoundError(
            f"Missing VV band: {vv_path}"
        )

    vh = read_band(vh_path)
    vv = read_band(vv_path)

    stacked = torch.from_numpy(
        np.stack(
            [vh, vv],
            axis=0,
        )
    )

    # [2, H, W] -> [1, 2, 1, 224, 224]
    stacked = F.interpolate(
        stacked.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    return stacked.unsqueeze(2)


if __name__ == "__main__":
    print("Preprocessing module loaded successfully.")