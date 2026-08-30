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


def find_band_file(s2_dir: Path, patch_id: str, band: str) -> Path:
    path = s2_dir / f"{patch_id}_{band}.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing band: {path}")
    return path


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1)

    return data.astype(np.float32)


def preprocess_s2_patch(s2_dir: str, patch_id: str) -> torch.Tensor:
    """
    Load B02-B07 from one BigEarthNet S2 patch and return:

        [1, 6, 1, 224, 224]

    suitable for the instantiated Prithvi-EO-2.0 100M-TL backbone.
    """
    s2_path = Path(s2_dir)

    bands: List[np.ndarray] = []

    for band in PRITHVI_BANDS:
        path = find_band_file(s2_path, patch_id, band)
        bands.append(read_band(path))

    # Bring all bands to the same spatial resolution.
    # B02-B04 are 120x120; B05-B07 are 60x60.
    target_h, target_w = 120, 120

    tensors = []

    for image in bands:
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        if tensor.shape[-2:] != (target_h, target_w):
            tensor = F.interpolate(
                tensor,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        tensors.append(tensor.squeeze(0))

    # [6, 120, 120]
    stacked = torch.cat(tensors, dim=0)

    # Resize to Prithvi's required spatial resolution.
    stacked = F.interpolate(
        stacked.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # Normalize using the official Prithvi-EO-2.0 statistics.
    mean = PRITHVI_MEAN[:, None, None]
    std = PRITHVI_STD[:, None, None]

    stacked = (stacked - mean) / std

    # [1, 6, 1, 224, 224]
    return stacked.unsqueeze(0).unsqueeze(2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python -m ml.data.preprocess "
            "<S2_DIRECTORY> <PATCH_ID>"
        )
        raise SystemExit(1)

    s2_dir = sys.argv[1]
    patch_id = sys.argv[2]

    x = preprocess_s2_patch(s2_dir, patch_id)

    print("Preprocessed tensor:")
    print("shape:", tuple(x.shape))
    print("dtype:", x.dtype)
    print("min:", float(x.min()))
    print("max:", float(x.max()))
    print("mean:", float(x.mean()))
    print("std:", float(x.std()))