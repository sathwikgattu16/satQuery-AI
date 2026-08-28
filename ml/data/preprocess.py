"""
ml/data/preprocess.py
Normalizes multi-spectral (Sentinel-2) and SAR (Sentinel-1) bands for Prithvi-EO-2.0 input format.
Owner: Member 3
"""

def preprocess_bands(image_path: str, modality: str = "optical"):
    """
    Standardize band ordering, apply mean/std normalization, and prepare tensor dimensions.
    """
    print(f"[ML Data] Preprocessing {modality} image at: {image_path}")
    # Placeholder: normalization logic for Prithvi 6-band optical or 2-band SAR
    pass
