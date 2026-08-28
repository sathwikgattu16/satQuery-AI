"""
ml/data/extract_patches.py
Extract multispectral and SAR patches for the selected subset.
Owner: Member 3
"""

def extract_patches(subset_manifest: str, output_dir: str = "data_subset/extracted"):
    """
    Extract GeoTIFF band arrays from raw archive for selected subset IDs.
    """
    print(f"[ML Data] Extracting patches from {subset_manifest} into {output_dir}")
    # Placeholder: crop and format 120x120 or 224x224 patches
    pass

if __name__ == "__main__":
    extract_patches("data_subset/samples.json")
