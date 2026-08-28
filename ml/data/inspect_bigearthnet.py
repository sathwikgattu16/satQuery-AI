"""
ml/data/inspect_bigearthnet.py
Inspect BigEarthNet metadata parquet / CSV splits without downloading full 110+ GiB archive.
Owner: Member 3
"""

def inspect_metadata(metadata_path: str):
    """
    Inspect BigEarthNet metadata schema, patch IDs, split distributions, and Sentinel-1/2 pair availability.
    """
    print(f"[ML Data] Inspecting BigEarthNet metadata at: {metadata_path}")
    # Placeholder: load metadata parquet / json and print summary statistics
    pass

if __name__ == "__main__":
    print("Run this script to inspect BigEarthNet metadata structure.")
