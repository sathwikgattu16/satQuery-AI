"""
ml/data/select_subset.py
Select small, balanced 50-100 sample subset of paired Sentinel-1 and Sentinel-2 patches.
Owner: Member 3
"""

def select_adaptation_subset(metadata_file: str, sample_size: int = 50, output_file: str = "data_subset/samples.json"):
    """
    Filter metadata for high-quality S1/S2 pairs across representative land-cover classes.
    """
    print(f"[ML Data] Selecting {sample_size} sample pairs from {metadata_file} -> {output_file}")
    # Placeholder: select stratified subset
    pass

if __name__ == "__main__":
    select_adaptation_subset("metadata.parquet", sample_size=50)
