import torch

from ml.data.preprocess import (
    preprocess_s1_patch,
    preprocess_s2_patch,
)
from ml.models.fusion_model import FusionSpecialist
from ml.models.prithvi_model import PrithviBackbone


S1_DIR = r"Portugal_2500\S1"
S2_DIR = r"Portugal_2500\S2"

S1_PATCH = (
    "S1A_IW_GRDH_1SDV_20170706T064235_29SND_10_5"
)

S2_PATCH = (
    "S2A_MSIL2A_20170704T112111_N9999_R037_T29SND_01_52"
)

print("Loading adapted shared backbone...")

backbone = PrithviBackbone(
    checkpoint_path="checkpoints/prithvi_lora_best"
)

print("Preprocessing optical...")
optical = preprocess_s2_patch(
    S2_DIR,
    S2_PATCH,
)

print("Preprocessing SAR...")
sar = preprocess_s1_patch(
    S1_DIR,
    S1_PATCH,
)

print("Optical:", tuple(optical.shape))
print("SAR:", tuple(sar.shape))

print("\nCreating FusionSpecialist...")

fusion = FusionSpecialist(
    backbone=backbone
)

print("Running fusion...")

result = fusion.predict(
    query="What does the combined optical and SAR imagery show?",
    image_primary=optical,
    image_secondary=sar,
)

print("\nRESULT:")
print(result)

print("\n=== FUSION SUCCESS ===")