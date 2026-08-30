import torch

from ml.models.prithvi_model import PrithviBackbone
from ml.data.preprocess import preprocess_s2_patch


PATCH_ID = (
    "S2A_MSIL2A_20170704T112111_N9999_R037_T29SND_01_52"
)

S2_DIR = r"Portugal_2500\S2"

print("=== Creating adapted shared backbone ===")

backbone = PrithviBackbone(
    checkpoint_path="checkpoints/prithvi_lora_best"
)

print("\n=== Preprocessing image ===")

x = preprocess_s2_patch(
    S2_DIR,
    PATCH_ID,
)

print("Input:", tuple(x.shape))

print("\n=== Extracting shared features ===")

features = backbone.extract_features(x)

print("Features:", tuple(features.shape))

cls = backbone.extract_cls_features(x)

print("CLS:", tuple(cls.shape))

print("\n=== SUCCESS ===")