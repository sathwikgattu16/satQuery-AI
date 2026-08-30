import torch

from peft import LoraConfig, get_peft_model
from terratorch.registry import BACKBONE_REGISTRY

from ml.data.preprocess import preprocess_s2_patch


MODEL_NAME = "prithvi_eo_v2_100_tl"
S2_DIR = r"Portugal_2500\S2"
PATCH_ID = "S2A_MSIL2A_20170704T112111_N9999_R037_T29SND_01_52"

print("Loading Prithvi...")
model = BACKBONE_REGISTRY.build(
    MODEL_NAME,
    pretrained=True,
)

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["qkv", "proj"],
    lora_dropout=0.05,
    bias="none",
)

print("Attaching LoRA...")
model = get_peft_model(model, lora_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print("Preprocessing real patch...")
x = preprocess_s2_patch(S2_DIR, PATCH_ID).to(device)

print("Input:", tuple(x.shape))

with torch.no_grad():
    features = model.base_model.model.forward_features(x)

print("Feature shapes:")
for i, feature in enumerate(features):
    print(i, tuple(feature.shape))

model.print_trainable_parameters()
print("LoRA + real-data forward pass: SUCCESS")