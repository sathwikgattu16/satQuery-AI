import glob

from ml.data.preprocess import preprocess_s2_patch
from ml.models.change_model import ChangeSpecialist
from ml.models.prithvi_model import PrithviBackbone


S2_DIR = r"Portugal_2500\S2"

# Two different real Portugal patches.
patches = sorted(
    {
        p.rsplit("\\", 1)[-1].rsplit("_B", 1)[0]
        for p in glob.glob(S2_DIR + r"\*_B02.tif")
    }
)

t1_id = patches[0]
t2_id = patches[1]

print("T1:", t1_id)
print("T2:", t2_id)

print("\nLoading adapted backbone...")

backbone = PrithviBackbone(
    checkpoint_path="checkpoints/prithvi_lora_best"
)

print("\nPreprocessing T1...")
t1 = preprocess_s2_patch(S2_DIR, t1_id)

print("Preprocessing T2...")
t2 = preprocess_s2_patch(S2_DIR, t2_id)

print("T1 shape:", tuple(t1.shape))
print("T2 shape:", tuple(t2.shape))

print("\nCreating ChangeSpecialist...")

change = ChangeSpecialist(
    backbone=backbone
)

print("\nRunning change analysis...")

result = change.predict(
    query="What changed between T1 and T2?",
    image_primary=t1,
    image_secondary=t2,
)

print("\nRESULT:")
print(result)

print("\n=== CHANGE SUCCESS ===")