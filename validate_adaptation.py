from pathlib import Path

import torch
import torch.nn.functional as F

from terratorch.registry import BACKBONE_REGISTRY
from ml.data.preprocess import preprocess_s2_patch


MODEL_NAME = "prithvi_eo_v2_100_tl"
S2_ROOT = Path("Portugal_2500/S2")

# Use patches other than the first training/demo patch.
patches = sorted(
    p.name.rsplit("_B", 1)[0]
    for p in S2_ROOT.glob("*_B02.tif")
)

patches = list(dict.fromkeys(patches))

# Small held-out check.
test_patches = patches[100:105]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = BACKBONE_REGISTRY.build(
        MODEL_NAME,
        pretrained=True,
    )
    model = model.to(device)
    model.eval()
    return model


def get_embedding(model, patch_id):
    x = preprocess_s2_patch(
        str(S2_ROOT),
        patch_id,
    ).to(device)

    with torch.no_grad():
        features = model.forward_features(x)[-1]

    # CLS token -> [1, 768]
    embedding = features[:, 0, :]

    return embedding


def main():
    print("Testing patches:", len(test_patches))
    print("Device:", device)

    # Original/base model.
    print("\nLoading original Prithvi...")
    original = load_model()

    # Adapted model with LoRA.
    # We load the base model and attach the saved adapter.
    print("Loading adapted Prithvi...")

    adapted = BACKBONE_REGISTRY.build(
        MODEL_NAME,
        pretrained=True,
    )

    from peft import PeftModel

    adapted = PeftModel.from_pretrained(
        adapted,
        "checkpoints/prithvi_lora_best",
    )

    adapted = adapted.to(device)
    adapted.eval()

    for patch_id in test_patches:
        print(f"\nPatch: {patch_id}")

        e_original = get_embedding(
            original,
            patch_id,
        )

        e_adapted = get_embedding(
            adapted,
            patch_id,
        )

        finite_original = torch.isfinite(
            e_original
        ).all().item()

        finite_adapted = torch.isfinite(
            e_adapted
        ).all().item()

        std_original = e_original.std().item()
        std_adapted = e_adapted.std().item()

        norm_original = e_original.norm().item()
        norm_adapted = e_adapted.norm().item()

        cosine = F.cosine_similarity(
            e_original,
            e_adapted,
            dim=-1,
        ).item()

        diff = (
            (e_original - e_adapted)
            .norm()
            .item()
        )

        print(
            f"finite: original={finite_original}, "
            f"adapted={finite_adapted}"
        )

        print(
            f"std: original={std_original:.6f}, "
            f"adapted={std_adapted:.6f}"
        )

        print(
            f"norm: original={norm_original:.4f}, "
            f"adapted={norm_adapted:.4f}"
        )

        print(
            f"cosine(original, adapted)={cosine:.6f}"
        )

        print(
            f"L2 difference={diff:.6f}"
        )


if __name__ == "__main__":
    main()