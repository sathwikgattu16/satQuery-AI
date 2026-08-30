from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from peft import LoraConfig, get_peft_model
from terratorch.registry import BACKBONE_REGISTRY

from ml.data.preprocess import preprocess_s2_patch


MODEL_NAME = "prithvi_eo_v2_100_tl"
S2_ROOT = Path("Portugal_2500/S2")

TOTAL_PATCHES = 2500
EPOCHS = 3

BATCH_SIZE = 2
GRAD_ACCUM = 8

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def get_patch_ids() -> list[str]:
    patch_ids = sorted(
        {
            p.name.rsplit("_B", 1)[0]
            for p in S2_ROOT.glob("*_B02.tif")
        }
    )

    if len(patch_ids) < TOTAL_PATCHES:
        raise RuntimeError(
            f"Expected at least {TOTAL_PATCHES} patches, "
            f"found {len(patch_ids)}"
        )

    rng = random.Random(SEED)
    rng.shuffle(patch_ids)

    return patch_ids[:TOTAL_PATCHES]


class PortugalDataset(Dataset):
    def __init__(self, patch_ids: list[str]) -> None:
        self.patch_ids = patch_ids

    def __len__(self) -> int:
        return len(self.patch_ids)

    @staticmethod
    def augment(x: torch.Tensor) -> torch.Tensor:
        # x = [1, 6, 1, 224, 224]
        x = x.clone()

        # Mild intensity scaling while preserving spectral relationships.
        scale = 1.0 + random.uniform(-0.10, 0.10)
        x = x * scale

        if random.random() < 0.5:
            x = torch.flip(x, dims=[-1])

        if random.random() < 0.5:
            x = torch.flip(x, dims=[-2])

        return x

    def __getitem__(self, idx: int):
        patch_id = self.patch_ids[idx]

        x = preprocess_s2_patch(
            str(S2_ROOT),
            patch_id,
        )

        v1 = self.augment(x).squeeze(0)
        v2 = self.augment(x).squeeze(0)

        return v1, v2


class MLP(nn.Module):
    def __init__(self, dim: int = 768, hidden: int = 256) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def negative_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1)
    target = F.normalize(target.detach(), dim=-1)

    return -(prediction * target).sum(dim=-1).mean()


def main() -> None:
    print("======================================")
    print("Prithvi Portugal Adaptation")
    print("======================================")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    patch_ids = get_patch_ids()
    print("Patches:", len(patch_ids))

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
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

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    model.train()

    # Optional checkpointing if supported.
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing: enabled")
        except Exception as exc:
            print("Gradient checkpointing unavailable:", exc)

    # ---------------------------------------------------------
    # Heads
    # ---------------------------------------------------------
    projector = MLP().to(device)
    predictor = MLP().to(device)

    projector.train()
    predictor.train()

    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    dataset = PortugalDataset(patch_ids)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    lora_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    head_params = (
        list(projector.parameters())
        + list(predictor.parameters())
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": 2e-4,
            },
            {
                "params": head_params,
                "lr": 3e-4,
            },
        ],
        weight_decay=0.05,
    )

    # ---------------------------------------------------------
    # Mixed precision
    # ---------------------------------------------------------
    use_amp = torch.cuda.is_available()

    scaler = torch.cuda.amp.GradScaler(
        enabled=use_amp
    )

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    best_loss = float("inf")

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    for epoch in range(EPOCHS):

        epoch_start = time.time()
        running_loss = 0.0

        print()
        print(
            f"========== EPOCH {epoch + 1}/{EPOCHS} =========="
        )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for batch_idx, (view1, view2) in enumerate(loader):

            view1 = view1.to(
                device,
                non_blocking=True,
            )

            view2 = view2.to(
                device,
                non_blocking=True,
            )

            with torch.cuda.amp.autocast(
                dtype=torch.float16,
                enabled=use_amp,
            ):

                f1 = model.base_model.model.forward_features(
                    view1
                )[-1]

                f2 = model.base_model.model.forward_features(
                    view2
                )[-1]

                cls1 = f1[:, 0, :]
                cls2 = f2[:, 0, :]

                z1 = projector(cls1)
                z2 = projector(cls2)

                p1 = predictor(z1)
                p2 = predictor(z2)

                loss = 0.5 * (
                    negative_cosine(p1, z2)
                    + negative_cosine(p2, z1)
                )

                scaled_loss = loss / GRAD_ACCUM

            scaler.scale(scaled_loss).backward()

            if (
                (batch_idx + 1) % GRAD_ACCUM == 0
                or (batch_idx + 1) == len(loader)
            ):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

            running_loss += loss.item()

            if (batch_idx + 1) % 25 == 0:
                elapsed = time.time() - epoch_start

                if torch.cuda.is_available():
                    peak_vram = (
                        torch.cuda.max_memory_allocated()
                        / (1024 ** 3)
                    )
                else:
                    peak_vram = 0.0

                print(
                    f"epoch={epoch + 1} "
                    f"batch={batch_idx + 1}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"time={elapsed:.1f}s "
                    f"peak_vram={peak_vram:.2f}GB"
                )

        avg_loss = running_loss / len(loader)

        print(
            f"Epoch {epoch + 1} complete | "
            f"avg_loss={avg_loss:.4f}"
        )

        # -----------------------------------------------------
        # Save epoch checkpoint
        # -----------------------------------------------------
        epoch_dir = (
            checkpoint_dir
            / f"prithvi_lora_epoch_{epoch + 1}"
        )

        model.save_pretrained(epoch_dir)

        torch.save(
            {
                "projector": projector.state_dict(),
                "predictor": predictor.state_dict(),
                "epoch": epoch + 1,
                "avg_loss": avg_loss,
            },
            checkpoint_dir
            / f"simsiam_heads_epoch_{epoch + 1}.pt",
        )

        if avg_loss < best_loss:
            best_loss = avg_loss

            model.save_pretrained(
                checkpoint_dir / "prithvi_lora_best"
            )

            torch.save(
                {
                    "projector": projector.state_dict(),
                    "predictor": predictor.state_dict(),
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                },
                checkpoint_dir / "simsiam_heads_best.pt",
            )

            print("Best checkpoint updated.")

    print()
    print("======================================")
    print("FULL ADAPTATION COMPLETE")
    print("======================================")
    print("Patches:", len(patch_ids))
    print("Epochs:", EPOCHS)
    print("Best loss:", best_loss)
    print(
        "Best adapter:",
        checkpoint_dir / "prithvi_lora_best",
    )


if __name__ == "__main__":
    main()