from pathlib import Path

import pandas as pd


METADATA_PATH = Path("data_metadata/metadata.parquet")
OUTPUT_PATH = Path("data_metadata/portugal_subset.csv")

TRAIN_SIZE = 800
VALIDATION_SIZE = 200
RANDOM_SEED = 42


def load_metadata() -> pd.DataFrame:
    """Load BigEarthNet metadata and keep usable Portugal patches."""
    df = pd.read_parquet(METADATA_PATH)

    portugal = df[df["country"] == "Portugal"].copy()

    portugal = portugal[
        (portugal["contains_seasonal_snow"] != True)
        & (portugal["contains_cloud_or_shadow"] != True)
    ].copy()

    return portugal


def select_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Select a reproducible train/validation subset."""
    train = (
        df[df["split"] == "train"]
        .sample(n=TRAIN_SIZE, random_state=RANDOM_SEED)
    )

    validation = (
        df[df["split"] == "validation"]
        .sample(n=VALIDATION_SIZE, random_state=RANDOM_SEED)
    )

    subset = pd.concat([train, validation], ignore_index=True)

    return subset[
        [
            "patch_id",
            "labels",
            "split",
            "country",
            "s1_name",
            "s2v1_name",
        ]
    ]


def main() -> None:
    df = load_metadata()
    subset = select_subset(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(OUTPUT_PATH, index=False)

    print(f"Selected patches: {len(subset)}")
    print("\nSplit:")
    print(subset["split"].value_counts().to_string())
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()