"""
ml/adaptation/prepare_dataset.py
Formats extracted subset into PyTorch Dataset / DataLoader for LoRA parameter-efficient adaptation.
Owner: Member 3
"""

class RemoteSensingAdaptationDataset:
    """PyTorch Dataset placeholder for adapted Prithvi training."""
    def __init__(self, data_manifest: str):
        self.data_manifest = data_manifest

    def __len__(self):
        return 50

    def __getitem__(self, idx: int):
        # Placeholder tensor return
        return {"input_bands": None, "labels": None}
