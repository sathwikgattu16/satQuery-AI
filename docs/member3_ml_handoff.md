# SatQuery AI — Member 3 ML Integration & Handoff Specification

> **Target Audience**: Member 3 (Machine Learning, Adaptation, & Specialist Models Owner)  
> **Author**: Member 1 (Backend & Orchestration Owner) / Principal Software Architect  
> **Status**: Authoritative Backend $\leftrightarrow$ ML Contract  
> **Last Updated**: Hackathon Day 2 / Checkpoint 9

---

## 1. System Architecture & Context

The backend infrastructure and agentic orchestration layer for SatQuery AI are **fully implemented and verified against 34 automated unit/integration tests and live HTTP runs**.

The backend operates under an **Inversion of Control** pattern:
```
React Frontend (Port 5173)
           │
           ▼ (POST /api/analyze - multipart/form-data)
     FastAPI Backend (Port 8000)
           │
           ▼
  [CompatibilityChecker]
           │
           ▼
  [RasterDataAdapter] ──► Converts GeoTIFFs to [1, 6, 1, 224, 224] tensors
           │
           ▼
    [TaskClassifier] ──► Authoritatively routes to: vqa, caption, change, fusion
           │
           ▼
    [ModelRegistry]
           │
           ▼
[BaseSpecialist.predict()] ◄─── YOU PLUG IN HERE (ml/models/*.py)
           │
           ▼
  [Shared Prithvi Backbone] (ml/models/prithvi_model.py)
```

> [!IMPORTANT]
> When you replace the placeholder specialists in `ml/models/*.py` with your real neural weights and inference heads, **zero lines of code in the frontend, FastAPI routers, controller, or data adapters need to change.**

---

## 2. Backend $\rightarrow$ ML Data Flow & Input Contracts

The backend [`RasterDataAdapter`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/backend/agent/data_adapter.py) pre-validates, resamples, and shapes all incoming geospatial files before invoking your specialists.

### 2.1 Multi-Spectral Optical Raster Contract (Sentinel-2)
When a 6-band optical GeoTIFF is uploaded:
* **Tensor Shape**: `torch.Size([B=1, C=6, T=1, H=224, W=224])`
* **Data Type (`dtype`)**: `torch.float32`
* **Band Ordering**: `["B02", "B03", "B04", "B05", "B06", "B07"]` (Blue, Green, Red, RedEdge 1, RedEdge 2, RedEdge 3).
* **Spatial Resolution**: Bilinearly interpolated to `224 × 224` pixels.
* **Temporal Dimension**: `T=1` frame.

### 2.2 SAR Radar Raster Contract (Sentinel-1 / RISAT-1A)
When a 2-band SAR radar GeoTIFF is uploaded in the `sar` slot:
* **Tensor Shape**: `torch.Size([B=1, C=2, T=1, H=224, W=224])`
* **Data Type (`dtype`)**: `torch.float32`
* **Polarization Channels**: `["VV", "VH"]` backscatter.
* **Spatial Resolution**: `224 × 224` pixels.

### 2.3 Synthetic / Demo Safe Path (SVG / 3-Channel RGB)
For quick offline UI testing, the frontend generates synthetic `.svg` and `.png` files.
* **Backend Behavior**: Passed to specialists as decoded `PIL.Image` or raw SVG string with `is_multispectral=False` and `is_demo=True`.
* **Guaranteed Integrity**: The backend will **never fabricate** the 6 multispectral bands from standard RGB.

---

## 3. Prithvi-EO-2.0 Shared Backbone Contract

Location: [`ml/models/prithvi_model.py`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/ml/models/prithvi_model.py)

```python
class PrithviBackbone:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.is_adapted = checkpoint_path is not None
        # TODO (Member 3): Instantiate prithvi_eo_v2_100_tl ViT architecture
        # TODO (Member 3): Load weights from self.checkpoint_path onto target device

    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Input:
            image_tensor: torch.Tensor of shape [B, 6, 1, 224, 224] (torch.float32)
        Output:
            Dense spatial token embeddings: torch.Tensor of shape [B, 197, 768]
            (REQUIRES MEMBER 3 CONFIRMATION: Confirm exact ViT embedding dimension)
        """
        pass
```

---

## 4. Specialist Model Contracts & Required Signatures

All specialist models in `ml/models/` inherit from `BaseSpecialist` ([`backend/models/base.py`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/backend/models/base.py)).

### 4.1 Single-Image VQA Specialist
* **File**: [`ml/models/vqa_model.py`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/ml/models/vqa_model.py)
* **Method Signature**:
  ```python
  def predict(self, query: str, image_primary: torch.Tensor, image_secondary: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  ```
* **Inputs**:
  * `query`: Natural language question (e.g. `"What is the dominant land cover?"`).
  * `image_primary`: Optical tensor `[1, 6, 1, 224, 224]` (or `PIL.Image` in demo mode).
* **Expected Return Dictionary**:
  ```python
  {
      "answer": "Land cover classification: 52% Dense Forest, 34% Agriculture, 14% Water body.",
      "confidence": 0.94,  # float between 0.0 and 1.0
      "evidence": {
          "type": "overlay",  # "overlay" | "heatmap" | "bbox"
          "title": "VQA Spatial Attention Map",
          "description": "Spatial attention anchors for detected target features.",
          "metrics": {
              "Ground Sample Distance": "10m",
              "Confidence Score": "94.2%"
          }
      },
      "execution_detail": "Executed VQASpecialist using adapted Prithvi-EO-2.0 ViT representation."
  }
  ```

---

### 4.2 Single-Image Captioning Specialist
* **File**: [`ml/models/captioning_model.py`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/ml/models/captioning_model.py)
* **Method Signature**:
  ```python
  def predict(self, query: str, image_primary: torch.Tensor, image_secondary: Optional[Any] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  ```
* **Inputs**:
  * `query`: Optional caption focus hint (e.g. `"Focus on transport infrastructure"`).
  * `image_primary`: Optical tensor `[1, 6, 1, 224, 224]`.
* **Expected Return Dictionary**:
  ```python
  {
      "answer": "High-resolution multispectral scene showing an active port with shipping docks and coastal vegetation.",
      "confidence": 0.91,
      "evidence": {
          "type": "overlay",
          "title": "Land-Cover Segmentation Mask",
          "description": "Automated semantic segmentation overlay.",
          "metrics": {
              "Dominant Class": "Coastal Harbor",
              "Cloud Cover": "< 2%"
          }
      },
      "execution_detail": "Executed CaptioningSpecialist with Prithvi spatial tokens."
  }
  ```

---

### 4.3 Bi-Temporal Change Detection Specialist
* **File**: [`ml/models/change_model.py`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/ml/models/change_model.py)
* **Method Signature**:
  ```python
  def predict(self, query: str, image_primary: torch.Tensor, image_secondary: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  ```
* **Inputs**:
  * `query`: Change query (e.g. `"What urban expansion occurred between T1 and T2?"`).
  * `image_primary`: **T1 Baseline** optical tensor `[1, 6, 1, 224, 224]`.
  * `image_secondary`: **T2 Observation** optical tensor `[1, 6, 1, 224, 224]`.
* **Expected Return Dictionary**:
  ```python
  {
      "answer": "Detected 14.2 hectares of new built-up expansion between T1 baseline and T2 observation.",
      "confidence": 0.92,
      "evidence": {
          "type": "change_mask",
          "title": "Bi-Temporal Change Intensity Map",
          "description": "Red indicates built-up expansion; Blue indicates water inundation.",
          "metrics": {
              "Changed Area": "14.2 ha",
              "NDVI Shift": "-0.24 avg",
              "Co-registration Error": "< 0.3 px"
          }
      },
      "execution_detail": "Executed ChangeSpecialist via dual-temporal Prithvi embedding comparison."
  }
  ```

---

### 4.4 Optical + SAR Multimodal Fusion Specialist
* **File**: [`ml/models/fusion_model.py`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/ml/models/fusion_model.py)
* **Method Signature**:
  ```python
  def predict(self, query: str, image_primary: torch.Tensor, image_secondary: torch.Tensor, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  ```
* **Inputs**:
  * `query`: Multimodal query (e.g. `"Identify maritime vessels beneath clouds using SAR."`).
  * `image_primary`: Optical tensor `[1, 6, 1, 224, 224]`.
  * `image_secondary`: SAR radar tensor `[1, 2, 1, 224, 224]` (VV/VH backscatter).
* **Expected Return Dictionary**:
  ```python
  {
      "answer": "Optical-SAR cross-sensor analysis detected 6 maritime vessels penetrating cirrus cloud layers.",
      "confidence": 0.95,
      "evidence": {
          "type": "heatmap",
          "title": "Optical + SAR Polarimetric Fusion Matrix",
          "description": "SAR VH backscatter overlaid on Optical L2A reflectance.",
          "metrics": {
              "Cloud Penetration": "98.4%",
              "Polarization Mode": "Dual Pol (VV + VH)",
              "Resolution": "10m GSD"
          }
      },
      "execution_detail": "Executed FusionSpecialist using multi-sensor joint Prithvi embeddings."
  }
  ```

---

## 5. Scientific Preprocessing Responsibilities

* **Backend Responsibility**: Byte decoding, TIFF channel splitting, bilinear resizing to `224 × 224`, and structuring into PyTorch `torch.float32` tensors.
* **Member 3 ML Responsibility**:
  * Implement exact band normalization constants in [`ml/data/preprocess.py`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/ml/data/preprocess.py) using Sentinel-2 L2A top-of-atmosphere/surface reflectance mean & standard deviation values:
    $$\text{normalized} = \frac{\text{tensor} - \mu}{\sigma}$$
  * (REQUIRES MEMBER 3 CONFIRMATION: Provide channel mean $\mu$ and std $\sigma$ constants for `[B02, B03, B04, B05, B06, B07]`).

---

## 6. Dataset & Adaptation Responsibilities (`Portugal_2500`)

* **Dataset Path**: `Portugal_2500/` containing paired Sentinel-1 (VV, VH) and Sentinel-2 multispectral patches.
* **Member 3 ML Responsibility**:
  * Complete dataset loader in [`ml/adaptation/prepare_dataset.py`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/ml/adaptation/prepare_dataset.py).
  * Implement parameter-efficient LoRA fine-tuning loop in [`ml/adaptation/train_lora.py`](file:///C:/Users/Lenovo/Desktop/satQuery-AI/ml/adaptation/train_lora.py).
  * Save adapter weights to a local directory (e.g., `checkpoints/prithvi_lora_last.pt`).

---

## 7. How to Activate Real Model Inference

When your real models and checkpoints are ready, activate real inference in the backend simply by setting environment variables:

```bash
# Set Real Model Flag
export USE_REAL_MODELS=true

# Point to Local Checkpoint Files
export PRITHVI_CHECKPOINT="checkpoints/prithvi_eo_v2_100_tl.pt"
export LORA_ADAPTER_PATH="checkpoints/prithvi_lora_last.pt"

# Compute Device (auto detects CUDA, falls back to CPU)
export MODEL_DEVICE="auto"
```

The backend `ModelLifecycleManager` will automatically bind your real `ml/models/*.py` instances into the active agent registry on startup.

---

## 8. Member 3 Completion Checklist

- [ ] Provide or link local `prithvi_eo_v2_100_tl.pt` foundation weights.
- [ ] Implement ViT forward pass in `PrithviBackbone.extract_features()`.
- [ ] Implement `VQASpecialist.predict()` in `ml/models/vqa_model.py`.
- [ ] Implement `CaptioningSpecialist.predict()` in `ml/models/captioning_model.py`.
- [ ] Implement `ChangeSpecialist.predict()` in `ml/models/change_model.py`.
- [ ] Implement `FusionSpecialist.predict()` in `ml/models/fusion_model.py`.
- [ ] Implement LoRA fine-tuning script in `ml/adaptation/train_lora.py`.
- [ ] Run evaluation script `ml/adaptation/evaluate_adaptation.py`.
