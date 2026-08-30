# SatQuery AI — Multimodal Remote-Sensing Vision-Language Assistant

> **Smart India Hackathon (SIH)**  
> **Foundation Model**: IBM/NASA Prithvi-EO-2.0 (`prithvi_eo_v2_100_tl`)  
> **Architecture**: Decoupled Multimodal AI System (FastAPI Backend + React Frontend + PyTorch Specialist Registry)

---

## 🛰️ 1. Project Overview

**SatQuery AI** is an intelligent vision-language assistant tailored for earth observation (EO) and remote sensing data. It orchestrates multimodal queries across four core capabilities:

1. **Single-Scene VQA**: Visual Question Answering with spatial grounding on optical satellite imagery.
2. **Automated Captioning**: Comprehensive scene descriptions and multi-class semantic land-cover summaries.
3. **Bi-Temporal Change Detection**: Quantified spatial change maps and difference analysis between T1 and T2 observations.
4. **Optical + SAR Multimodal Fusion**: All-weather cloud-penetrating analysis combining Optical reflectance and Synthetic Aperture Radar (SAR) backscatter.

---

## 🏛️ 2. System Architecture

```
React / Vite UI (Port 5173)
         │
         ▼ (POST /api/analyze — multipart/form-data)
  FastAPI Backend (Port 8000)
         │
         ├── Stage 1: [CompatibilityChecker] (Integrity & multi-sensor validation)
         ├── Stage 2: [RasterDataAdapter] (Converts multi-band GeoTIFFs to [1, 6, 1, 224, 224] tensors)
         ├── Stage 3: [TaskClassifier] (Authoritative routing: vqa | caption | change | fusion)
         ├── Stage 4: [ModelLifecycleManager] (Singleton caching & device resolution)
         ├── Stage 5: [ModelRegistry] (Decoupled specialist dispatch)
         ├── Stage 6: [BaseSpecialist.predict()] (Execution with diagnostics)
         └── Stage 7: [ResponseSynthesizer] (Pydantic contract + visual evidence)
```

---

## 🚀 3. Quick Start & Setup Guide

### 3.1 Backend Setup (FastAPI)
```bash
# 1. Navigate to repository root
cd satQuery-AI

# 2. Install backend dependencies
pip install -r backend/requirements.txt

# 3. (Optional) Configure environment
cp .env.example .env

# 4. Start the backend server
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
* **API Health Check**: `http://localhost:8000/`
* **Interactive OpenAPI Docs**: `http://localhost:8000/docs`

### 3.2 Frontend Setup (React + Vite)
```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install node dependencies
npm install

# 3. Start development server
npm run dev
```
* **Frontend UI**: `http://localhost:5173/`

---

## ⚙️ 4. Configuration & Model Modes

Configuration is managed via `.env` (template in `.env.example`):

| Variable | Default | Description |
| :--- | :--- | :--- |
| `USE_REAL_MODELS` | `false` | Set to `true` to activate real neural model weights in `ml/models/`. Default is `false` for instant, reliable offline judging and demo mode. |
| `PRITHVI_CHECKPOINT` | `None` | Path to local `prithvi_eo_v2_100_tl.pt` foundation weights. |
| `LORA_ADAPTER_PATH` | `None` | Path to fine-tuned LoRA adapter weights. |
| `MODEL_DEVICE` | `auto` | Target compute device (`"auto"`, `"cuda:0"`, `"cpu"`). |

---

## 🧪 5. Verification & Testing

### 5.1 Automated Test Suite (34/34 Passing)
Run all backend unit, contract, routing, and lifecycle tests:
```bash
python -m pytest backend/tests/ -v
```

### 5.2 Live End-to-End Test
Verify live HTTP communication across all 4 modes against a running server:
```bash
python backend/tests/live_e2e_test.py
```

---

## 📋 6. Team Roles & Handoff Documentation

* **Member 1 (User)**: Backend API, Compatibility Validation, Task Routing, Model Lifecycle & Execution Trace.
* **Member 2**: React/Vite Frontend UI, Evidence Viewer & Execution Trace Display.
* **Member 3**: IBM/NASA Prithvi-EO-2.0 Adaptation, LoRA Fine-Tuning & Real Specialist Neural Heads.
  * 📖 **Detailed ML Handoff Specification**: See [`docs/member3_ml_handoff.md`](docs/member3_ml_handoff.md).
* **Members 4, 5, 6**: Specialist models development, domain evaluation, and SIH demonstration preparation.
