# SATQUERY AI

**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

> **Current Status**: Initial Development Skeleton  
> This repository currently contains the initial architectural skeleton and modular interfaces for 6 teammates to clone and develop independently.

---

## 📌 Project Overview & Purpose

**SATQUERY AI** is an intelligent vision-language assistant tailored for Earth Observation (EO) and remote sensing data. Built for the ISRO Hackathon, the system enables researchers, disaster management teams, and urban planners to query multi-sensor remote sensing imagery using natural language.

### Core Capabilities
1. **Single-Image Vision Question Answering (VQA)** on optical, multispectral, or SAR imagery.
2. **Single-Image Captioning / Text-Guided Grounding**.
3. **Bi-temporal Change Analysis** (T1 vs. T2 change description and verification).
4. **Co-registered Optical + SAR Joint Analysis** (fusion for all-weather intelligence).
5. **Domain Adaptation**: Fine-tuning / adaptation on remote-sensing datasets (BigEarthNet) with **Prithvi-EO-2.0** as the shared adapted backbone.
6. **Agentic Orchestration**: Input validation $\rightarrow$ task classification $\rightarrow$ specialist selection $\rightarrow$ visual evidence generation $\rightarrow$ auditable execution trace.

---

## 🏗️ High-Level System Architecture

```
User Query + Remote Sensing Imagery (Single / Optical+SAR / Bi-temporal)
                                  │
                                  ▼
                     React + Vite Frontend (UI)
                                  │
                                  ▼ (POST /api/query)
                         FastAPI Backend
                                  │
                                  ▼
                   SatQuery Agent / Controller
          ├── Input Compatibility Checker (Modality & Format)
          ├── Task Classifier (Intent & Routing)
          └── Specialist Model Registry
                                  │
                                  ▼
           Specialist Dispatch (Shared Prithvi Backbone)
          ├── Single-Image VQA Specialist
          ├── Captioning / Grounding Specialist
          ├── Bi-temporal Change Specialist
          └── Optical-SAR Fusion Specialist
                                  │
                                  ▼
        Response Integration (Answer + Confidence + Evidence + Trace)
                                  │
                                  ▼
                     Interactive Results in UI
```

---

## 📁 Repository Structure

```
satQuery-AI/
├── backend/                  # FastAPI service, agent controller, task classifier, registry
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   ├── schemas/              # Typed Pydantic request/response models
│   ├── agent/                # Agent orchestration, routing, compatibility
│   └── models/               # Base specialist interface definitions
├── frontend/                 # React + TypeScript + Vite interactive web application
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   └── src/
│       ├── App.tsx
│       ├── types/            # TypeScript schemas mirroring backend API
│       ├── services/         # API client targeting /api/query
│       └── components/       # Modular UI components (Trace, Evidence, QueryBar, etc.)
├── ml/                       # Machine learning data processing, adaptation, and models
│   ├── data/                 # BigEarthNet subset selection, patch extraction, preprocessing
│   ├── adaptation/           # Prithvi-EO-2.0 LoRA / parameter-efficient adaptation
│   └── models/               # Specialist model wrappers (VQA, Captioning, Change, Fusion)
├── docs/                     # Specifications and architectural documentation
│   ├── api_contract.md       # Shared frontend-backend API contract
│   ├── architecture.md       # Detailed system design and data flows
│   └── model_interfaces.md   # Python abstract interfaces for ML specialists
├── README.md
└── .gitignore
```

---

## 👥 Team Ownership & Responsibilities

| Team Member | Module / Area | Core Responsibilities |
| :--- | :--- | :--- |
| **Member 1** | **Backend + Agent + Integration** | FastAPI application, API schemas, `SatQueryController`, routing, compatibility checker, execution trace, final end-to-end integration. |
| **Member 2** | **Frontend (React + Vite)** | Interactive UI, multi-modal image uploader (single, optical+SAR, bi-temporal), evidence viewer, execution trace visualization. |
| **Member 3** | **BigEarthNet + Prithvi Adaptation** | BigEarthNet subsetting, S1/S2 patch extraction, Prithvi backbone setup, LoRA fine-tuning/adaptation pipeline. |
| **Member 4** | **VQA + Captioning Specialists** | Single-image remote-sensing VQA model and single-image captioning/grounding workflow. |
| **Member 5** | **Bi-temporal Change Specialist** | Change detection, change description, and time-series VQA workflows using dual-timestamp imagery. |
| **Member 6** | **Optical + SAR Fusion Specialist** | Multi-sensor joint analysis and optical-SAR feature fusion workflows. |

---

## 📚 Documentation Links
- [API Contract](file:///C:/Users/Lenovo/Desktop/satQuery-AI/docs/api_contract.md)
- [Architecture & Data Flows](file:///C:/Users/Lenovo/Desktop/satQuery-AI/docs/architecture.md)
- [Model Specialist Interfaces](file:///C:/Users/Lenovo/Desktop/satQuery-AI/docs/model_interfaces.md)
