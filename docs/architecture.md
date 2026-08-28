# SATQUERY AI — System Architecture

## 1. Overview
SATQUERY AI is structured around a centralized agentic controller that inspects incoming user queries and remote sensing images, verifies input modality and dimension compatibility, determines the primary analytical task, routes execution to specialized models, and synthesizes answers, confidence metrics, and visual evidence with an auditable execution trace.

```
[User Interface (React+Vite)]
         │
         │ (HTTP REST)
         ▼
[FastAPI Router: /api/query]
         │
         ▼
[SatQueryController]
  ├── TaskClassifier ──────────────► Determines target task (VQA, Captioning, Change, Fusion)
  ├── CompatibilityChecker ────────► Validates single / optical+SAR / bitemporal alignment
  └── ModelRegistry ───────────────► Resolves specialist instances
         │
         ▼
[Shared Adapted Prithvi-EO-2.0 Backbone]
         │
  ┌──────┴───────────────┬──────────────────────┬──────────────────────┐
  ▼                      ▼                      ▼                      ▼
[Single VQA]    [Captioning/Grounding]    [Change Detection]    [Optical-SAR Fusion]
  └──────┬───────────────┴──────────────────────┴──────────────────────┘
         │
         ▼
[Response Synthesizer]
  ├── Natural Language Answer
  ├── Confidence Score
  ├── Visual Evidence (Heatmaps / Masks / Bounding Boxes)
  └── Auditable Execution Trace
```

## 2. ML Foundation Model Strategy
Instead of maintaining 4 disconnected, heavy training pipelines:
* **Prithvi-EO-2.0** serves as the **shared foundational remote sensing representation**.
* Adapted via parameter-efficient fine-tuning (LoRA) on a curated subset of **BigEarthNet** (Sentinel-1 and Sentinel-2 pairs).
* Task-specific specialists reuse or hook into the adapted Prithvi feature representations.
