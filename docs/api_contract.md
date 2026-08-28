# SATQUERY AI — API Contract

> **Source of Truth for Frontend-Backend Communication**  
> All changes to API fields or parameters must be documented and agreed upon across backend and frontend owners.

---

## 📡 Endpoint: `POST /api/query`

Submits a multimodal remote sensing query along with one or more satellite images for analysis.

### 📥 Request Format

- **Content-Type**: `multipart/form-data` or `application/json` (with base64/file URIs)

```json
{
  "query": "Detect flood extent comparing T1 and T2 images",
  "input_type": "bitemporal",
  "image_primary": "<file_path_or_upload_id>",
  "image_secondary": "<file_path_or_upload_id_optional>"
}
```

#### Fields:
* `query` *(string, required)*: Natural language question or instruction.
* `input_type` *(string, required)*: Input configuration:
  * `"single"`: One optical, multispectral, or SAR image.
  * `"optical_sar"`: Primary optical image + secondary SAR image.
  * `"bitemporal"`: Time-1 image + Time-2 image.
* `image_primary` *(string / binary, required)*: Primary input image.
* `image_secondary` *(string / binary, optional)*: Secondary input image (required for `optical_sar` and `bitemporal`).

---

## 📤 Response Format

```json
{
  "answer": "Significant flooding detected along the northern river basin, covering an estimated 14.2 sq km.",
  "confidence": 0.89,
  "task": "bitemporal_change",
  "specialists": ["PrithviBackbone", "ChangeDetectionSpecialist"],
  "evidence": {
    "type": "heatmap",
    "data_url": "data:image/png;base64,...",
    "description": "Difference mask highlighting areas submerged in T2 compared to T1."
  },
  "execution_summary": {
    "steps": [
      {
        "step_name": "Input Validation",
        "status": "success",
        "detail": "Verified 2 GeoTIFF files for bitemporal alignment."
      },
      {
        "step_name": "Task Classification",
        "status": "success",
        "detail": "Classified query as bitemporal change analysis."
      },
      {
        "step_name": "Specialist Execution",
        "status": "success",
        "detail": "Executed ChangeDetectionSpecialist using adapted Prithvi backbone."
      }
    ],
    "total_duration_ms": 412
  }
}
```
