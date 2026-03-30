---
description: High-Speed Structured OCR Pipeline (PaddleOCR + VLM Fallback)
---

# High-Speed Structured OCR Flow (`ppocr_grid`)

This workflow dictates the exact execution pipeline when `extraction_mode` inside `config.yaml` is set to `ppocr_grid`. 

Because it uses deterministic pixel coordinates to slice the scanned PDF into strict grid rows and runs PaddleOCR over the distinct, isolated crops, it is extremely fast and scalable. Qwen2.5-VL is used purely as an intelligent fallback layer to re-read any illegible cells.

## ⚙️ Architecture

```mermaid
graph TD
    A[Start: Read Timesheet PDF Page] --> B[Preprocess Image & Deskew]
    B --> C[Layout Detection Engine]
    
    C -->|Identify Header Zone| D[Extract Employee & Patient Names]
    C -->|Identify Table Row Boundaries| E[Slice Page into Row Regions]
    
    E --> F[Run Global PaddleOCR Box Detection]
    F --> G{For Each Grid Cell Crop}
    
    G --> H[Calculate Confidence]
    H -->|Confidence > 0.60| I[Accept PP-OCR String Result]
    H -->|Confidence < 0.60| J[Crop Isolated Cell]
    
    J --> K[Pass Crop to Qwen2.5-VL via Ollama]
    K --> L[Extract Re-Parsed String e.g., '14:30']
    
    I & L --> M[Compile TimesheetRow Model]
    M --> N[Pass Row to Validation Sandbox]
    N --> O[Export to Merged Excel File]
```

## 🛠️ Configuration
Adjust the strict `layout: columns:` thresholds within `config.yaml` to ensure the bounding boxes correctly envelop your exact timesheet structure. If standard boundaries fail, the `vlm_full_page` mode must be enabled!
