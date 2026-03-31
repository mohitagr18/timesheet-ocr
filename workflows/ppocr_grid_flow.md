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
    C -->|Identify Table Row Boundaries| E[Slice Page into Logical Shifts]
    
    E --> F[Run Global PaddleOCR Box Detection]
    F --> G{For Each Time Field Column in Shift}
    
    G --> H{Cell Blank / Too Long \n or low confidence?}
    H -->|No| I[Accept Raw PP-OCR String Result]
    H -->|Yes| J[Crop Origin Un-Binarized Color Cell]
    
    J --> K[Re-Pass Isolated Crop to PP-OCR]
    K -->|Fails Base Confidence| L[Pass Crop to Qwen2.5-VL via Ollama]
    K -->|Passes| I
    
    L --> I
    I --> M[Run Aggressive Parser]
    
    %% Parser logic
    M -->|Regex Handlers| N[Clear Artifacts e.g., 'Q'->'2']
    N --> O[Create TimesheetRow Dictionary]
    
    O --> P[Pass Row to Validation Sandbox]
    P --> Q[Export to Merged Excel File & Review Queue]
```

## 🛠️ Configuration & Adjustments

- **Strict Coordinates**: Adjust the `layout: columns:` or `table_zone:` X/Y fractions within `config.yaml` to ensure the bounding boxes correctly envelop your exact timesheet structure.
- **Un-Binarized Cropping**: By operating natively on the origin color image rather than relying on standard black-and-white (binarized/denoised) images, localized cell cropping preserves critical pen-pressure and ink-hue constraints to maximize the PP-OCR fallback capability.
- **Aggressive Time Regex Parsing**: `src/parser.py` scrubs known trailing PaddleOCR noise (like `/20` strings) and directly translates erroneous mapping digits (`Q:` to `2:`) directly within the pipeline flow.

If a provided home-health template is heavily cursive, or entirely skips standard boundaries, the `vlm_full_page` mode must be enabled!
