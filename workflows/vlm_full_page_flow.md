---
description: Contextual Vision-LLM Full Page Extraction
---

# Deep Contextual Vision-LLM Flow (`vlm_full_page`)

This workflow dictates the exact execution logic when `extraction_mode` inside `config.yaml` is set to `vlm_full_page`.

Because it relies exclusively on Ollama hosting Qwen2.5-VL natively across the entire 1024px downsampled input image, the system completely bypasses PaddleOCR and rigid line coordination grids! It allows the Vision Model to deeply understand the structural intent of the document, even if it is a matrix form or cursive, and output perfectly structured JSON arrays mapping shifts directly to data fields.

## ⚙️ Architecture

```mermaid
graph TD
    A[Start: Read Timesheet PDF Page] --> B[Downsample to Max 1024px]
    B --> C[Convert to High-Q JPG Base64 String]
    
    C --> D[System Prompt Assembly]
    D --> E[Inject Strict Validation Directives]
    E --> F[API Request to qwen2.5-vl:7b]
    
    F -->|Raw Markdown Content| G[Regex JSON Extractor]
    G --> H[Convert to Python Dictionaries]
    
    H --> I{Does Shift Have Valid 'Time In' AND 'Time Out'?}
    I -->|Yes| J[Build TimesheetRow Object]
    I -->|No / Blank| K[Drop Extracted Ghost Shift]
    
    J & K --> L[Aggregate Row Indexes]
    L --> M[Pass Compiled Records to Validation]
    M --> N[Export Validated Row Statuses to Merged Excel File]
```

## 🛠️ Configuration
When using this mode, the parameters inside `layout` in `config.yaml` are strictly ignored. The system handles table logic dynamically through prompt instructions found within `vlm_fallback.py`.
