---
description: Baseline OCR-Only Pipeline (No VLM Fallback)
---

# OCR Only Flow (`ocr_only`)

This workflow dictates the exact execution pipeline when `extraction_mode` inside `config.yaml` is set to `ocr_only`.

This is the **pure baseline** approach — PaddleOCR extracts text from the timesheet grid with **zero VLM involvement**. Empty cells remain empty, low-confidence results are accepted as-is. Useful for measuring the raw OCR capability and establishing a baseline for comparison against VLM-enhanced approaches.

## Architecture

```mermaid
graph TD
    A[Start: Read Timesheet PDF Page] --> B[Preprocess Image & Deskew]
    B --> C[Run PaddleOCR on Full Page]
    
    C --> D[Layout Detection Engine]
    D -->|Identify Header Zone| E[Extract Employee & Patient Names]
    D -->|Identify Table Row Boundaries| F[Slice Page into Logical Shifts]
    
    F --> G{For Each Time Field Column in Shift}
    
    G --> H{OCR Boxes Found in Cell Zone?}
    H -->|Yes| I[Concatenate OCR Text, Take Min Confidence]
    H -->|No| J[Leave Cell Empty]
    
    I --> K[Accept Result As-Is]
    J --> K
    
    K --> L[Run Parser on Extracted Text]
    L --> M[Create TimesheetRow Dictionary]
    
    M --> N[Pass Row to Validation Sandbox]
    N --> O[Export to Merged Excel File & Review Queue]
    
    style J fill:#ffebee,stroke:#c62828,color:#000
    style K fill:#f5f5f5,stroke:#616161,color:#000
```

## Key Characteristics

| Aspect | Behavior |
|--------|----------|
| VLM calls | **Zero** — no Ollama, no cloud API |
| Empty cells | Stay empty (no fallback) |
| Low confidence | Accepted as-is (no re-extraction) |
| Speed | **Fastest** — only PaddleOCR inference |
| Accuracy | **Lowest** — struggles with handwritten text |
| Best use | Baseline comparison, printed timesheets |

## Configuration

- **No VLM config needed** — Ollama and cloud VLM settings are ignored
- **Layout zones** — Same `layout:` configuration as `ppocr_grid`
- **Confidence thresholds** — Ignored (no routing decisions made)
- **Debug visualization** — Generates `ocr_only_` prefixed images showing OCR boxes only (no VLM fallback annotations)
