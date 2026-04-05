---
description: Layout-Guided VLM Pipeline (Local Ollama Model)
---

# Layout-Guided VLM (Local) Flow (`layout_guided_vlm_local`)

This workflow dictates the exact execution pipeline when `extraction_mode` inside `config.yaml` is set to `layout_guided_vlm_local`.

This approach uses **PP-DocLayoutV3** to detect and crop the table zone from the timesheet, then sends the cropped table to a **local VLM** (via Ollama) for structured JSON extraction. No PaddleOCR parsing is used — the VLM reads the entire cropped table and returns structured shift data.

## Architecture

```mermaid
graph TD
    A[Start: Read Timesheet PDF Page] --> B[Quick PaddleOCR Pass for Classification]
    B --> C{Signature Page?\n< 100 OCR boxes}
    
    C -->|Yes| D[Extract Employee Name from Footer]
    D --> E[Skip VLM, Return Empty Rows]
    
    C -->|No, Grid Page| F[Run PP-DocLayoutV3 Table Detection]
    F --> G{Table Zone Detected?}
    
    G -->|Yes| H[Crop Table Zone with Padding]
    G -->|No| I[Use Full Page as Fallback]
    
    H --> J[Send Cropped Table to Local VLM via Ollama]
    I --> J
    
    J --> K[VLM Returns JSON: shifts array]
    K --> L{Hallucination Check\n> 7 rows?}
    
    L -->|Yes| M[Discard, Return Empty]
    L -->|No| N[Parse Each Shift Row]
    
    N --> O[Disambiguate Time In/Out]
    O --> P[Create TimesheetRow Dictionary]
    
    P --> Q[Pass Row to Validation Sandbox]
    Q --> R[Export to Merged Excel File & Review Queue]
    
    style E fill:#fff8e1,stroke:#f9a825,color:#000
    style M fill:#ffebee,stroke:#c62828,color:#000
```

## Key Characteristics

| Aspect | Behavior |
|--------|----------|
| OCR role | Page classification only (grid vs signature) |
| Layout detection | PP-DocLayoutV3 detects table zone |
| VLM model | Local via Ollama (e.g., `qwen2.5vl:7b`) |
| Input to VLM | Cropped table zone (with padding) |
| Anti-hallucination | Discards results with > 7 rows |
| Speed | Moderate (layout detection + local VLM) |
| Accuracy | High (VLM reads full table context) |
| Best use | Privacy-first deployment, no cloud API needed |

## Configuration

- **`ollama.model`** — Local VLM model name (default: `qwen2.5vl:7b`)
- **`ollama.host`** — Ollama server URL (default: `http://localhost:11434`)
- **`ollama.timeout_seconds`** — Max wait time per VLM call
- **`layout.table_zone`** — Fallback if PP-DocLayoutV3 fails to detect table
- **Debug visualization** — Generates `vlm_` prefixed images with extracted text annotations
