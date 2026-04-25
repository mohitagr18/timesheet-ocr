---
description: Layout-Guided VLM Pipeline (Cloud API)
---

# Layout-Guided VLM (Cloud) Flow (`layout_guided_vlm_cloud`)

This workflow dictates the exact execution pipeline when `extraction_mode` inside `config.yaml` is set to `layout_guided_vlm_cloud`.

This approach uses **PP-DocLayoutV3** to detect and crop the table zone from the timesheet, then sends the cropped table to a **cloud VLM API** (e.g., Google Gemini) for structured JSON extraction. Same layout detection as the local variant, but uses a cloud model for potentially higher accuracy.

## Architecture

```mermaid
graph TD
    A[Start: Read Timesheet PDF Page] --> B[Quick PaddleOCR Pass for Classification]
    B --> C{Signature Page? < 100 OCR boxes}
    
    C -->|Yes| D[Extract Employee Name from Footer]
    D --> E[Skip VLM, Return Empty Rows]
    
    C -->|No, Grid Page| F[Run PP-DocLayoutV3 Table Detection]
    F --> G{Table Zone Detected?}
    
    G -->|Yes| H[Crop Table Zone with Padding]
    G -->|No| I[Use Full Page as Fallback]
    
    H --> J[Send Cropped Table to Cloud VLM API]
    I --> J
    
    J --> K[Cloud VLM Returns JSON: shifts array]
    K --> L{Hallucination Check > 50 rows?}
    
    L -->|Yes| M[Discard, Return Empty]
    L -->|No| N[Parse Each Shift Row]
    
    N --> O[Disambiguate Time In/Out]
    O --> P[Create TimesheetRow Dictionary]
    
    P --> Q[Pass Row to Validation Sandbox]
    Q --> R[Export to Merged Excel File & Review Queue]
    
    style E fill:#fff8e1,stroke:#f9a825,color:#000
    style M fill:#ffebee,stroke:#c62828,color:#000
    style J fill:#e8f5e9,stroke:#2e7d32,color:#000
```

## Key Characteristics

| Aspect | Behavior |
|--------|----------|
| OCR role | Page classification only (grid vs signature) |
| Layout detection | PP-DocLayoutV3 detects table zone |
| VLM model | Cloud API (default: Google Gemini `gemini-2.5-flash`) |
| Input to VLM | Cropped table zone (with padding) |
| Anti-hallucination | Discards results with > 50 rows |
| Speed | Fast (cloud inference, no local model load) |
| Accuracy | **Highest** — cloud models have superior handwriting recognition |
| Best use | Maximum accuracy, API key available, no privacy constraints |

## Configuration

- **`cloud_vlm.provider`** — Cloud provider (default: `google`)
- **`cloud_vlm.model`** — Cloud model name (default: `gemini-2.5-flash`)
- **`cloud_vlm.api_key_env`** — Environment variable name for API key
- **`cloud_vlm.timeout_seconds`** — Max wait time per API call
- **`GOOGLE_API_KEY`** — Must be set in `.env` or environment
- **Debug visualization** — Generates `vlm_` prefixed images with extracted text annotations
