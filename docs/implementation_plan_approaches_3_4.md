# Implementation Plan: Approaches 3 & 4

## Overview

This document outlines the implementation plan for two new extraction approaches:
- **Approach 3: OCR-Only** — Disable VLM fallback entirely, pure PP-OCRv5 pipeline
- **Approach 4: Layout-Guided VLM** — Use PP-DocLayoutV3 for table detection, then send cropped table to Qwen

---

## Approach 3: OCR-Only

### Concept

Run the existing `ppocr_grid` pipeline with VLM fallback completely disabled. No LLM calls, no network dependency. Pure PP-OCRv5 extraction.

### Changes Required

#### 1. Config Update (`src/config.py`)

Add a new extraction mode:

```python
class AppConfig(BaseModel):
    extraction_mode: str = "vlm_full_page"  # "ppocr_grid" | "vlm_full_page" | "ocr_only" | "layout_guided_vlm"
```

The `ocr_only` mode will route through the existing `ppocr_grid` code path but with fallback disabled.

#### 2. Pipeline Update (`src/pipeline.py`)

In `_extract_row()` and `_extract_transposed_row()`, add a check for `extraction_mode == "ocr_only"` that skips the VLM fallback loop entirely:

```python
# In _extract_row(), before the fallback loop:
if getattr(self.config, "extraction_mode", "ppocr_grid") == "ocr_only":
    # Skip all VLM fallback, use OCR results as-is
    row = self._build_timesheet_row(row_idx, cell_data, sources, field_bounds, expected_year)
    return row, 0, []
```

Alternatively, set the fallback threshold to 0.0 in config when `ocr_only` mode is active, so no cell ever triggers fallback.

#### 3. Benchmark Mode Label (`src/benchmark.py`)

Update the mode label mapping to include `"ocr_only"`:

```python
mode_label = {
    "ppocr_grid": "ppocr",
    "vlm_full_page": "vlm",
    "ocr_only": "ocr_only",
    "layout_guided_vlm": "layout_guided_vlm",
}.get(extraction_mode, extraction_mode)
```

#### 4. Tests

Add tests verifying:
- `ocr_only` mode produces zero VLM calls
- All fields sourced from PPOCR
- Pipeline completes without Ollama connectivity

### Effort: ~2-3 hours

---

## Approach 4: Layout-Guided VLM

### Concept

Replace the rule-based morphological layout detection with PP-DocLayoutV3 for table region detection. Crop the detected table and send it to the VLM for full extraction. Falls back to full-page VLM if table detection fails.

### Architecture

```
Preprocessed Image → PP-DocLayoutV3 → Table BBox → Crop → VLM (table crop) → Parse → Validate
                                                    ↓ (if no table detected)
                                              Full Page → VLM → Parse → Validate
```

### Changes Required

#### 1. Dependency (`pyproject.toml`)

Add PP-DocLayoutV3 dependency. Options:
- **PaddleX** (recommended): `pip install paddlex` — includes PP-DocLayoutV3 as a pretrained model
- **Direct PaddlePaddle**: Load the model weights directly from PaddlePaddle model zoo

```toml
dependencies = [
    # ... existing ...
    "paddlex>=3.0.0",  # for PP-DocLayoutV3
]
```

#### 2. New Module: `src/layout_model.py`

Create a new layout detection module wrapping PP-DocLayoutV3:

```python
class DocLayoutDetector:
    """PP-DocLayoutV3 wrapper for table region detection."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._model = None
        self._available = False

    def _ensure_initialized(self):
        if self._model is not None:
            return
        # Load PP-DocLayoutV3 model
        from paddlex import create_pipeline
        self._model = create_pipeline("PP-DocLayoutV3")
        self._available = True

    def detect_table(self, image: np.ndarray) -> Optional[BoundingBox]:
        """Detect the table/grid region in a timesheet image.

        Returns:
            BoundingBox (x1, y1, x2, y2) in pixel coordinates, or None if not detected.
        """
        if not self._ensure_initialized():
            return None

        result = self._model.predict(image)
        # Parse result to find table/grid region
        # PP-DocLayoutV3 returns layout elements with class labels
        # Look for "table" or "grid" class
        for element in result:
            if element.category in ("table", "grid", "figure"):
                return BoundingBox.from_bbox(element.bbox)
        return None
```

#### 3. New VLM Method: `extract_table_crop()` in `src/vlm_fallback.py`

Add a method similar to `extract_full_page()` but with a prompt optimized for table crops:

```python
def extract_table_crop(self, image: np.ndarray) -> dict:
    """Extract shifts from a cropped table image.

    The image contains only the timesheet grid (no headers, signatures, footers).
    """
    if not self._ensure_client():
        return {"shifts": [], "rn_lpn_name": ""}

    prompt = (
        "This image shows a handwritten timesheet grid. "
        "Extract all shift entries as a JSON array.\n\n"
        "Each row has: Date, Time In, Time Out, Total Hours.\n"
        "Return ONLY a JSON object:\n"
        '{\n'
        '  "shifts": [\n'
        '    {"date": "...", "time_in": "...", "time_out": "...", "total_hours": "..."}\n'
        '  ]\n'
        '}\n\n'
        "If a field is missing or illegible, use an empty string.\n"
        "CRITICAL: For times, strictly extract the exact text written (e.g., '8:00 AM', '4:30p').\n"
        "Do NOT convert to 24-hour format.\n"
        "CRITICAL: Only extract rows with visible handwritten content.\n"
        "Do not invent or guess missing data."
    )

    # Same image encoding as extract_full_page()
    # Same JSON parsing as extract_full_page()
    # Same anti-hallucination check (>7 rows → discard)
```

#### 4. Pipeline Update (`src/pipeline.py`)

Add a new branch in `_process_page()` for `extraction_mode == "layout_guided_vlm"`:

```python
if extraction_mode == "layout_guided_vlm":
    # Step 1: Detect table region with PP-DocLayoutV3
    layout_start = time_module.time()
    table_bbox = self.layout_detector.detect_table(image)
    page_metrics.layout_detection_time_s = time_module.time() - layout_start

    # Step 2: Crop table or fall back to full page
    if table_bbox is not None:
        table_crop = image[table_bbox.y1:table_bbox.y2, table_bbox.x1:table_bbox.x2]
        logger.info(f"Page {page_number}: Table detected, cropping to {table_bbox}")
    else:
        table_crop = image
        logger.warning(f"Page {page_number}: No table detected, using full page")

    # Step 3: VLM extraction on cropped table
    vlm_start = time_module.time()
    vlm_results = self.vlm.extract_table_crop(table_crop)
    page_metrics.vlm_inference_time_s = time_module.time() - vlm_start

    # Step 4: Parse and build rows (same as vlm_full_page)
    # ... reuse existing parsing logic ...
```

#### 5. Config Update (`src/config.py`)

Add `layout_guided_vlm` as a valid extraction mode:

```python
extraction_mode: str = "vlm_full_page"  # "ppocr_grid" | "vlm_full_page" | "ocr_only" | "layout_guided_vlm"
```

#### 6. Benchmark Update (`src/benchmark.py`)

Add timing metric for PP-DocLayoutV3 detection phase:

```python
class PageMetrics:
    doclayout_detection_time_s: float = 0.0  # NEW
```

#### 7. Tests

- Table detection on sample timesheet images
- Fallback to full page when no table detected
- VLM extraction on cropped table produces valid shifts
- End-to-end pipeline with `layout_guided_vlm` mode

### Effort: ~8-12 hours

---

## Implementation Order

1. **Approach 3 (OCR-Only)** — 2-3 hours
   - Simplest change, minimal code modifications
   - Can be done first to establish baseline metrics

2. **Approach 4 (Layout-Guided VLM)** — 8-12 hours
   - Requires new dependency (PP-DocLayoutV3/PaddleX)
   - New module, new VLM method, new pipeline branch
   - More complex testing and validation

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| PP-DocLayoutV3 doesn't detect timesheet tables well | Fallback to full-page VLM; consider fine-tuning on timesheet samples |
| PaddleX dependency conflicts with existing PaddlePaddle | Use separate virtual env or check version compatibility |
| Table crop removes important context (e.g., column headers) | Ensure crop includes header row; adjust bounding box with padding |
| VLM prompt needs tuning for cropped tables | A/B test prompts against full-page prompt on same samples |

---

## Benchmark Comparison Plan

Once all 4 approaches are implemented:

1. Run the same dataset through all 4 modes
2. Export combined benchmark (`benchmark_combined.xlsx`)
3. Compare:
   - Field-level accuracy (CER for text, exact match for dates/times)
   - Hours mismatch rate (calculated vs. written)
   - Processing time per page
   - VLM call count and cost
   - Hallucination rate
   - PHI exposure level
4. Generate statistical analysis for IEEE paper
